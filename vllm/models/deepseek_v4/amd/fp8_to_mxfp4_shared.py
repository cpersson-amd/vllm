# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time FP8(block-scale) -> MXFP4 re-quant for DSv4 shared experts.

DeepSeek-V4-Flash routes experts in MXFP4 but stores the always-on shared
expert as FP8 (E4M3 + 128x128 block scale). To fold the shared expert into
the routed grouped-GEMM, weights are converted to the MXFP4 checkpoint layout
expected by routed expert slots, then loaded into slots
``[n_routed .. n_routed+n_shared)``.
"""

from __future__ import annotations

import os

import torch

# MXFP4 (E2M1) code -> value lookup, matching the ordering in
# ``aiter.utility.fp4_utils.mxfp4_to_f32`` (indices 0-7 positive, 8-15 negative).
_MXFP4_CODE_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
# Largest representable MXFP4 magnitude and the fixed OCP micro-block size.
_MXFP4_MAX = 6.0
_MXFP4_BLOCK = 32

# Number of extra (smaller) E8M0 exponents to try *below* the no-clip scale
# during the load-time re-quant MSE search. ``0`` reproduces the original
# absmax RTN behaviour of ``dynamic_mxfp4_quant`` exactly. Overridable via
# ``VLLM_DSV4_SHARED_MXFP4_SCALE_SEARCH`` for easy A/B on GSM8K.
_DEFAULT_SCALE_SEARCH = int(
    os.getenv("VLLM_DSV4_SHARED_MXFP4_SCALE_SEARCH", "3")
)


def dequant_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize a DeepSeek block-quantized FP8 weight to bf16."""
    assert weight.dim() == 2, f"expected 2D weight, got {tuple(weight.shape)}"
    out, inn = weight.shape
    bn, bk = block
    w = weight.to(torch.float32)
    scale = weight_scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0)[:out]
    scale = scale.repeat_interleave(bk, dim=1)[:, :inn]
    return (w * scale).to(torch.bfloat16)


def quant_bf16_to_mxfp4(
    w_bf16: torch.Tensor,
    scale_search: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [out, in] -> (packed_uint8 [out, in//2], scale_uint8 [out, in//32]).

    By default this performs a per-32-block MSE-optimal E8M0 *clipping* search
    (see :func:`_mxfp4_quant_mse_search`) instead of plain absmax round-to-
    nearest. Fitting each block's power-of-two scale to the block max wastes
    MXFP4's 16-level grid whenever a single outlier dominates the block;
    clipping that outlier slightly buys mantissa resolution for the other 31
    weights and materially reduces the FP8->MXFP4 re-quant error on the DSv4
    shared expert.

    Set ``scale_search=0`` (or ``VLLM_DSV4_SHARED_MXFP4_SCALE_SEARCH=0``) to
    fall back to the original ``dynamic_mxfp4_quant`` absmax path, which emits
    a byte-identical packed/scale layout.
    """
    assert w_bf16.dim() == 2
    if scale_search is None:
        scale_search = _DEFAULT_SCALE_SEARCH

    orig_device = w_bf16.device
    w = w_bf16.contiguous()
    if w.device.type != "cuda":
        w = w.cuda()

    if scale_search <= 0:
        from aiter.utility.fp4_utils import dynamic_mxfp4_quant

        packed, scale = dynamic_mxfp4_quant(w, shuffle=False)
        packed = packed.view(torch.uint8).to(orig_device).contiguous()
        scale = scale.view(torch.uint8).to(orig_device).contiguous()
        return packed, scale

    packed, scale = _mxfp4_quant_mse_search(w, num_lower=scale_search, rtol=1e-6)
    return packed.to(orig_device).contiguous(), scale.to(orig_device).contiguous()


def _mxfp4_quant_mse_search(
    w: torch.Tensor,
    num_lower: int,
    rtol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE-optimal MXFP4 re-quant with a per-block E8M0 clipping search.

    Produces the *exact* same packed/scale layout as
    ``dynamic_mxfp4_quant(w, shuffle=False)`` -- it reuses aiter's own
    ``_f32_to_floatx_unpacked`` (identical E2M1 round-half-to-even) and
    ``pack_uint4`` (identical nibble order), so the only difference from the
    stock kernel is *which* power-of-two scale each 32-element block gets.

    For every block we evaluate the no-clip scale ``e0 = ceil(log2(amax/6))``
    plus ``num_lower`` progressively smaller (more aggressively clipping)
    exponents, dequantize each candidate, and keep the one with the lowest
    reconstruction SSE. All work is vectorized over ``[out, num_blocks]``;
    candidates are looped to bound peak memory.
    """
    from aiter.utility.fp4_utils import _f32_to_floatx_unpacked, pack_uint4

    blk = _MXFP4_BLOCK
    out, inn = w.shape
    assert inn % blk == 0, f"in dim {inn} not a multiple of {blk}"
    nb = inn // blk

    x = w.to(torch.float32)
    xb = x.reshape(out, nb, blk)  # [out, nb, 32]
    lut = torch.tensor(_MXFP4_CODE_VALUES, dtype=torch.float32, device=x.device)

    amax = xb.abs().amax(dim=-1)  # [out, nb]
    # No-clip exponent e0 = ceil(log2(amax / 6)), guarded so amax / 2^e0 <= 6
    # even under float rounding at exact powers of two.
    ratio = amax / _MXFP4_MAX
    log2r = torch.where(amax > 0, torch.log2(ratio), torch.zeros_like(ratio))
    e0 = torch.ceil(log2r)
    e0 = torch.where(amax > _MXFP4_MAX * torch.exp2(e0), e0 + 1.0, e0)
    e0 = e0.clamp(-127.0, 127.0)

    best_sse = torch.full((out, nb), float("inf"), dtype=torch.float64,
                          device=x.device)
    best_codes = torch.zeros((out, nb, blk), dtype=torch.uint8, device=x.device)
    best_e = torch.zeros((out, nb), dtype=torch.float32, device=x.device)

    for k in range(num_lower + 1):
        ec = (e0 - float(k)).clamp(-127.0, 127.0)  # [out, nb]
        scale = torch.exp2(ec).unsqueeze(-1)  # [out, nb, 1]
        scaled = (xb / scale).to(torch.float32)
        codes = _f32_to_floatx_unpacked(scaled.reshape(-1).contiguous(), 2, 1)
        codes = codes.reshape(out, nb, blk)
        deq = lut[codes.long()] * scale  # dequantized bf16-domain reconstruction
        # float64 reduction: block is 32 elements so this is ~exact and stable.
        sse = ((xb - deq).double() ** 2).sum(dim=-1)  # [out, nb]
        improve = sse < best_sse * (1.0 - rtol)
        best_sse = torch.where(improve, sse, best_sse)
        best_e = torch.where(improve, ec, best_e)
        best_codes = torch.where(improve.unsqueeze(-1), codes, best_codes)

    packed = pack_uint4(best_codes.reshape(out, inn).contiguous())  # [out, in//2]
    scale_e8m0 = (best_e.to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    return packed.contiguous(), scale_e8m0.contiguous()


def convert_shared_fp8_to_mxfp4(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 block-scale [out, in] -> MXFP4 (packed uint8, scale uint8)."""
    return quant_bf16_to_mxfp4(dequant_fp8_block(weight, weight_scale_inv, block))
