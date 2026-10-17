# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer on aiter's paged MXFP4 MQA-logits kernel.

gfx950 only. The kernel reads the preshuffled paged indexer K cache in place,
so no layer gathers K into a contiguous buffer, prefill included. Any model
whose indexer supplies the inputs `rocm_mxfp4_sparse_attn_indexer` documents
can run the dense path. DeepSeek-V4.1's two-level indexer adds to it: the
candidate source takes its block maxima from the same walk that writes its
logits, and the candidate consumers can walk the candidate pool instead of
the whole context: the pool is resolved once per step and shared by all of
them.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    _mask_candidates_kernel,
    apply_candidate_mask,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_candidate_mask_strided,
    _max_decode_logits_rows,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        RocmMxfp4GatherLaunch,
        RocmMxfp4IndexerMetadata,
        RocmMxfp4NativeDecode,
        RocmMxfp4PrefillPlan,
    )

MXFP4_BLOCK_SIZE = 32


@functools.cache
def _aiter():
    """The aiter module with the paged MXFP4 MQA-logits launcher, its schedule
    and cache_format."""
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4

    return pa_mqa_logits_mxfp4


@functools.cache
def _aiter_topk() -> Callable[..., None]:
    """The aiter per-row top-k."""
    from aiter.ops.topk import top_k_per_row_decode

    return top_k_per_row_decode


class RocmPagedMxfp4CacheLayout(NamedTuple):
    """The byte order of an indexer K page, as the kernel reads it, and the
    shuffle pattern aiter's K cache op writes it with.

    A page is cut into runs of ``n_per_tile`` tokens. Inside a run, each token's
    packed values are split into ``d_per_tile``-byte chunks along the head
    dimension, and the run stores chunk 0 of every token, then chunk 1, and so
    on: ``[chunk, token, byte]``. Its e8m0 scales are stored as
    ``[scale % scale_lanes, token, scale // scale_lanes]``.
    """

    n_per_tile: int
    d_per_tile: int
    scale_lanes: int


# The scale order the kernel reads: aiter's mode 1, a lane's scales innermost.
# cache_format does not report the order or the lane split yet.
_SCALE_ORDER = 1
_WAVE_SIZE = 64


@functools.cache
def rocm_paged_mxfp4_cache_layout(
    num_heads: int, head_dim: int, page_entries: int
) -> RocmPagedMxfp4CacheLayout:
    """The K page layout from aiter's ``cache_format``, the only place vLLM
    reads it, so a change there cannot reach the writer unchecked.

    Raises:
        ValueError: aiter rejects the page size, or describes an order this
            layout cannot express.

    """
    fmt = _aiter().cache_format(num_heads, head_dim, page_entries)
    try:
        n_per_tile = int(fmt["n_per_tile"])
        d_per_tile = int(fmt["d_per_tile"])
        scale_order = fmt.get("scale_mode", _SCALE_ORDER)
        scale_lanes = int(fmt.get("scale_lanes", _WAVE_SIZE // n_per_tile))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"aiter's cache_format() returned {fmt!r}") from e
    if (
        scale_order != _SCALE_ORDER
        or min(n_per_tile, d_per_tile, scale_lanes) <= 0
        or page_entries % n_per_tile
        or (head_dim // 2) % d_per_tile
        or (head_dim // MXFP4_BLOCK_SIZE) % scale_lanes
    ):
        raise ValueError(
            f"aiter's cache_format() returned {fmt!r} for {num_heads} heads of "
            f"{head_dim} in {page_entries}-entry pages, a K page order "
            "RocmPagedMxfp4CacheLayout cannot express"
        )
    return RocmPagedMxfp4CacheLayout(n_per_tile, d_per_tile, scale_lanes)


def rocm_mxfp4_decode_schedule_words(
    num_heads: int, head_dim: int, page_entries: int, next_n: int = 1
) -> int:
    """int32 words of the largest schedule a decode step can build, flattened or
    not. The slice cap can take it past target_wgs, up to SCHED_SLOT_CAP."""
    words = 4 * _aiter().SCHED_SLOT_CAP
    for rows in {1, next_n}:
        config = _aiter().select_config(num_heads, head_dim, rows, page_entries)
        words = max(words, 4 * config["target_wgs"])
    return words


def build_rocm_mxfp4_decode_schedule(
    row_lens: torch.Tensor,
    num_heads: int,
    head_dim: int,
    page_entries: int,
    out: torch.Tensor,
    logits_width: int,
    native: "RocmMxfp4NativeDecode | None" = None,
) -> torch.Tensor | None:
    """Work descriptors that even out a decode step, or None where the static
    grid already fills the machine. Depends only on the rows' lengths, the
    cache geometry and the logits width, so one serves every layer of a group.
    The width sizes the slices, capped the way the static grid caps them."""
    if native is None:
        return _aiter().build_schedule(
            row_lens,
            1,
            num_heads,
            head_dim,
            page_entries,
            out=out,
            max_model_len=logits_width,
        )
    return _aiter().build_schedule(
        native.context_lens,
        native.next_n,
        num_heads,
        head_dim,
        page_entries,
        out=out,
        row_ends=row_lens,
        max_model_len=logits_width,
    )


def _kv_view(kv_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    """The indexer cache as [pages, entries, 1, bytes]. The page stride is the
    block-major pool's, not the page's own size."""
    num_pages, entries, width = kv_cache.shape
    assert kv_cache.dtype == torch.uint8
    assert width == head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE, width
    return torch.as_strided(
        kv_cache, (num_pages, entries, 1, width), (kv_cache.stride(0), width, width, 1)
    )


def rocm_mxfp4_consumer_rows(num_candidate_cols: int) -> int:
    """Query rows per candidate-consumer launch. Its logits are [rows, pool]
    fp32 whatever the context, so the logits budget alone sizes it."""
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    return max(1, budget // (4 * num_candidate_cols))


def reserve_rocm_mxfp4_indexer_workspace(
    hidden_states: torch.Tensor,
    logits_width: int,
    candidate_block_size: int = 0,
    gather_block_size: int = 0,
    num_candidate_cols: int = 0,
) -> None:
    """Profiling run: claim the decode logits workspace and the peak prefill
    logits, block scores included when the layer writes them, candidate lists
    when it gathers."""
    rows = _max_decode_logits_rows(hidden_states.shape[0])
    specs = [((rows, logits_width), torch.float32)]
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    if candidate_block_size:
        nblocks = triton.cdiv(logits_width, candidate_block_size)
        specs.append(((rows, nblocks), torch.float32))
        budget += budget // candidate_block_size
    if gather_block_size:
        # aiter allocates the candidate lists: each gathered row, 8 B per
        # candidate block (int32 slot and position). The first consumer builds
        # them and they live until the step ends, across the later layers, so
        # one reservation is held for the rest of the forward.
        held = get_forward_context().additional_kwargs
        if "rocm_mxfp4_candidate_lists" not in held:
            held["rocm_mxfp4_candidate_lists"] = torch.empty(
                hidden_states.shape[0] * (num_candidate_cols // gather_block_size) * 8,
                dtype=torch.uint8,
                device=hidden_states.device,
            )
    current_workspace_manager().get_simultaneous(*specs)
    torch.empty(budget, dtype=torch.uint8, device=hidden_states.device)


def _topk(
    logits: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor, k: int
) -> None:
    """Top-k over each row's [0, lengths[row]); -1 pads rows shorter than k.

    aiter's kernel is faster on the candidate top-k and from 128 rows on; below
    that, a wide decode step is faster on vLLM's. Shape alone decides, so a FULL
    graph replays the same choice at every context length.
    """
    rows, width = logits.shape
    if rows == 0:
        return
    if width >= k and (k >= 2048 or rows >= 128):
        _aiter_topk()(
            logits,
            1,
            lengths.reshape(-1),
            out,
            rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
        )
    else:
        torch.ops._C.top_k_per_row_decode(
            logits, 1, lengths, out, rows, logits.stride(0), logits.stride(1), k
        )


@triton.jit
def _remap_compact_topk_kernel(
    idx_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PADDED_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, PADDED_K)
    live = cols < K
    slot = tl.load(idx_ptr + row * idx_stride + cols, mask=live, other=-1)
    hit = slot >= 0
    slot = tl.where(hit, slot, 0)
    start = tl.load(
        pos_ptr + row * pos_stride + slot // BLOCK, mask=live & hit, other=0
    )
    tl.store(
        idx_ptr + row * idx_stride + cols,
        tl.where(hit, start + slot % BLOCK, -1).to(tl.int32),
        mask=live,
    )


def _remap_compact_topk(
    indices: torch.Tensor, positions: torch.Tensor, block: int
) -> None:
    """Candidate slots to context positions, in place: slot j sits in pool
    block j // block, which starts at positions[j // block]."""
    rows, k = indices.shape
    if rows == 0:
        return
    _remap_compact_topk_kernel[(rows,)](
        indices,
        indices.stride(0),
        positions,
        positions.stride(0),
        K=k,
        BLOCK=block,
        PADDED_K=triton.next_power_of_2(k),
        num_warps=4,
    )


@dataclass
class _Layer:
    """One indexer layer's inputs and outputs for this step."""

    metadata: "RocmMxfp4IndexerMetadata"
    kv: torch.Tensor
    q: torch.Tensor
    q_scale: torch.Tensor
    weights: torch.Tensor
    topk_buffer: torch.Tensor
    topk_tokens: int
    compress_ratio: int
    candidates: torch.Tensor | None
    block: int
    full_graph: bool

    @property
    def num_heads(self) -> int:
        return self.q.shape[1]

    @property
    def head_dim(self) -> int:
        return self.q.shape[2] * 2

    def rows(self, lo: int, hi: int, seqs: int = 1) -> tuple[torch.Tensor, ...]:
        """Q, its scales and the weights of token rows [lo, hi), as ``seqs``
        sequences of next_n rows each."""
        return (
            self.q[lo:hi].view(seqs, -1, *self.q.shape[1:]),
            self.q_scale[lo:hi].view(seqs, -1, *self.q_scale.shape[1:]),
            self.weights[lo:hi],
        )

    def decode_rows(self, n: int) -> tuple[torch.Tensor, ...]:
        """The first n token rows, one sequence each."""
        return (
            self.q[:n].unsqueeze(1),
            self.q_scale[:n].unsqueeze(1),
            self.weights[:n],
        )


def _layer(
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    compress_ratio: int,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
) -> _Layer:
    from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
        RocmMxfp4IndexerMetadata,
    )

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    assert isinstance(attn_metadata, dict)
    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, RocmMxfp4IndexerMetadata), (
        "the ROCm MXFP4 indexer needs RocmMxfp4IndexerMetadataBuilder metadata"
    )
    kv = _kv_view(kv_cache, head_dim)
    num_heads = q_values.shape[1]
    return _Layer(
        metadata=metadata,
        kv=kv,
        q=q_values,
        # The fused Q kernel returns one int32 per head; the kernel reads the
        # four ue8m0 bytes behind it.
        q_scale=q_scale.view(torch.uint8).view(q_scale.shape[0], num_heads, -1),
        weights=weights,
        topk_buffer=topk_indices_buffer,
        topk_tokens=topk_tokens,
        compress_ratio=compress_ratio,
        candidates=candidate_blocks,
        block=candidate_block_size,
        full_graph=forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL,
    )


def _block_scores(layer: _Layer, scores: torch.Tensor | None) -> dict:
    if scores is None:
        return {}
    return dict(
        calc_block_scores=True, block_scores=scores, candidate_block_size=layer.block
    )


def _dense_prefill(
    layer: _Layer,
    chunk: "DeepseekV32IndexerPrefillChunkMetadata",
    plan: "RocmMxfp4PrefillPlan",
    candidate_write: bool,
) -> None:
    pa = _aiter()
    t0, t1 = chunk.token_start, chunk.token_end
    # A row stride that is a multiple of 16 whatever the chunk's width keeps
    # the candidate mask to the one specialization its warmup compiles.
    logits = layer.q.new_empty(
        (t1 - t0, round_up(plan.width, 16)), dtype=torch.float32
    )[:, : plan.width]
    scores = None
    if candidate_write:
        nblocks = triton.cdiv(plan.width, layer.block)
        scores = layer.q.new_empty((t1 - t0, nblocks), dtype=torch.float32)
    # One launch for the chunk: the kernel finds each row's request from
    # query_start_loc on its own grid.
    pa.paged_mxfp4_mqa_logits(
        layer.q[t0:t1],
        layer.q_scale[t0:t1],
        layer.kv,
        layer.weights[t0:t1],
        plan.context_lens,
        chunk.block_table,
        plan.width,
        out_logits=logits,
        clean_logits=False,
        row_ends=plan.row_ends,
        query_start_loc=plan.query_start_loc,
        **_block_scores(layer, scores),
    )
    candidates = None if layer.candidates is None else layer.candidates[t0:t1]
    if scores is not None:
        assert candidates is not None and plan.block_ends is not None
        _topk(scores, plan.block_ends, candidates, candidates.shape[1])
    elif candidates is not None:
        apply_candidate_mask(logits, None, plan.row_ends, candidates, layer.block)
    _topk(
        logits,
        plan.row_ends,
        layer.topk_buffer[t0:t1, : layer.topk_tokens],
        layer.topk_tokens,
    )


# The select_config fields the kernel compiles in; the rest only plan the grid.
_PREFILL_CONFIG_FIELDS = (
    "num_warps",
    "num_buffers",
    "waves_per_eu",
    "depth",
    "unroll",
    "fold_asm",
    "relu_add",
    "relaxed_store",
    "block_m",
    "block_kv",
    "n_per_tile",
    "page_pipe",
    "m_chunk",
    "num_chains",
)


class RocmMxfp4PrefillLogitsKernel(
    VllmJitKernel["RocmMxfp4PrefillLogitsKernel.CompileKey"]
):
    """Compiles aiter's paged MXFP4 MQA-logits kernel for every varlen launch
    `_dense_prefill` can make on one cache. The launch plans its config from a
    chunk's mean rows per request, and splits the KV walk or not depending on
    the batch, so both are swept."""

    @dataclass(frozen=True)
    class CompileKey:
        num_heads: int
        head_dim: int
        page_size: int
        page_stride: int
        cache_within_2gb: bool
        block_table_width: int
        block_scores: bool
        candidate_block_size: int
        config: tuple[tuple[str, int], ...]
        kv_reread: bool
        has_kv_split: bool
        # A mean rows per request that selects config.
        next_n: int = field(compare=False)

    def get_warmup_keys(
        self,
        *,
        vllm_config: VllmConfig,
        k_cache_prefix: str,
        num_heads: int,
        head_dim: int,
        block_table_width: int,
        max_tokens: int,
        candidate_block_size: int,
    ) -> list[CompileKey]:
        pa = _aiter()
        forward_context = vllm_config.compilation_config.static_forward_context
        kv_cache = forward_context[k_cache_prefix].kv_cache
        page_size, page_stride, _ = pa.cache_strides(
            _kv_view(kv_cache, head_dim), head_dim
        )
        cache_within_2gb = kv_cache.untyped_storage().size() < 2**31
        keys: dict[RocmMxfp4PrefillLogitsKernel.CompileKey, None] = {}
        for next_n in range(1, max_tokens + 1):
            cfg = pa.select_config(
                num_heads, head_dim, next_n, page_size, clean_logits=False
            )
            config = tuple((name, cfg[name]) for name in _PREFILL_CONFIG_FIELDS)
            for block_scores in {False, bool(candidate_block_size)}:
                for has_kv_split in (False, True):
                    key = self.CompileKey(
                        num_heads,
                        head_dim,
                        page_size,
                        page_stride,
                        cache_within_2gb,
                        block_table_width,
                        block_scores,
                        candidate_block_size,
                        config,
                        cfg["row_blocks"] > 1,
                        has_kv_split,
                        next_n,
                    )
                    keys.setdefault(key, None)
        return list(keys)

    def compile(self, compile_key: CompileKey) -> None:
        from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_mqa_logits_mxfp4 import (  # noqa: E501
            _pa_mqa_logits_mxfp4_kernel,
        )

        key = compile_key
        cfg = dict(key.config)
        rows, heads, head_dim = key.next_n, key.num_heads, key.head_dim
        u8, i32, f32 = torch.uint8, torch.int32, torch.float32
        q = TritonWarmupTensor(u8, shape=(rows, heads, head_dim // 2))
        q_scales = TritonWarmupTensor(
            u8, shape=(rows, heads, head_dim // MXFP4_BLOCK_SIZE)
        )
        storage_bytes = 0 if key.cache_within_2gb else 2**31
        head_bytes = head_dim // 2
        values = TritonWarmupTensor(
            u8,
            shape=(1, key.page_size * head_bytes),
            strides=(key.page_stride, 1),
            storage_bytes=storage_bytes,
        )
        scales = TritonWarmupTensor(
            u8,
            shape=(1, key.page_size * head_dim // MXFP4_BLOCK_SIZE),
            strides=(key.page_stride, 1),
            storage_bytes=storage_bytes,
        )
        weights = TritonWarmupTensor(f32, shape=(rows, heads))
        context_lens = TritonWarmupTensor(i32, shape=(1,))
        block_table = TritonWarmupTensor(i32, shape=(1, key.block_table_width))
        logits = TritonWarmupTensor(f32, shape=(rows, 16))
        block_scores = (
            TritonWarmupTensor(f32, shape=(rows, 16)) if key.block_scores else None
        )
        _pa_mqa_logits_mxfp4_kernel.warmup(
            Q_ptr=q,
            q_scales_ptr=q_scales,
            KV_ptr=values,
            kv_scales_ptr=scales,
            weights_ptr=weights,
            context_lens_ptr=context_lens,
            row_ends_ptr=TritonWarmupTensor(i32, shape=(rows,)),
            query_start_loc_ptr=TritonWarmupTensor(i32, shape=(2,)),
            block_table_ptr=block_table,
            sched_ptr=context_lens,
            slots_ptr=context_lens,
            block_scores_ptr=block_scores,
            logits_ptr=logits,
            next_n=rows,
            batch=1,
            num_kv_splits=2 if key.has_kv_split else 1,
            stride_q_b=0,
            stride_q_n=q.stride(0),
            stride_qs_b=0,
            stride_qs_n=q_scales.stride(0),
            stride_w_s=weights.stride(0),
            stride_logits_s=logits.stride(0),
            stride_logits_k=logits.stride(1),
            stride_blk_b=block_table.stride(0),
            stride_gather_r=0,
            stride_bs_s=block_scores.stride(0) if block_scores is not None else None,
            max_blocks=key.block_table_width,
            NUM_HEADS=heads,
            HEAD_SIZE=head_dim,
            PAGE_SIZE=key.page_size,
            KV_PAGE_STRIDE=key.page_stride,
            KVS_PAGE_STRIDE=key.page_stride,
            BLOCK_KV=cfg["block_kv"],
            BLOCK_M=cfg["block_m"],
            NUM_WARPS=cfg["num_warps"],
            NUM_BUFFERS=cfg["num_buffers"],
            DEPTH=cfg["depth"],
            UNROLL=cfg["unroll"],
            PAGE_PIPE=cfg["page_pipe"],
            M_CHUNK=cfg["m_chunk"],
            NUM_CHAINS=cfg["num_chains"],
            FOLD_ASM=cfg["fold_asm"],
            RELU_ADD=cfg["relu_add"],
            RELAXED_STORE=cfg["relaxed_store"],
            VARLEN=1,
            PRESHUFFLE=1,
            SCALE_MODE=_aiter().SCALE_MODE_WIDE,
            USE_BUFFER_LOAD=True,
            MFMA_NONK_DIM=cfg["n_per_tile"],
            HAS_KV_SPLIT=int(key.has_kv_split),
            KV_REREAD=int(key.kv_reread),
            DYNAMIC=0,
            HAS_ROW_ENDS=1,
            GATHER=0,
            GATHER_BLOCK=8,
            GATHER_PIPE=0,
            BSCORE=int(key.block_scores),
            BSCORE_BLOCK=(
                key.candidate_block_size
                if key.block_scores
                else _aiter().CANDIDATE_BLOCK
            ),
            num_warps=cfg["num_warps"],
            waves_per_eu=cfg["waves_per_eu"],
            enable_fp_fusion=True,
            grid=(1, 1, 1),
        )


ROCM_MXFP4_PREFILL_LOGITS_KERNEL = RocmMxfp4PrefillLogitsKernel()


class RocmMxfp4PrefillMaskKernel(
    VllmJitKernel["RocmMxfp4PrefillMaskKernel.CompileKey"]
):
    """Compiles the candidate mask a consumer's dense prefill applies to its
    logits."""

    @dataclass(frozen=True)
    class CompileKey:
        candidate_block_size: int

    def get_warmup_keys(self, *, candidate_block_size: int) -> list[CompileKey]:
        return [self.CompileKey(candidate_block_size)]

    def compile(self, compile_key: CompileKey) -> None:
        logits = TritonWarmupTensor(torch.float32, shape=(1, 16))
        _mask_candidates_kernel.warmup(
            logits,
            None,
            TritonWarmupTensor(torch.int32, shape=(1,)),
            TritonWarmupTensor(torch.uint8, shape=(1, 3)),
            logits.stride(0),
            logits.stride(1),
            0,
            1,
            16,
            2,
            compile_key.candidate_block_size,
            False,
            1,
            1024,
            grid=(1, 1),
        )


ROCM_MXFP4_PREFILL_MASK_KERNEL = RocmMxfp4PrefillMaskKernel()


def _dense_decode(layer: _Layer, logits_width: int, candidate_write: bool) -> None:
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    rows = lengths.shape[0]
    specs = [((rows, logits_width), torch.float32)]
    if candidate_write:
        specs.append(((rows, triton.cdiv(logits_width, layer.block)), torch.float32))
    logits, *scores = current_workspace_manager().get_simultaneous(*specs)
    native = metadata.decode_native
    if native is not None:
        # A request's next_n rows go in as one sequence, so a workgroup walks
        # each KV tile once for all of them.
        q, q_scale, weights = layer.rows(0, rows, native.context_lens.shape[0])
        context_lens, block_table, row_ends = (
            native.context_lens,
            native.block_table,
            lengths,
        )
    else:
        q, q_scale, weights = layer.decode_rows(rows)
        context_lens, block_table, row_ends = lengths, metadata.decode.block_table, None
        assert block_table.shape[0] == rows, "one block-table row per query row"
    _aiter().paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        context_lens,
        block_table,
        logits_width,
        out_logits=logits,
        clean_logits=False,
        row_ends=row_ends,
        schedule=metadata.decode_schedule,
        **_block_scores(layer, scores[0] if scores else None),
    )
    candidates = None if layer.candidates is None else layer.candidates[:rows]
    if scores:
        assert candidates is not None and metadata.decode_block_ends is not None
        _topk(scores[0], metadata.decode_block_ends, candidates, candidates.shape[1])
    elif candidates is not None:
        _apply_candidate_mask_strided(logits, None, lengths, candidates, layer.block)
    _topk(
        logits,
        lengths,
        layer.topk_buffer[:rows, : layer.topk_tokens],
        layer.topk_tokens,
    )


def _gather_prefill(
    layer: _Layer, launch: "RocmMxfp4GatherLaunch", num_cols: int
) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    t0, t1 = launch.token_start, launch.token_end
    if launch.pool is None:
        launch.pool = pa.build_candidate_gather(
            layer.candidates[t0:t1],
            launch.row_ends,
            launch.block_table.expand(t1 - t0, -1),
            layer.kv,
            layer.num_heads,
            layer.head_dim,
            layer.block,
        )
    gather, slot_ends = launch.pool
    compact = layer.q.new_empty((t1 - t0, num_cols), dtype=torch.float32)
    q, q_scale, weights = layer.rows(t0, t1)
    pa.paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        launch.context_len,
        launch.block_table,
        num_cols,
        out_logits=compact,
        clean_logits=False,
        row_ends=slot_ends,
        use_gather=True,
        candidates=gather,
    )
    out = layer.topk_buffer[t0:t1, : layer.topk_tokens]
    _topk(compact, slot_ends, out, layer.topk_tokens)
    _remap_compact_topk(out, gather["positions"], layer.block)


def _gather_decode(layer: _Layer, num_cols: int) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    block_table = metadata.decode.block_table
    rows = lengths.shape[0]
    if metadata.decode_gather is None:
        metadata.decode_gather = pa.build_candidate_gather(
            layer.candidates[:rows],
            lengths,
            block_table,
            layer.kv,
            layer.num_heads,
            layer.head_dim,
            layer.block,
        )
    gather, slot_ends = metadata.decode_gather
    (compact,) = current_workspace_manager().get_simultaneous(
        ((rows, num_cols), torch.float32)
    )
    q, q_scale, weights = layer.decode_rows(rows)
    pa.paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        lengths,
        block_table,
        num_cols,
        out_logits=compact,
        clean_logits=False,
        row_ends=slot_ends,
        use_gather=True,
        candidates=gather,
    )
    out = layer.topk_buffer[:rows, : layer.topk_tokens]
    _topk(compact, slot_ends, out, layer.topk_tokens)
    _remap_compact_topk(out, gather["positions"], layer.block)


@eager_break_during_capture
def rocm_mxfp4_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    """Dense indexer: every layer that scores the whole context, the
    candidate source included. With ``candidate_blocks`` and not
    ``candidate_write`` the scores are masked to the pool first, as the
    shared path does.

    The model supplies ``q_values`` [T, H, D // 2] packed e2m1, ``q_scale``
    holding each head's D // 32 ue8m0 bytes (one int32 per head at D = 128),
    fp32 ``weights`` with the softmax and head scales folded in, and
    ``kv_cache`` pages in `rocm_paged_mxfp4_cache_layout` order for H heads.
    """
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states,
            max_model_len,
            candidate_block_size if candidate_write else 0,
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            _dense_prefill(layer, chunk, plan, candidate_write)
    if metadata.decode is not None:
        _dense_decode(layer, max_model_len, candidate_write)
    return topk_indices_buffer


@eager_break_during_capture
def rocm_mxfp4_sparse_mqa_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    num_candidate_cols: int,
) -> torch.Tensor:
    """Candidate consumer: score only the source's pool where the builder's
    length gate says it pays, else the dense walk masked to the pool."""
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states,
            max(max_model_len, num_candidate_cols),
            gather_block_size=candidate_block_size,
            num_candidate_cols=num_candidate_cols,
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            if not plan.use_gather:
                _dense_prefill(layer, chunk, plan, candidate_write=False)
        for launch in metadata.gather_launches:
            _gather_prefill(layer, launch, num_candidate_cols)
    if metadata.decode is not None:
        # A FULL graph cannot follow the per-step gate; the gather is the side
        # that stays flat in the context length.
        if metadata.decode_use_gather or layer.full_graph:
            _gather_decode(layer, num_candidate_cols)
        else:
            _dense_decode(layer, max_model_len, candidate_write=False)
    return topk_indices_buffer
