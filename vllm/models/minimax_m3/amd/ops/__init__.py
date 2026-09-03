# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD/ROCm fused Triton ops for MiniMax-M3.

These replace per-element PyTorch fallbacks (FlashInfer / fused HIP kernels are
unavailable on ROCm) with single-pass Triton kernels to cut launch overhead and
intermediate-tensor traffic during decode.

The SwiGLU-OAI kernels are re-exported from
`vllm.model_executor.layers.fused_swiglu_oai`: they now also back
`SiluAndMulWithClamp` on ROCm, so they cannot live under a model package.
"""

from vllm.model_executor.layers.fused_swiglu_oai import (
    swiglu_oai_quantize_mxfp8,
    swiglu_oai_split,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)

__all__ = [
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "swiglu_oai_split",
    "swiglu_oai_quantize_mxfp8",
]
