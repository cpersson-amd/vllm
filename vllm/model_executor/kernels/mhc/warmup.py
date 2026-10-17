# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    compute_num_split,
    mhc_pre_big_fuse_with_norm_tilelang,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.math_utils import cdiv


def compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:
    splits = compute_num_split(64, input_size, cdiv(num_tokens, 64))
    # Bound both GEMM and fused-normalization specializations during startup.
    return 1 if splits == 1 else 4 if splits <= 4 else 16


class MHCPreNormKernel(VllmJitKernel["MHCPreNormKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        rms_eps: float
        hc_pre_eps: float
        hc_sinkhorn_eps: float
        hc_post_mult_value: float
        sinkhorn_repeat: int
        norm_eps: float
        n_splits: int
        hc_mult: int
        use_pre_mix_in: bool
        rms_numel: int
        save_pre_mix: bool = True
        split_mode: str = "fused"
        write_aux: bool = False

    kernel: Any = staticmethod(mhc_pre_big_fuse_with_norm_tilelang)

    def dispatch(self, *, n_splits, **fields) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(n_splits=n_splits, **fields)

    def get_warmup_keys(
        self,
        *,
        max_tokens: int,
        extra_splits: Iterable[int] = (),
        **fields,
    ) -> list[CompileKey]:
        # The split heuristic changes only at 64-token boundaries. Callers that
        # reach this kernel through another GEMM (the fused post + pre-norm
        # GEMM) pass the splits that GEMM picks in extra_splits.
        splits = sorted(
            set(extra_splits)
            | (
                {
                    compute_mhc_pre_num_splits(fields["rms_numel"], num_tokens)
                    for num_tokens in range(1, max_tokens + 1, 64)
                }
                if is_deep_gemm_supported()
                else {1}
            )
        )
        return self._trace_dispatch(self.dispatch)(n_splits=splits, **fields)

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key not in self._compiled_cache:
            self._compiled_cache[compile_key] = self.kernel.compile(
                **asdict(compile_key)
            )

    def compile_many(self, compile_keys: Iterable[CompileKey]) -> None:
        missing = list(
            dict.fromkeys(
                key for key in compile_keys if key not in self._compiled_cache
            )
        )
        if len(missing) < 2:
            return super().compile_many(missing)

        # TileLang elaborates serially before compiling independent TIR modules.
        compiled = self.kernel.par_compile(
            [asdict(key) for key in missing], num_workers=min(4, len(missing))
        )
        self._compiled_cache.update(zip(missing, compiled, strict=True))

    def __call__(self, *tensors, **fields):
        compile_key = self.dispatch(
            n_splits=tensors[0].shape[0],
            **fields,
        )
        if not current_platform.is_cuda():
            return self.kernel(*tensors, **asdict(compile_key))
        return self._get_or_compile(compile_key)(*tensors)


MHC_PRE_NORM_KERNEL = MHCPreNormKernel()


class AiterMHCFusedPostPreDelayedKernel(
    VllmJitKernel["AiterMHCFusedPostPreDelayedKernel.CompileKey"]
):
    """Compiles aiter's ``mhc_fused_post_pre_delayed_rmsnorm`` main and reduce
    kernels for every launch config its token-count table picks. The main
    kernel also specializes on the token count (1, multiple of 16, other)."""

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        sinkhorn_repeat: int
        has_post: bool
        config: tuple[tuple[str, int], ...]
        num_tokens_class: int
        # A token count that selects config and num_tokens_class.
        num_tokens: int = field(compare=False)

    def get_warmup_keys(
        self, *, max_tokens: int, hidden_size: int, sinkhorn_repeat: int
    ) -> list[CompileKey]:
        from aiter.ops.triton.utils.mhc_config_utils import (
            get_mhc_fused_post_pre_delayed_rmsnorm_config,
        )

        keys: dict[AiterMHCFusedPostPreDelayedKernel.CompileKey, None] = {}
        for num_tokens in range(1, max_tokens + 1):
            config = tuple(
                sorted(
                    get_mhc_fused_post_pre_delayed_rmsnorm_config(num_tokens).items()
                )
            )
            num_tokens_class = (
                1 if num_tokens == 1 else 16 if num_tokens % 16 == 0 else 0
            )
            for has_post in (False, True):
                key = self.CompileKey(
                    hidden_size,
                    sinkhorn_repeat,
                    has_post,
                    config,
                    num_tokens_class,
                    num_tokens,
                )
                keys.setdefault(key, None)
        return list(keys)

    def compile(self, compile_key: CompileKey) -> None:
        from aiter.ops.triton._triton_kernels.fusions.mhc_fused_post_pre_delayed_rmsnorm import (  # noqa: E501
            _mhc_fused_post_pre_delayed_rmsnorm_main_kernel as main_kernel,
        )
        from aiter.ops.triton._triton_kernels.fusions.mhc_fused_post_pre_delayed_rmsnorm import (  # noqa: E501
            _mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel as reduce_kernel,
        )

        T, H = compile_key.num_tokens, compile_key.hidden_size
        config = dict(compile_key.config)
        bf16, fp32 = torch.bfloat16, torch.float32
        residual = TritonWarmupTensor(bf16, shape=(T, 4, H))
        if compile_key.has_post:
            sublayer_out = TritonWarmupTensor(bf16, shape=(T, H))
            post = TritonWarmupTensor(fp32, shape=(T, 4))
            comb = TritonWarmupTensor(fp32, shape=(T, 4, 4))
            residual_out = TritonWarmupTensor(bf16, shape=(T, 4, H))
        else:
            sublayer_out = post = comb = residual_out = residual
        layer_input = TritonWarmupTensor(bf16, shape=(T, H))
        partial = TritonWarmupTensor(fp32, shape=(T, config["NUM_KSPLIT"], 32))
        main_kernel.warmup(
            residual,
            sublayer_out,
            post,
            comb,
            TritonWarmupTensor(fp32, shape=(T, 4)),
            TritonWarmupTensor(fp32, shape=(24, 4 * H)),
            residual_out,
            layer_input,
            partial,
            T,
            H=H,
            BLOCK_M=config["BLOCK_M"],
            TILE_K=config["TILE_K"],
            NUM_KSPLIT=config["NUM_KSPLIT"],
            HAS_POST=compile_key.has_post,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            grid=(1,),
        )
        reduce_kernel.warmup(
            partial,
            TritonWarmupTensor(fp32, shape=(3,)),
            TritonWarmupTensor(fp32, shape=(24,)),
            TritonWarmupTensor(fp32, shape=(T, 4)),
            TritonWarmupTensor(fp32, shape=(T, 4, 1)),
            TritonWarmupTensor(fp32, shape=(T, 4, 4)),
            layer_input,
            TritonWarmupTensor(bf16, shape=(H,)),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            1e-6,
            H=H,
            NUM_KSPLIT=config["NUM_KSPLIT"],
            NUM_SINKHORN_ITERS=compile_key.sinkhorn_repeat,
            BLOCK_C=config["BLOCK_C"],
            num_warps=config["reduce_num_warps"],
            grid=(1,),
        )


AITER_MHC_FUSED_POST_PRE_DELAYED_KERNEL = AiterMHCFusedPostPreDelayedKernel()
