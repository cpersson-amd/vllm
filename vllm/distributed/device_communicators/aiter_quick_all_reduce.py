# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned wrapper over AITER's FlyDSL INT4 QuickReduce (``QRInt4``).

``QRInt4`` is a two-shot INT4-quantized all-reduce for gfx942/gfx950. It targets
the case vLLM's built-in QuickReduce codecs leave on the table: the
``QuickAllReduce._QR_MIN_SIZE`` table sets a 2048 MB floor for
``(bfloat16, 8)``, so the INT4 codec effectively never engages for bf16 TP8
unless the input is first cast to fp16. ``QRInt4`` is bf16-native and needs no
cast.

``CudaCommunicator`` stores one of these as ``aiter_qr_comm`` when
``VLLM_ROCM_USE_AITER_QUICK_REDUCE`` is set, occupying the same dispatch slot as
``QuickAllReduce``.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

KB = 1024
MB = 1024 * KB


class AiterQuickAllReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    # QRInt4 rejects anything but bf16 in its own payload check.
    _SUPPORTED_DTYPES = [torch.bfloat16]
    _SUPPORTED_ARCHS = ["gfx942", "gfx950"]

    # QRInt4._check_payload rejects payloads above this. The IPC inbox itself is
    # sized by grid (not by payload) and the kernel walks tiles grid-strided, so
    # this 4 GiB window is the only upper bound.
    _MAX_SIZE = 0xFFFFFFFF

    # Provisional lower bounds, below which two-shot INT4 loses to RCCL/custom
    # AR. Seeded from the fp16 INT4 column of QuickAllReduce._QR_MIN_SIZE, which
    # is the closest measured proxy. Override with
    # VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB once measured on real hardware.
    _MIN_SIZE = {2: 1 * MB, 4: 2 * MB, 8: 2 * MB}

    # Shape used to force the FlyDSL JIT at init. QRInt4 kernels are specialized
    # on (world_size, super_tile, grid) only -- payload byte count and tile count
    # are runtime kernel arguments -- so any valid payload compiles every engine
    # and no per-shape recompile happens later. Kept model-independent so all
    # ranks agree without consulting the model config.
    _WARMUP_SHAPE = (512, 4096)

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        super_tile: int = 8,
        grid_cap: int | None = None,
    ) -> None:
        """
        Args:
            group: the process group to work on. Must be a non-NCCL, single-node
                group: QRInt4 exchanges HIP IPC handles via
                ``broadcast_object_list``.
            device: the device to bind this communicator to.

        Never raises. Any unmet precondition leaves ``self.disabled`` True so
        ``CudaCommunicator`` falls through to the next all-reduce backend.
        """
        self.disabled = True
        self._impl = None

        if not current_platform.is_rocm():
            return

        arch = self._rocm_arch()
        if arch is None or not any(a in arch for a in self._SUPPORTED_ARCHS):
            logger.debug(
                "AITER quick allreduce is disabled: requires one of %s, got %s.",
                self._SUPPORTED_ARCHS,
                arch,
            )
            return

        if dist.get_backend(group) == dist.Backend.NCCL:
            logger.warning(
                "AITER quick allreduce is disabled because it was attached to a "
                "NCCL group; HIP IPC handle exchange needs a CPU-side group."
            )
            return

        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "AITER quick allreduce is disabled because this process group "
                "spans across nodes; HIP IPC handles are node-local."
            )
            return

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "AITER quick allreduce is disabled due to an unsupported world "
                "size: %d. Supported world sizes: %s.",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        vllm_config = get_current_vllm_config_or_none()
        model_dtype = getattr(getattr(vllm_config, "model_config", None), "dtype", None)
        if model_dtype is not None and model_dtype not in self._SUPPORTED_DTYPES:
            logger.debug(
                "AITER quick allreduce is disabled: QRInt4 is bf16-only, but the "
                "model dtype is %s.",
                model_dtype,
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = world_size
        self.qr_max_size = self._MAX_SIZE
        self.qr_min_size = self._resolve_min_size(world_size)

        try:
            from aiter.ops.flydsl import QRInt4
        except ImportError as e:
            logger.warning(
                "AITER quick allreduce is disabled because aiter.ops.flydsl "
                "could not be imported: %s",
                e,
            )
            return

        try:
            impl = QRInt4(
                group=group,
                device=device,
                rank=self.rank,
                world_size=world_size,
                super_tile=super_tile,
                grid_cap=grid_cap,
            )
        except Exception as e:
            logger.warning("AITER quick allreduce initialization failed: %s", e)
            return

        try:
            self._warmup(impl)
        except Exception as e:
            logger.warning(
                "AITER quick allreduce is disabled because the FlyDSL warmup "
                "failed: %s",
                e,
            )
            impl.close()
            return

        self._impl = impl
        self.disabled = False
        logger.info(
            "AITER quick allreduce (FlyDSL INT4) enabled for world size %d, "
            "eligible input range [%d KB, %d MB].",
            world_size,
            self.qr_min_size // KB,
            self.qr_max_size // MB,
        )

    def _warmup(self, impl) -> None:
        """Compile every FlyDSL engine before the first real all-reduce.

        ``QRInt4.allreduce`` JIT-compiles on its first launch, so without this
        the first call would compile inside whatever region it lands in --
        including a CUDA graph capture, where compiling is fatal. All ranks
        launch the same shape and the kernel is a peer-reading two-shot
        all-reduce, so the barriers keep the ranks in lockstep.
        """
        inp = torch.zeros(self._WARMUP_SHAPE, device=self.device, dtype=torch.bfloat16)
        out = torch.empty_like(inp)
        dist.barrier(group=self.group)
        impl.compile(inp, out)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    @classmethod
    def _resolve_min_size(cls, world_size: int) -> int:
        override = envs.VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB
        if override is not None:
            if override < 0:
                raise ValueError(
                    "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB must be "
                    f"non-negative, got {override}"
                )
            return override * MB
        return cls._MIN_SIZE[world_size]

    @staticmethod
    def _rocm_arch() -> str | None:
        try:
            return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        except Exception as e:
            logger.warning("Failed to determine ROCm arch for quick allreduce: %s", e)
            return None

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if not is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        # QRInt4 reduces in 16-byte units.
        if inp_size % 16 != 0:
            return False
        return self.qr_min_size <= inp_size <= self.qr_max_size

    def quick_all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Performs an out-of-place INT4 quick all-reduce.

        No separate graph-capture mode is needed: the IPC inbox is allocated
        once at init and every launch takes its pointers as runtime arguments.
        """
        assert self._impl is not None
        if out is None:
            out = torch.empty_like(inp)
        # QRInt4 requires contiguous, non-overlapping buffers; is_weak_contiguous
        # in should_quick_allreduce admits non-contiguous-but-dense views.
        self._impl.allreduce(inp.contiguous(), out)
        return out

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not raise, especially at interpreter shutdown.
            return
