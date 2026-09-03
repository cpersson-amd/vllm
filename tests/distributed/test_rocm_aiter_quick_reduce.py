# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER FlyDSL INT4 QuickReduce backend.

The gating tests build a bare ``AiterQuickAllReduce`` via ``__new__`` so they
run without GPUs or a process group. The correctness tests need gfx942/gfx950,
a TP size in {2, 4, 8}, and ``aiter.ops.flydsl``.
"""

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.aiter_quick_all_reduce import (
    KB,
    MB,
    AiterQuickAllReduce,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

# QRInt4 is lossy, so correctness is a signal-quality bar rather than a
# tolerance. Both figures are the ones aiter's own op_tests/test_flydsl_qr_int4.py
# asserts (SQNR_MIN_DB / TILE_SQNR_MIN_DB).
SQNR_MIN_DB = 18.0
TILE_SQNR_MIN_DB = 8.0
# aiter measures the per-tile floor over 32 KiB tiles.
SQNR_TILE_BYTES = 32 * KB


def _supported_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    return on_gfx942() or on_gfx950()


def _has_flydsl() -> bool:
    try:
        from aiter.ops.flydsl import QRInt4  # noqa: F401
    except ImportError:
        return False
    return True


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    got = got.to(torch.float32)
    reference = reference.to(torch.float32)
    mse = ((got - reference) ** 2).mean()
    if mse == 0:
        return float("inf")
    return float(10.0 * torch.log10((reference * reference).mean() / mse))


def _min_tile_sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst-tile SQNR. A dropped or stale tile shows up here as ~0 dB even
    when the whole-tensor figure still looks healthy."""
    tile_elems = SQNR_TILE_BYTES // got.element_size()
    flat_got = got.flatten()
    flat_ref = reference.flatten()
    n_full = (flat_got.numel() // tile_elems) * tile_elems
    if n_full == 0:
        return _sqnr_db(flat_got, flat_ref)
    worst = min(
        _sqnr_db(flat_got[i : i + tile_elems], flat_ref[i : i + tile_elems])
        for i in range(0, n_full, tile_elems)
    )
    if flat_got.numel() > n_full:
        worst = min(worst, _sqnr_db(flat_got[n_full:], flat_ref[n_full:]))
    return worst


def _assert_quality(got: torch.Tensor, reference: torch.Tensor) -> None:
    sqnr = _sqnr_db(got, reference)
    min_tile = _min_tile_sqnr_db(got, reference)
    assert sqnr >= SQNR_MIN_DB, f"SQNR {sqnr:.2f} dB < {SQNR_MIN_DB}"
    assert min_tile >= TILE_SQNR_MIN_DB, (
        f"min-tile SQNR {min_tile:.2f} dB < {TILE_SQNR_MIN_DB}"
    )


def _make_comm_for_test(
    min_size: int = 2 * MB,
    max_size: int = 16 * MB,
) -> AiterQuickAllReduce:
    """A bare instance with only the fields ``should_quick_allreduce`` reads."""
    comm = AiterQuickAllReduce.__new__(AiterQuickAllReduce)
    comm.disabled = False
    comm._impl = None
    comm.qr_min_size = min_size
    comm.qr_max_size = max_size
    return comm


@pytest.fixture
def envs_cache_disabled():
    disable_envs_cache()
    yield
    disable_envs_cache()


def test_gate_accepts_bf16_in_range():
    comm = _make_comm_for_test()
    inp = torch.zeros(2 * MB // 2, dtype=torch.bfloat16)
    assert comm.should_quick_allreduce(inp)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gate_rejects_non_bf16(dtype):
    """QRInt4 raises on non-bf16 payloads, so the gate has to filter them."""
    comm = _make_comm_for_test()
    elem_size = torch.empty(0, dtype=dtype).element_size()
    inp = torch.zeros(4 * MB // elem_size, dtype=dtype)
    assert not comm.should_quick_allreduce(inp)


def test_gate_rejects_below_min_and_above_max():
    comm = _make_comm_for_test(min_size=2 * MB, max_size=16 * MB)
    too_small = torch.zeros((2 * MB // 2) - 8, dtype=torch.bfloat16)
    too_large = torch.zeros((16 * MB // 2) + 8, dtype=torch.bfloat16)
    assert not comm.should_quick_allreduce(too_small)
    assert not comm.should_quick_allreduce(too_large)


def test_gate_rejects_non_16_byte_multiple():
    """QRInt4 reduces in 16-byte units and raises otherwise."""
    comm = _make_comm_for_test(min_size=0)
    # 9 bf16 elements = 18 bytes, not a multiple of 16.
    assert not comm.should_quick_allreduce(torch.zeros(9, dtype=torch.bfloat16))


def test_gate_rejects_when_disabled():
    comm = _make_comm_for_test()
    comm.disabled = True
    assert not comm.should_quick_allreduce(
        torch.zeros(2 * MB // 2, dtype=torch.bfloat16)
    )


def test_min_size_env_override(envs_cache_disabled, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "8")
    assert AiterQuickAllReduce._resolve_min_size(8) == 8 * MB


def test_min_size_env_override_rejects_negative(
    envs_cache_disabled, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "-1")
    with pytest.raises(ValueError):
        AiterQuickAllReduce._resolve_min_size(8)


def test_min_size_default_table():
    for world_size in AiterQuickAllReduce._SUPPORTED_WORLD_SIZES:
        assert AiterQuickAllReduce._resolve_min_size(world_size) > 0


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_QUICK_REDUCE", "1")
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)
    comm = get_tp_group().device_communicator.aiter_qr_comm
    assert comm is not None, "AITER quick reduce communicator was not initialized."
    assert not comm.disabled, "AITER quick reduce should be enabled."
    return device, comm


def _sample(numel: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(numel, generator=gen, dtype=torch.float32) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )


def _eager_aiter_quick_reduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    with monkeypatch.context() as m:
        device, comm = _init_worker(m, tp_size, pp_size, rank, distributed_init_port)
        group = get_tp_group().device_group

        # 4 MB, comfortably above the TP-8 default threshold.
        inp = _sample(2 * 1024 * 1024, device, seed=1234 + rank)
        assert comm.should_quick_allreduce(inp)

        reference = inp.clone()
        dist.all_reduce(reference, group=group)
        out = comm.quick_all_reduce(inp)
        _assert_quality(out, reference)

        # The wrapper must not reduce in place.
        assert out.data_ptr() != inp.data_ptr()


def _dispatch_aiter_quick_reduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    """The communicator must actually be reached through ``all_reduce``."""
    with monkeypatch.context() as m:
        device, comm = _init_worker(m, tp_size, pp_size, rank, distributed_init_port)
        group = get_tp_group().device_group

        inp = _sample(2 * 1024 * 1024, device, seed=99 + rank)
        reference = inp.clone()
        dist.all_reduce(reference, group=group)
        _assert_quality(tensor_model_parallel_all_reduce(inp), reference)

        # Below the min-size gate the chain must fall through to a lossless
        # backend, so this one is exact.
        small = _sample(64, device, seed=7 + rank)
        assert not comm.should_quick_allreduce(small)
        small_reference = small.clone()
        dist.all_reduce(small_reference, group=group)
        torch.testing.assert_close(
            tensor_model_parallel_all_reduce(small), small_reference
        )


def _graph_aiter_quick_reduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    """QRInt4 JIT-compiles on first launch, which would abort a graph capture.
    This is what the init-time warmup in AiterQuickAllReduce exists to prevent."""
    with monkeypatch.context() as m:
        device, comm = _init_worker(m, tp_size, pp_size, rank, distributed_init_port)
        group = get_tp_group().device_group

        inp = _sample(2 * 1024 * 1024, device, seed=555 + rank)
        reference = inp.clone()
        dist.all_reduce(reference, group=group)

        with graph_capture(device=device) as ctx:
            assert comm.should_quick_allreduce(inp)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=ctx.stream):
                out = comm.quick_all_reduce(inp)
        graph.replay()
        torch.accelerator.synchronize()
        _assert_quality(out, reference)


try:
    import ray

    _eager_target = ray.remote(num_gpus=1, max_calls=1)(_eager_aiter_quick_reduce)
    _dispatch_target = ray.remote(num_gpus=1, max_calls=1)(_dispatch_aiter_quick_reduce)
    _graph_target = ray.remote(num_gpus=1, max_calls=1)(_graph_aiter_quick_reduce)
except ImportError:
    _eager_target = _dispatch_target = _graph_target = None


@pytest.mark.skipif(
    not _supported_arch(),
    reason="AITER quick reduce requires ROCm gfx942/gfx950",
)
@pytest.mark.skipif(not _has_flydsl(), reason="aiter.ops.flydsl is not available")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "test_target",
    ["eager", "dispatch", "graph"],
)
def test_aiter_quick_reduce(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, test_target: str
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    target = {
        "eager": _eager_target,
        "dispatch": _dispatch_target,
        "graph": _graph_target,
    }[test_target]
    if target is None:
        pytest.skip("ray is required for the multi-process quick reduce tests.")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_QUICK_REDUCE", "1")
    multi_process_parallel(monkeypatch, tp_size, 1, target)
