# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.passes.fusion.allreduce_add_fusion import AllReduceAddFusionPass
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type


class TestAllReduceAddModel(torch.nn.Module):
    """One foldable add in front of an all-reduce, and three that must stay:
    a broadcasting add, a dtype-promoting add, and an add with another user."""

    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.w = torch.rand(hidden_size, hidden_size, dtype=dtype) / hidden_size
        self.bias = torch.rand(hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor):
        routed = torch.relu(x)
        shared = torch.mm(routed, self.w)
        folded = tensor_model_parallel_all_reduce(shared + routed)

        broadcast = tensor_model_parallel_all_reduce(shared + self.bias)
        promoted = tensor_model_parallel_all_reduce(routed + shared.float())

        kept = folded + shared
        multi_user = tensor_model_parallel_all_reduce(kept)
        return folded, broadcast, promoted, multi_user, kept


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_reduce_add_fusion_pass_replace(dtype: torch.dtype):
    use_aiter = IS_AITER_FOUND and current_platform.is_rocm()
    num_processes = 2
    torch.multiprocessing.spawn(
        all_reduce_add_fusion_pass_on_test_model,
        args=(num_processes, get_open_port(), dtype, use_aiter),
        nprocs=num_processes,
    )


def all_reduce_add_fusion_pass_on_test_model(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype: torch.dtype,
    use_aiter: bool,
):
    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "VLLM_ROCM_USE_AITER": str(int(use_aiter)),
            "VLLM_ROCM_USE_AITER_CUSTOM_AR": str(int(use_aiter)),
        }
    )
    if use_aiter:
        rocm_aiter_ops.refresh_env_variables()

    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_add=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        fusion_pass = AllReduceAddFusionPass(vllm_config)
        backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            FixFunctionalizationPass(vllm_config),
            PostCleanupPass(vllm_config),
        )

        hidden_size = 256
        model = TestAllReduceAddModel(hidden_size, dtype)
        # Different per rank, so the all-reduce isn't just a multiply.
        hidden_states = torch.randn(16, hidden_size) * (local_rank + 1)

        compiled_model = torch.compile(model, backend=backend)
        results_fused = compiled_model(hidden_states)
        results_unfused = model(hidden_states)
        torch.testing.assert_close(results_fused, results_unfused, atol=0, rtol=0)

        assert fusion_pass.matched_count == 1, f"{fusion_pass.matched_count=}"
        all_reduce = torch.ops.vllm.all_reduce.default
        all_reduce_add = torch.ops.vllm.all_reduce_add.default
        assert backend.op_count(all_reduce, before=True) == 4
        assert backend.op_count(all_reduce) == 3
        assert backend.op_count(all_reduce_add) == 1
