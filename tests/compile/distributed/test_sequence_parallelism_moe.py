# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.sequence_parallelism_moe import SequenceParallelismMoEPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

from ...utils import multi_gpu_test
from ..backend import TestBackend


class TestMoEModel(torch.nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.w_up = torch.nn.Parameter(torch.rand(hidden_size, intermediate_size))
        self.w_down = torch.nn.Parameter(torch.rand(intermediate_size, hidden_size))

    def forward(self, x):
        h = torch.mm(x, self.w_up)

        h_reduced = tensor_model_parallel_all_reduce(h)
        h_sharded = sequence_parallel_chunk(h_reduced)

        h_act = torch.relu(h_sharded)

        h_out_sharded = torch.mm(h_act, self.w_down)

        tp_group = get_tp_group()
        tp_size = get_tensor_model_parallel_world_size()

        h_final = torch.ops.vllm.all_gather.default(
            h_out_sharded, dim=0, world_size=tp_size, group_name=tp_group.unique_name
        )

        return h_final

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops.vllm.sequence_parallel_chunk_impl.default,
        ]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.reduce_scatter.default,
        ]

    def ops_in_model(self):
        return []


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model_cls, custom_ops",
    [
        (TestMoEModel, "+sequence_parallel_chunk_impl"),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_sequence_parallelism_moe_pass(
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model_cls,
                custom_ops,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                dynamic,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(sequence_parallelism_moe_pass_on_test_model, num_processes)


def sequence_parallelism_moe_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    custom_ops_list = custom_ops.split(",") if custom_ops else []
    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        custom_ops=custom_ops_list,
        pass_config=PassConfig(
            enable_sp_moe=True,
            enable_noop=True,
        ),
    )
    device_config = DeviceConfig(device=torch.device("cuda"))

    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        compilation_config=compilation_config,
    )

    with set_current_vllm_config(vllm_config):
        noop_pass = NoOpEliminationPass(vllm_config)
        sp_moe_pass = SequenceParallelismMoEPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        assert (
            sp_moe_pass.compilation_config.splitting_ops
            == vllm_config.compilation_config.splitting_ops
        )
        assert (
            sp_moe_pass.compilation_config.use_inductor_graph_partition
            == vllm_config.compilation_config.use_inductor_graph_partition
        )
        passes_for_backend: list[VllmInductorPass] = [
            noop_pass,
            sp_moe_pass,
        ]

        passes_for_backend.append(cleanup_pass)

        backend = TestBackend(*passes_for_backend)

        model = test_model_cls(hidden_size)

        hidden_states = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sp_moe_pass.matched_count == 1

        # In pre-nodes, all reduce should be there,
        # reduce scatter and all gather should not
        for op in model.ops_in_model_before():
            assert backend.op_count(op, before=True) == 1

        # In post-nodes, reduce scatter and all gather should be there,
        # all reduce should not
        for op in model.ops_in_model_after():
            assert backend.op_count(op, before=False) == 1
