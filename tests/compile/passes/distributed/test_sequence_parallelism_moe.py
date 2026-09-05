# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm.compilation.passes.fusion.sequence_parallelism_moe import (
    SequenceParallelismMoEPass,
)
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test CUDA")


class AllReduceRMSNormChunkModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tensor_model_parallel_all_reduce(x)
        return sequence_parallel_chunk(self.norm(x))


class AllGatherRMSNormModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(tensor_model_parallel_all_gather(x, 0))


class AllReduceFusedAddRMSNormChunkModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = tensor_model_parallel_all_reduce(x)
        x, residual = self.norm(x, residual)
        return sequence_parallel_chunk(x), residual


def _op_position(graph: torch.fx.GraphModule, target: object) -> int:
    for index, node in enumerate(graph.graph.nodes):
        if node.op == "call_function" and node.target == target:
            return index
    raise AssertionError(f"{target} not found in graph")


@multi_gpu_test(num_gpus=2)
@torch.no_grad()
def test_sequence_parallelism_moe_pass_rewrites_collectives():
    if envs.VLLM_TARGET_DEVICE != "cuda":
        return

    torch.multiprocessing.spawn(
        _run_sequence_parallelism_moe_test,
        args=(2,),
        nprocs=2,
    )


def _run_sequence_parallelism_moe_test(local_rank: int, world_size: int):
    set_random_seed(0)
    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
        }
    )
    init_distributed_environment()

    model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        seed=42,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device=device),
        parallel_config=ParallelConfig(tensor_parallel_size=world_size),
        compilation_config=CompilationConfig(
            splitting_ops=[],
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                enable_sp_moe=True,
                sp_min_token_num=1,
                eliminate_noops=True,
            ),
        ),
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        sequence_parallelism_moe_pass = SequenceParallelismMoEPass(vllm_config)
        backend = TestBackend(sequence_parallelism_moe_pass)

        model = AllReduceRMSNormChunkModel(hidden_size=16)
        hidden_states = torch.randn((8, 16), dtype=torch.bfloat16)
        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sequence_parallelism_moe_pass.matched_count == 1
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=True) == 1
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=False) == 0
        assert (
            backend.op_count(
                torch.ops.vllm.sequence_parallel_chunk_impl.default, before=True
            )
            == 1
        )
        assert (
            backend.op_count(
                torch.ops.vllm.sequence_parallel_chunk_impl.default, before=False
            )
            == 0
        )
        assert (
            backend.op_count(torch.ops.vllm.reduce_scatter.default, before=False) == 1
        )

        model = AllReduceFusedAddRMSNormChunkModel(hidden_size=16)
        hidden_states = torch.randn((8, 16), dtype=torch.bfloat16)
        sequence_parallelism_moe_pass = SequenceParallelismMoEPass(vllm_config)
        backend = TestBackend(sequence_parallelism_moe_pass)
        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sequence_parallelism_moe_pass.matched_count == 1
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=False) == 0
        assert (
            backend.op_count(
                torch.ops.vllm.sequence_parallel_chunk_impl.default, before=False
            )
            == 0
        )
        assert (
            backend.op_count(torch.ops.vllm.reduce_scatter.default, before=False) == 1
        )

        model = AllGatherRMSNormModel(hidden_size=16)
        hidden_states = torch.randn((4, 16), dtype=torch.bfloat16)
        sequence_parallelism_moe_pass = SequenceParallelismMoEPass(vllm_config)
        backend = TestBackend(sequence_parallelism_moe_pass)
        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sequence_parallelism_moe_pass.matched_count == 1
        assert _op_position(
            backend.graph_pre_pass, torch.ops.vllm.all_gather.default
        ) < _op_position(backend.graph_pre_pass, torch.ops.vllm_ir.rms_norm.default)
        assert _op_position(
            backend.graph_post_pass, torch.ops.vllm_ir.rms_norm.default
        ) < _op_position(backend.graph_post_pass, torch.ops.vllm.all_gather.default)
