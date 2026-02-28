# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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
    ParallelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed


class TestAllReduceRMSNormChunkModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid matching directly on a graph input.
        y = torch.relu(x)
        y = tensor_model_parallel_all_reduce(y)
        y = self.norm(y)
        return sequence_parallel_chunk(y)


class TestAllReduceFusedAddRMSNormChunkModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Avoid matching directly on a graph input.
        y = torch.relu(x)
        residual = torch.sigmoid(residual)
        y = tensor_model_parallel_all_reduce(y)
        y, residual = self.norm(y, residual)
        return sequence_parallel_chunk(y), residual


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("custom_ops", ["+rms_norm", "-rms_norm"])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only on CUDA")
def test_sequence_parallelism_moe_pass(
    custom_ops: str,
    with_residual: bool,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    num_processes = 2

    torch.multiprocessing.spawn(
        sequence_parallelism_moe_pass_on_test_model,
        args=(num_processes, custom_ops, with_residual, seq_len, hidden_size, dtype),
        nprocs=num_processes,
    )


def sequence_parallelism_moe_pass_on_test_model(
    local_rank: int,
    world_size: int,
    custom_ops: str,
    with_residual: bool,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    set_random_seed(0)

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
            "MASTER_PORT": "12346",
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
            enable_sp=False,
            enable_sp_moe=True,
            eliminate_noops=True,
        ),
    )
    device_config = DeviceConfig(device=torch.device("cuda"))
    parallel_config = ParallelConfig(tensor_parallel_size=world_size)
    vllm_config = VllmConfig(
        compilation_config=compilation_config,
        device_config=device_config,
        parallel_config=parallel_config,
    )

    with set_current_vllm_config(vllm_config):
        sequence_parallelism_moe_pass = SequenceParallelismMoEPass(vllm_config)

        # In piecewise mode, the pass should be applicable for this compile range.
        piecewise_compilation_config = CompilationConfig(
            splitting_ops=["vllm::unified_attention_with_output"],
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=custom_ops_list,
            pass_config=PassConfig(
                enable_sp=False,
                enable_sp_moe=True,
                eliminate_noops=True,
            ),
        )
        piecewise_vllm_config = VllmConfig(
            compilation_config=piecewise_compilation_config,
            device_config=device_config,
            parallel_config=parallel_config,
        )
        with set_current_vllm_config(piecewise_vllm_config):
            piecewise_pass = SequenceParallelismMoEPass(piecewise_vllm_config)
            assert piecewise_pass.is_applicable_for_range(Range(seq_len, seq_len))
            assert not piecewise_pass.is_applicable_for_range(Range(15, 15))

        backend = TestBackend(sequence_parallelism_moe_pass)
        model: torch.nn.Module = (
            TestAllReduceFusedAddRMSNormChunkModel(hidden_size)
            if with_residual
            else TestAllReduceRMSNormChunkModel(hidden_size)
        )
        hidden_states = torch.randn((seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)

        if with_residual:
            eager_output = model(hidden_states, residual)
        else:
            eager_output = model(hidden_states)

        compiled_model = torch.compile(model, backend=backend)
        if with_residual:
            compiled_output = compiled_model(hidden_states, residual)
            torch.testing.assert_close(compiled_output[0], eager_output[0])
            # Residual is temporarily prefix-sliced in this isolated graph; in
            # full model graphs upstream replacements make this slice a no-op.
            torch.testing.assert_close(
                compiled_output[1], eager_output[1][: seq_len // world_size]
            )
        else:
            compiled_output = compiled_model(hidden_states)
            torch.testing.assert_close(compiled_output, eager_output)

        assert sequence_parallelism_moe_pass.matched_count == 1

        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=True) == 1
        assert (
            backend.op_count(
                torch.ops.vllm.sequence_parallel_chunk_impl.default, before=True
            )
            == 1
        )
        assert (
            backend.op_count(torch.ops.vllm.reduce_scatter.default, before=False) == 1
        )
        assert backend.op_count(torch.ops.vllm.all_gather.default, before=False) == 0
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=False) == 0
        expected_chunk_after = 0
        assert (
            backend.op_count(
                torch.ops.vllm.sequence_parallel_chunk_impl.default, before=False
            )
            == expected_chunk_after
        )
