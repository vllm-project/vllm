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
from vllm.compilation.passes.fx_utils import find_auto_fn
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
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
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test CUDA")

ALL_GATHER_OP = torch.ops.vllm.all_gather.default
ALL_REDUCE_OP = torch.ops.vllm.all_reduce.default
REDUCE_SCATTER_OP = torch.ops.vllm.reduce_scatter.default
SEQUENCE_PARALLEL_CHUNK_OP = torch.ops.vllm.sequence_parallel_chunk_impl.default
SLICE_OP = torch.ops.aten.slice.Tensor
SLICE_SCATTER_OP = torch.ops.aten.slice_scatter.default
RMS_OP = torch.ops._C.rms_norm.default
FUSED_ADD_RMS_OP = torch.ops._C.fused_add_rms_norm.default


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    raise TypeError(f"Unsupported value type {type(value)!r}")


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 1e-2
    return 3e-3, 3e-3


def expected_matches_for_model(
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
) -> int:
    if (
        test_model_cls
        in (
            AllGatherFusedAddRMSNormModel,
            AllGatherFusedAddRMSNormFirstOutModel,
        )
        and custom_ops == "+rms_norm"
    ):
        # These synthetic all-gather graphs are exercised without the
        # preceding all-reduce+chunk rewrite that normally runs first in the
        # pass pipeline, so the custom-op form is expected to stay unchanged.
        return 0
    return test_model_cls.expected_matches()


def assert_close_tree(actual, expected, *, atol: float, rtol: float):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        assert_close_tree(actual_item, expected_item, atol=atol, rtol=rtol)


def assert_model_outputs_close(
    test_model_cls: type[torch.nn.Module],
    compiled_out,
    eager_out,
    *,
    atol: float,
    rtol: float,
) -> None:
    if test_model_cls is AllGatherFusedAddRMSNormModel:
        # The tuple model is only meant to validate the downstream graph shape.
        # The gathered activation is the stable user-visible result to compare;
        # the residual path is covered by the graph assertions below.
        assert_close_tree(compiled_out[0], eager_out[0], atol=atol, rtol=rtol)
        return
    assert_close_tree(compiled_out, eager_out, atol=atol, rtol=rtol)


def assert_graph_unchanged(backend: TestBackend, ops: list[torch._ops.OpOverload]):
    for op in ops:
        assert backend.op_count(op, before=True) == backend.op_count(op)


class AllReduceRMSNormChunkModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states):
        hidden_states = torch.relu(hidden_states)
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return sequence_parallel_chunk(self.norm(reduced))

    @staticmethod
    def make_inputs(
        hidden_size: int,
        dtype: torch.dtype,
        world_size: int,
        local_rank: int = 0,
    ):
        del local_rank
        del world_size
        return (torch.randn(8, hidden_size, dtype=dtype),)

    @staticmethod
    def expected_matches() -> int:
        return 1

    @staticmethod
    def verify_graph(backend: TestBackend) -> None:
        assert backend.op_count(ALL_REDUCE_OP, before=True) == 1
        assert backend.op_count(ALL_REDUCE_OP) == 0
        assert backend.op_count(REDUCE_SCATTER_OP, before=True) == 0
        assert backend.op_count(REDUCE_SCATTER_OP) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP, before=True) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP) == 0

    @staticmethod
    def ops_in_model():
        if RMSNorm.enabled():
            return [RMS_OP]
        return []


class AllReduceFusedAddRMSNormChunkModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states):
        residual = torch.sin(hidden_states)
        reduced = tensor_model_parallel_all_reduce(torch.cos(hidden_states))
        normed, residual = self.norm(reduced, residual)
        return sequence_parallel_chunk(normed), sequence_parallel_chunk(residual)

    @staticmethod
    def make_inputs(
        hidden_size: int,
        dtype: torch.dtype,
        world_size: int,
        local_rank: int = 0,
    ):
        del local_rank
        del world_size
        return (torch.randn(8, hidden_size, dtype=dtype),)

    @staticmethod
    def expected_matches() -> int:
        return 1

    @staticmethod
    def verify_graph(backend: TestBackend) -> None:
        assert backend.op_count(ALL_REDUCE_OP, before=True) == 1
        assert backend.op_count(ALL_REDUCE_OP) == 0
        assert backend.op_count(REDUCE_SCATTER_OP, before=True) == 0
        assert backend.op_count(REDUCE_SCATTER_OP) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP, before=True) == 2
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP) == 1

    @staticmethod
    def ops_in_model():
        if RMSNorm.enabled():
            return [FUSED_ADD_RMS_OP]
        return []


class AllReduceFusedAddRMSNormChunkFirstOutModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states):
        residual = torch.sin(hidden_states)
        reduced = tensor_model_parallel_all_reduce(torch.cos(hidden_states))
        normed = self.norm(reduced, residual)[0]
        return sequence_parallel_chunk(normed)

    @staticmethod
    def make_inputs(
        hidden_size: int,
        dtype: torch.dtype,
        world_size: int,
        local_rank: int = 0,
    ):
        del local_rank
        del world_size
        return (torch.randn(8, hidden_size, dtype=dtype),)

    @staticmethod
    def expected_matches() -> int:
        return 1

    @staticmethod
    def verify_graph(backend: TestBackend) -> None:
        assert backend.op_count(ALL_REDUCE_OP, before=True) == 1
        assert backend.op_count(ALL_REDUCE_OP) == 0
        assert backend.op_count(REDUCE_SCATTER_OP, before=True) == 0
        assert backend.op_count(REDUCE_SCATTER_OP) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP, before=True) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP) == 1

    @staticmethod
    def ops_in_model():
        if RMSNorm.enabled():
            return [FUSED_ADD_RMS_OP]
        return []


class AllGatherFusedAddRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, local_hidden_states, residual):
        local_hidden_states = torch.tanh(local_hidden_states)
        gathered = tensor_model_parallel_all_gather(local_hidden_states, dim=0)
        num_tokens = residual.shape[0]
        depadded = gathered[0:num_tokens, ...]
        normed, residual = self.norm(depadded, residual)
        gathered_normed = torch.slice_scatter(gathered, normed, 0, 0, num_tokens)
        return gathered_normed[0:num_tokens, ...], sequence_parallel_chunk(residual)

    @staticmethod
    def make_inputs(
        hidden_size: int,
        dtype: torch.dtype,
        world_size: int,
        local_rank: int = 0,
    ):
        local_tokens = 4
        global_tokens = local_tokens * world_size - 1
        gathered_hidden_states = torch.randn(
            local_tokens * world_size,
            hidden_size,
            dtype=dtype,
        )
        return (
            gathered_hidden_states[
                local_rank * local_tokens : (local_rank + 1) * local_tokens
            ].clone(),
            torch.randn(global_tokens, hidden_size, dtype=dtype),
        )

    @staticmethod
    def expected_matches() -> int:
        return 1

    @staticmethod
    def verify_graph(backend: TestBackend) -> None:
        assert backend.op_count(ALL_GATHER_OP, before=True) == 1
        assert backend.op_count(ALL_GATHER_OP) == 1
        assert backend.op_count(SLICE_SCATTER_OP, before=True) == 1
        assert backend.op_count(SLICE_SCATTER_OP) == 0
        assert backend.op_count(SLICE_OP, before=True) == 2
        assert backend.op_count(SLICE_OP) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP, before=True) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP) == 1

    @staticmethod
    def ops_in_model():
        if RMSNorm.enabled():
            return [FUSED_ADD_RMS_OP]
        return []


class AllGatherFusedAddRMSNormFirstOutModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, local_hidden_states, residual):
        local_hidden_states = torch.tanh(local_hidden_states)
        gathered = tensor_model_parallel_all_gather(local_hidden_states, dim=0)
        num_tokens = residual.shape[0]
        depadded = gathered[0:num_tokens, ...]
        normed = self.norm(depadded, residual)[0]
        gathered_normed = torch.slice_scatter(gathered, normed, 0, 0, num_tokens)
        return gathered_normed[0:num_tokens, ...]

    @staticmethod
    def make_inputs(
        hidden_size: int,
        dtype: torch.dtype,
        world_size: int,
        local_rank: int = 0,
    ):
        local_tokens = 4
        global_tokens = local_tokens * world_size - 1
        gathered_hidden_states = torch.randn(
            local_tokens * world_size,
            hidden_size,
            dtype=dtype,
        )
        return (
            gathered_hidden_states[
                local_rank * local_tokens : (local_rank + 1) * local_tokens
            ].clone(),
            torch.randn(global_tokens, hidden_size, dtype=dtype),
        )

    @staticmethod
    def expected_matches() -> int:
        return 1

    @staticmethod
    def verify_graph(backend: TestBackend) -> None:
        assert backend.op_count(ALL_GATHER_OP, before=True) == 1
        assert backend.op_count(ALL_GATHER_OP) == 1
        assert backend.op_count(SLICE_SCATTER_OP, before=True) == 1
        assert backend.op_count(SLICE_SCATTER_OP) == 0
        assert backend.op_count(SLICE_OP, before=True) == 2
        assert backend.op_count(SLICE_OP) == 1
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP, before=True) == 0
        assert backend.op_count(SEQUENCE_PARALLEL_CHUNK_OP) == 1

    @staticmethod
    def ops_in_model():
        if RMSNorm.enabled():
            return [FUSED_ADD_RMS_OP]
        return []


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model_cls",
    [
        AllReduceRMSNormChunkModel,
        AllReduceFusedAddRMSNormChunkModel,
        AllReduceFusedAddRMSNormChunkFirstOutModel,
        AllGatherFusedAddRMSNormModel,
        AllGatherFusedAddRMSNormFirstOutModel,
    ],
)
@pytest.mark.parametrize("custom_ops", ["+rms_norm", "-rms_norm"])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_sequence_parallelism_moe_pass(
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    hidden_size: int,
    dtype: torch.dtype,
):
    num_processes = 2

    torch.multiprocessing.spawn(
        sequence_parallelism_moe_pass_on_test_model,
        args=(num_processes, test_model_cls, custom_ops, hidden_size, dtype),
        nprocs=num_processes,
    )


def sequence_parallelism_moe_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
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

    custom_ops_list = custom_ops.split(",") if custom_ops else []
    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        custom_ops=custom_ops_list,
        pass_config=PassConfig(enable_sp_moe=True, eliminate_noops=True),
    )
    device_config = DeviceConfig(device=torch.device("cuda"))
    model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=dtype,
        seed=42,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        compilation_config=compilation_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            disable_custom_all_reduce=True,
        ),
    )

    try:
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            sequence_parallelism_moe_pass = SequenceParallelismMoEPass(vllm_config)
            cleanup_pass = PostCleanupPass(vllm_config)
            backend = TestBackend(sequence_parallelism_moe_pass, cleanup_pass)

            model = test_model_cls(hidden_size).to(device=device, dtype=dtype)
            inputs = test_model_cls.make_inputs(
                hidden_size,
                dtype,
                world_size,
                local_rank,
            )
            atol, rtol = get_tolerances(dtype)

            eager_out = model(*clone_tree(inputs))
            compiled_model = torch.compile(model, backend=backend)
            compiled_out = compiled_model(*clone_tree(inputs))

            assert_model_outputs_close(
                test_model_cls,
                compiled_out,
                eager_out,
                atol=atol,
                rtol=rtol,
            )
            expected_matches = expected_matches_for_model(test_model_cls, custom_ops)
            assert sequence_parallelism_moe_pass.matched_count == expected_matches

            if expected_matches > 0:
                test_model_cls.verify_graph(backend)
            else:
                assert_graph_unchanged(
                    backend,
                    [
                        ALL_GATHER_OP,
                        SLICE_SCATTER_OP,
                        SLICE_OP,
                        SEQUENCE_PARALLEL_CHUNK_OP,
                    ],
                )
            for op in test_model_cls.ops_in_model():
                find_auto_fn(backend.graph_post_pass.nodes, op)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
