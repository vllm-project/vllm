# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import (
    multi_gpu_test,
)
from vllm.compilation.passes.fusion.collective_fusion import AsyncTPPass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestMMRSModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.gate_proj = torch.nn.Parameter(
            torch.empty((self.hidden_size * 2, hidden_size)), requires_grad=False
        )
        # Initialize weights
        torch.nn.init.normal_(self.gate_proj, std=0.02)

    def forward(self, hidden_states):
        """
        Forward pass implementing the mm + reduce scatter in the FX graph

        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)

        # matrix multiplication
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_matmul_reduce_scatter.default]


class TestAGMMModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.empty((hidden_size, hidden_size)), requires_grad=False
        )
        # Initialize weights
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        """
        Forward pass implementing the mm + all gather in the FX graph
        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)
        all_gather = tensor_model_parallel_all_gather(view, dim=0)
        permute = self.weight.permute(1, 0)
        mm = torch.mm(all_gather, permute)
        return mm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_matmul.default]


class _BaseScaledMMModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = (
            torch.empty([hidden_size, hidden_size], dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        # Initialize scale_b for _scaled_mm.
        self.scale_b = torch.ones(1, self.hidden_size, dtype=torch.float32)


class TestScaledMMRSModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the scaled_mm + reduce scatter in the FX graph

        """
        fp8_input = input.to(FP8_DTYPE)
        scale_a = torch.ones(input.shape[0], 1, dtype=torch.float32)
        scaled_mm = torch._scaled_mm(
            fp8_input,
            self.weight,
            scale_a=scale_a,
            scale_b=self.scale_b,
            out_dtype=self.dtype,
        )
        reduce_scatter = tensor_model_parallel_reduce_scatter(scaled_mm, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter.default]


class TestAGScaledMMModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the all gather + scaled_mm in the FX graph
        """
        # Reshape input
        fp8_input = input.to(FP8_DTYPE)
        all_gather = tensor_model_parallel_all_gather(fp8_input, dim=0)

        scale_a = torch.ones(all_gather.shape[0], 1, dtype=torch.float32)
        scaled_mm = torch._scaled_mm(
            all_gather,
            self.weight,
            scale_a=scale_a,
            scale_b=self.scale_b,
            out_dtype=self.dtype,
        )
        return scaled_mm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_scaled_matmul.default]


class TestCutlassScaledMMRSModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the cutlass_scaled_mm + reduce scatter
        in the FX graph

        """
        fp8_input = input.to(FP8_DTYPE)
        scale_a = torch.ones(input.shape[0], 1, dtype=torch.float32)
        mm_out = torch.empty(
            (fp8_input.shape[0], self.weight.shape[1]),
            dtype=self.dtype,
            device=input.device,
        )
        torch.ops._C.cutlass_scaled_mm(
            mm_out, fp8_input, self.weight, scale_a, self.scale_b, None
        )
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm_out, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter.default]


class TestAGCutlassScaledMMModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the all gather + cutlass_scaled_mm
        in the FX graph
        """
        # Reshape input
        fp8_input = input.to(FP8_DTYPE)
        all_gather = tensor_model_parallel_all_gather(fp8_input, dim=0)

        scale_a = torch.ones(all_gather.shape[0], 1, dtype=torch.float32)

        mm_out = torch.empty(
            (all_gather.shape[0], self.weight.shape[1]),
            dtype=self.dtype,
            device=all_gather.device,
        )
        torch.ops._C.cutlass_scaled_mm(
            mm_out, all_gather, self.weight, scale_a, self.scale_b, None
        )
        return mm_out

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_scaled_matmul.default]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model",
    [
        TestMMRSModel,
        TestAGMMModel,
        TestScaledMMRSModel,
        TestAGScaledMMModel,
        TestCutlassScaledMMRSModel,
        TestAGCutlassScaledMMModel,
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_async_tp_pass_replace(
    test_model: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
):
    if (
        test_model
        in (
            TestScaledMMRSModel,
            TestAGScaledMMModel,
            TestCutlassScaledMMRSModel,
            TestAGCutlassScaledMMModel,
        )
        and dtype == torch.float16
    ):
        pytest.skip(
            "Only bf16 high precision output types are supported for "
            "per-token (row-wise) scaling"
        )

    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                dynamic,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(async_tp_pass_on_test_model, num_processes)


def async_tp_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
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
            "MASTER_PORT": "12345",
        }
    )

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # configure vllm config for SequenceParallelismPass
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(
        pass_config=PassConfig(
            fuse_gemm_comms=True,
        ),
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    async_tp_pass = AsyncTPPass(vllm_config)

    # Set the global vllm_config for TestBackend which calls
    # get_current_vllm_config()
    with set_current_vllm_config(vllm_config):
        backend = TestBackend(async_tp_pass)

        assert (
            async_tp_pass.compilation_config.splitting_ops
            == vllm_config.compilation_config.splitting_ops
        )
        assert (
            async_tp_pass.compilation_config.use_inductor_graph_partition
            == vllm_config.compilation_config.use_inductor_graph_partition
        )

        model = test_model_cls(hidden_size, dtype)  # Pass dtype to model constructor

        hidden_states = torch.randn(
            (batch_size * seq_len, hidden_size), dtype=dtype, requires_grad=False
        )

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert async_tp_pass.matched_count == 1

        # In pre-nodes, all gather or reduce scatter should exist,
        # fused_matmul_reduce_scatter or fused_all_gather_matmul should not
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)

        # In post-nodes, fused_matmul_reduce_scatter or \
        # fused_all_gather_matmul should exist
        backend.check_after_ops(model.ops_in_model_after())
