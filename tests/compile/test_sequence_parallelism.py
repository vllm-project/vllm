# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fusion import FusionPass
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.sequence_parallelism import SequenceParallelismPass
from vllm.config import (CompilationConfig, DeviceConfig, ModelConfig,
                         PassConfig, VllmConfig)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..utils import multi_gpu_test
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestModel(torch.nn.Module):

    def __init__(self,
                 hidden_size=16,
                 intermediate_size=32,
                 vllm_config: VllmConfig = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty((intermediate_size, hidden_size)))
        self.norm = RMSNorm(intermediate_size, 1e-05)
        # Initialize weights
        torch.nn.init.normal_(self.gate_proj, std=0.02)

    def forward(self, hidden_states, residual):
        """
        Forward pass implementing the operations in the FX graph
        
        Args:
            hidden_states: Input tensor
            residual: Residual tensor from previous layer
            
        Returns:
            Tuple containing the output tensor
        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)

        #matrix multiplication
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)

        # Tensor parallel all-reduce
        all_reduce = tensor_model_parallel_all_reduce(mm)

        # layer normalization
        norm_output, residual_output = self.norm(all_reduce, residual)

        return norm_output, residual_output

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.reduce_scatter.default,
            torch.ops.vllm.all_gather.default
        ]

    def ops_in_model(self):
        return [torch.ops._C.fused_add_rms_norm.default]


class TestQuantModel(torch.nn.Module):

    def __init__(self,
                 hidden_size=16,
                 intermediate_size=32,
                 vllm_config: VllmConfig = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vllm_config = vllm_config
        self.gate_proj = torch.nn.Parameter(torch.empty(
            (intermediate_size, hidden_size)),
                                            requires_grad=False)
        self.norm = RMSNorm(intermediate_size, 1e-05)
        # Initialize weights
        torch.nn.init.normal_(self.gate_proj, std=0.02)

        self.fp8_linear = Fp8LinearOp(cutlass_fp8_supported=True,
                                      use_per_token_if_dynamic=False)

        self.scale = torch.rand(1, dtype=torch.float32)
        # Create a weight that is compatible with torch._scaled_mm,
        # which expects a column-major layout.
        self.w = torch.rand(hidden_size,
                            intermediate_size).to(dtype=FP8_DTYPE).t()
        self.wscale = torch.rand(1, dtype=torch.float32)

    def forward(self, hidden_states, residual):
        """
        Forward pass implementing the operations in the FX graph
        
        Args:
            hidden_states: Input tensor
            residual: Residual tensor from previous layer
            
        Returns:
            Tuple containing the output tensor
        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)

        #matrix multiplication
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)

        # Tensor parallel all-reduce
        all_reduce = tensor_model_parallel_all_reduce(mm)

        # layer normalization
        norm_output, residual_output = self.norm(all_reduce, residual)

        # for static input quantization
        # self.fp8_linear is initialized with use_per_token_if_dynamic=False
        fp8_linear_result = self.fp8_linear.apply(norm_output,
                                                  self.w,
                                                  self.wscale,
                                                  input_scale=self.scale.to(
                                                      norm_output.device))

        return fp8_linear_result, residual_output

    def ops_in_model_before(self):
        ops_to_remove = [torch.ops.vllm.all_reduce.default
                         ]  # Always removed by SP
        # The following are only removed if fusion happens
        if self.vllm_config and self.vllm_config.compilation_config \
            .pass_config.enable_fusion:
            ops_to_remove.extend([
                torch.ops._C.fused_add_rms_norm.default,
                torch.ops._C.static_scaled_fp8_quant.default,
            ])
        return ops_to_remove

    def ops_in_model_after(self):
        ops_to_add = [
            torch.ops.vllm.reduce_scatter.default,
            torch.ops.vllm.all_gather.default
        ]
        # The following is only added if fusion happens
        if self.vllm_config and self.vllm_config.compilation_config \
            .pass_config.enable_fusion:
            ops_to_add.append(
                torch.ops._C.fused_add_rms_norm_static_fp8_quant.default)
        return ops_to_add

    def ops_in_model(self):
        if self.vllm_config and self.vllm_config.compilation_config \
            .pass_config.enable_fusion:
            # If fusion happens, the fused op is the one
            # we check for (de)functionalization
            return [torch.ops._C.fused_add_rms_norm_static_fp8_quant.default
                    ]  # noqa: E501
        else:
            # If no fusion, the original ops are checked
            return [
                torch.ops._C.fused_add_rms_norm.default,
                # TODO  functionalization pass does not handle this yet
                # torch.ops._C.static_scaled_fp8_quant.default,
            ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model_cls", [TestModel, TestQuantModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_fusion", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_sequence_parallelism_pass(test_model_cls: type[torch.nn.Module],
                                   batch_size: int, seq_len: int,
                                   hidden_size: int, dtype: torch.dtype,
                                   enable_fusion: bool):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(fn,
                                    args=(num_processes, test_model_cls,
                                          batch_size, seq_len, hidden_size,
                                          dtype, enable_fusion),
                                    nprocs=nprocs)

    run_torch_spawn(sequence_parallelism_pass_on_test_model, num_processes)


def sequence_parallelism_pass_on_test_model(
        local_rank: int, world_size: int,
        test_model_cls: type[torch.nn.Module], batch_size: int, seq_len: int,
        hidden_size: int, dtype: torch.dtype, enable_fusion: bool):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345',
    })

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # configure vllm config for SequenceParallelismPass
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(pass_config=PassConfig(
        enable_sequence_parallelism=True,
        enable_fusion=enable_fusion,
        enable_noop=True))  # NoOp needed for fusion
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"
    vllm_config.model_config = ModelConfig(model=model_name,
                                           task="auto",
                                           tokenizer=model_name,
                                           tokenizer_mode="auto",
                                           trust_remote_code=True,
                                           dtype=dtype,
                                           seed=42)

    sequence_parallelism_pass = SequenceParallelismPass(vllm_config)
    noop_pass = NoOpEliminationPass(vllm_config)
    func_pass = FixFunctionalizationPass(vllm_config)

    passes_for_backend = [noop_pass, sequence_parallelism_pass]

    if enable_fusion:
        fusion_pass = FusionPass.instance(vllm_config)
        passes_for_backend.append(fusion_pass)

    backend_no_func = TestBackend(*passes_for_backend)
    backend_func = TestBackend(*passes_for_backend, func_pass)

    model = test_model_cls(hidden_size,
                           hidden_size * 2,
                           vllm_config=vllm_config)

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype)
    residual = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)

    compiled_model_no_func = torch.compile(model, backend=backend_no_func)
    compiled_model_no_func(hidden_states, residual)
    compiled_model_func = torch.compile(model, backend=backend_func)
    compiled_model_func(hidden_states, residual)

    # In pre-nodes, all reduce should be there,
    # reduce scatter and all gather should not
    backend_no_func.check_before_ops(model.ops_in_model_before())

    # In post-nodes, reduce scatter and all gather should be there,
    # all reduce should not
    backend_no_func.check_after_ops(model.ops_in_model_after())

    # check if the functionalization pass is applied
    for op in model.ops_in_model():
        find_auto_fn(backend_no_func.graph_post_pass.nodes, op)
        assert find_auto_fn_maybe(backend_func.graph_post_pass.nodes,
                                  op) is None  # noqa: E501

    # make sure the ops were all de-functionalized
    found = dict()
    for node in backend_func.graph_post_pass.nodes:
        for op in model.ops_in_model():
            if is_func(node, op):
                found[op] = True
    assert all(found[op] for op in model.ops_in_model())
