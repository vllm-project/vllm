# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fx_utils import (find_auto_fn, find_auto_fn_maybe,
                                       find_specified_fn,
                                       find_specified_fn_maybe, is_func)
from vllm.compilation.sequence_parallelism import SequenceParallelismPass
from vllm.config import (CompilationConfig, DeviceConfig, ModelConfig,
                         VllmConfig)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..utils import multi_gpu_test
from .backend import TestBackend

OPS_IN_MODEL_BEFORE = [
    torch.ops.vllm.all_reduce.default,
]

OPS_IN_MODEL_AFTER = [
    torch.ops.vllm.reduce_scatter.default,
    torch.ops.vllm.all_gather.default,
]

OPS_IN_MODEL = [torch.ops._C.fused_add_rms_norm.default]

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestModel(torch.nn.Module):

    def __init__(self, hidden_size=16, intermediate_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty((intermediate_size, hidden_size)))
        self.norm = RMSNorm(hidden_size, 1e-05)
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


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_sequence_parallelism_pass(batch_size: int, seq_len: int,
                                   hidden_size: int, dtype: torch.dtype):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(fn,
                                    args=(num_processes, batch_size, seq_len,
                                          hidden_size, dtype),
                                    nprocs=nprocs)

    run_torch_spawn(sequence_parallelism_pass_on_test_model, num_processes)


def sequence_parallelism_pass_on_test_model(local_rank: int, world_size: int,
                                            batch_size: int, seq_len: int,
                                            hidden_size: int,
                                            dtype: torch.dtype):
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
    vllm_config.compilation_config = CompilationConfig(
        pass_config=CompilationConfig.PassConfig(
            enable_sequence_parallelism=True, ), )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"
    vllm_config.model_config = ModelConfig(model=model,
                                           task="auto",
                                           tokenizer=model,
                                           tokenizer_mode="auto",
                                           trust_remote_code=True,
                                           dtype=dtype,
                                           seed=42)

    sequence_parallelism_pass = SequenceParallelismPass(vllm_config)
    backend_no_func = TestBackend(sequence_parallelism_pass)
    func_pass = FixFunctionalizationPass(vllm_config)
    backend_func = TestBackend(sequence_parallelism_pass, func_pass)

    model = TestModel(hidden_size, hidden_size * 2)
    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype)
    residual = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)

    compiled_model_no_func = torch.compile(model, backend=backend_no_func)
    compiled_model_no_func(hidden_states, residual)
    compiled_model_func = torch.compile(model, backend=backend_func)
    compiled_model_func(hidden_states, residual)

    # Check substitution worked
    pre_nodes = backend_no_func.graph_pre_pass.nodes
    post_nodes = backend_no_func.graph_post_pass.nodes

    # In pre-nodes, all reduce should be there,
    # reduce scatter and all gather should not
    for op in OPS_IN_MODEL_BEFORE:
        find_specified_fn(pre_nodes, op)
    for op in OPS_IN_MODEL_AFTER:
        assert find_specified_fn_maybe(pre_nodes, op) is None

    # In post-nodes, reduce scatter and all gather should be there,
    # all reduce should not
    for op in OPS_IN_MODEL_AFTER:
        find_specified_fn(post_nodes, op)
    for op in OPS_IN_MODEL_BEFORE:
        assert find_specified_fn_maybe(post_nodes, op) is None

    # check if the functionalization pass is applied
    for op in OPS_IN_MODEL:
        find_auto_fn(backend_no_func.graph_post_pass.nodes, op)
        assert find_auto_fn_maybe(backend_func.graph_post_pass.nodes,
                                  op) is None  # noqa: E501

    # make sure the ops were all de-functionalized
    found = dict()
    for node in backend_func.graph_post_pass.nodes:
        for op in OPS_IN_MODEL:
            if is_func(node, op):
                found[op] = True
    assert all(found[op] for op in OPS_IN_MODEL)
