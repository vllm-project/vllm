# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.collective_fusion import AsyncTPPass
from vllm.compilation.fx_utils import (find_specified_fn,
                                       find_specified_fn_maybe)
from vllm.config import (CompilationConfig, DeviceConfig, ModelConfig,
                         PassConfig, VllmConfig)
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter)
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..utils import multi_gpu_test
from .backend import TestBackend

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestMMRSModel(torch.nn.Module):

    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj = torch.nn.Parameter(torch.empty(
            (self.hidden_size * 2, hidden_size)),
                                            requires_grad=False)
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

    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.empty(
            (hidden_size, hidden_size)),
                                         requires_grad=False)
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


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model", ["TestMMRSModel", "TestAGMMModel"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_sequence_parallelism_pass(test_model: str, batch_size: int,
                                   seq_len: int, hidden_size: int,
                                   dtype: torch.dtype):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(fn,
                                    args=(num_processes, test_model,
                                          batch_size, seq_len, hidden_size,
                                          dtype),
                                    nprocs=nprocs)

    run_torch_spawn(async_tp_pass_on_test_model, num_processes)


def async_tp_pass_on_test_model(local_rank: int, world_size: int,
                                test_model: str, batch_size: int, seq_len: int,
                                hidden_size: int, dtype: torch.dtype):
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
        enable_async_tp=True, ), )
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

    async_tp_pass = AsyncTPPass(vllm_config)
    backend = TestBackend(async_tp_pass)

    if test_model == "TestMMRSModel":
        model = TestMMRSModel(hidden_size)
    elif test_model == "TestAGMMModel":
        model = TestAGMMModel(hidden_size)
    else:
        raise ValueError(f"Unknown model: {test_model}")

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype,
                                requires_grad=False)

    compiled_model = torch.compile(model, backend=backend)
    compiled_model(hidden_states)

    # Check substitution worked
    pre_nodes = backend.graph_pre_pass.nodes
    post_nodes = backend.graph_post_pass.nodes

    # In pre-nodes, all reduce should exist,
    # fused_matmul_reduce_scatter or fused_all_gather_matmul should not
    for op in model.ops_in_model_before():
        find_specified_fn(pre_nodes, op)
    for op in model.ops_in_model_after():
        assert find_specified_fn_maybe(pre_nodes, op) is None

    # In post-nodes, fused_matmul_reduce_scatter or \
    # fused_all_gather_matmul should exist
    for op in model.ops_in_model_after():
        find_specified_fn(post_nodes, op)
