# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.collective_fusion import AsyncTPPass
from vllm.config import (CompilationConfig, DeviceConfig, ModelConfig,
                         PassConfig, VllmConfig)
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter)
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import (compare_two_settings, create_new_process_for_each_test,
                     multi_gpu_test)
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
@pytest.mark.parametrize("test_model", [TestMMRSModel, TestAGMMModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_async_tp_pass_replace(test_model: str, batch_size: int, seq_len: int,
                               hidden_size: int, dtype: torch.dtype):
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
                                test_model_cls: torch.nn.Module,
                                batch_size: int, seq_len: int,
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
    model_name = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"
    vllm_config.model_config = ModelConfig(model=model_name,
                                           task="auto",
                                           tokenizer=model_name,
                                           tokenizer_mode="auto",
                                           trust_remote_code=True,
                                           dtype=dtype,
                                           seed=42)

    async_tp_pass = AsyncTPPass(vllm_config)
    backend = TestBackend(async_tp_pass)

    model = test_model_cls(hidden_size)

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype,
                                requires_grad=False)

    compiled_model = torch.compile(model, backend=backend)
    compiled_model(hidden_states)

    # In pre-nodes, all gather or reduce scatter should exist,
    # fused_matmul_reduce_scatter or fused_all_gather_matmul should not
    backend.check_before_ops(model.ops_in_model_before(),
                             ops_fully_replaced=False)

    # In post-nodes, fused_matmul_reduce_scatter or \
    # fused_all_gather_matmul should exist
    backend.check_after_ops(model.ops_in_model_after())


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_id", ["meta-llama/Llama-3.2-1B-Instruct"])
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("async_tp_enabled", [True])
@pytest.mark.parametrize("distributed_backend", ["mp"])
@pytest.mark.parametrize("eager_mode", [False, True])
def test_async_tp_pass_correctness(
    model_id: str,
    tp_size: int,
    async_tp_enabled: bool,
    distributed_backend: str,
    eager_mode: bool,
    num_gpus_available: int,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    pp_size = 1
    if num_gpus_available < tp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")

    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
    ]
    if eager_mode:
        common_args.append("--enforce-eager")

    compilation_config = {
        'level': 3,
        'compile_sizes': [2, 4, 8],
        'splitting_ops': [],
        'pass_config': {
            'enable_async_tp': async_tp_enabled
        },
    }

    async_tp_env = tp_env = {
        "VLLM_USE_V1": "1",
    }

    aysnc_tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        distributed_backend,
        "--compilation_config",
        json.dumps(compilation_config),
    ]

    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
    ]

    compare_two_settings(model_id,
                         aysnc_tp_args,
                         tp_args,
                         async_tp_env,
                         tp_env,
                         method="generate")
