# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.collective_fusion import AllReduceFusionPass
from vllm.config import (CompilationConfig, CompilationLevel, DeviceConfig,
                         ModelConfig, PassConfig, VllmConfig)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

from ..utils import multi_gpu_test
from .backend import TestBackend


class TestAllReduceRMSNormModel(torch.nn.Module):

    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states, residual):
        view = hidden_states.reshape(-1, self.hidden_size)
        all_reduce = tensor_model_parallel_all_reduce(view)
        norm = self.norm(all_reduce)
        return norm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]


class TestAllReduceFusedAddRMSNormModel(torch.nn.Module):

    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states, residual):
        view = hidden_states.reshape(-1, self.hidden_size)
        all_reduce = tensor_model_parallel_all_reduce(view)
        norm, _ = self.norm(all_reduce, residual)
        return norm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model",
    [TestAllReduceRMSNormModel, TestAllReduceFusedAddRMSNormModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
@pytest.mark.skipif(not find_spec("flashinfer"),
                    reason="flashinfer is not installed")
@pytest.mark.skipif(not current_platform.is_device_capability(100),
                    reason="Only test on SM100")
def test_all_reduce_fusion_pass_replace(test_model: torch.nn.Module,
                                        batch_size: int, seq_len: int,
                                        hidden_size: int, dtype: torch.dtype):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(fn,
                                    args=(num_processes, test_model,
                                          batch_size, seq_len, hidden_size,
                                          dtype),
                                    nprocs=nprocs)

    run_torch_spawn(all_reduce_fusion_pass_on_test_model, num_processes)


def all_reduce_fusion_pass_on_test_model(local_rank: int, world_size: int,
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

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(level=CompilationLevel.PIECEWISE,
                                             custom_ops=["+rms_norm"],
                                             compile_sizes=[2, 4, 8]))
    vllm_config.compilation_config.pass_config = PassConfig(
        enable_fi_allreduce_fusion=True)
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

    all_reduce_fusion_pass = AllReduceFusionPass(vllm_config)
    backend = TestBackend(all_reduce_fusion_pass)

    model = test_model_cls(hidden_size)

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                requires_grad=False)
    residual = torch.randn((batch_size * seq_len, hidden_size),
                           requires_grad=False)

    compiled_model = torch.compile(model, backend=backend)
    compiled_model(hidden_states, residual)

    backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
    backend.check_after_ops(model.ops_in_model_after())
    del all_reduce_fusion_pass
