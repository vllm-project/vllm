# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype


@pytest.fixture
def dist_init():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_fused_qkv_a_proj_min_latency_gemm_disabled_under_batch_invariance(
    monkeypatch: pytest.MonkeyPatch, dist_init, batch_invariant: bool
) -> None:
    """The fused A GEMM kernel only handles up to 16 tokens, so it must not
    be selected in batch-invariant mode."""
    monkeypatch.setattr(deepseek_v2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    monkeypatch.setattr(deepseek_v2.envs, "VLLM_BATCH_INVARIANT", batch_invariant)

    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16):
        layer = deepseek_v2.DeepSeekV2FusedQkvAProjLinear(
            input_size=7168, output_size=[1536, 576], quant_config=None
        )

    assert layer.weight.dtype == torch.bfloat16
    assert tuple(layer.weight.shape) == (2112, 7168)
    assert layer._use_min_latency_gemm == (not batch_invariant)
