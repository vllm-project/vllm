# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import pytest


@pytest.mark.cpu_test
@pytest.mark.parametrize("block_size", [None, 4096, 16])
@pytest.mark.parametrize("prefix_caching", [True, False])
def test_cpu_mla_hybrid_cache_config(block_size, prefix_caching):
    """CPU MLA must reconcile derived Mamba defaults without hiding bad inputs."""
    # Select the CPU platform in a fresh process, including on GPU test hosts.
    code = """
import sys

import pytest

from vllm.config import (
    AttentionConfig, CacheConfig, DeviceConfig, ModelConfig, VllmConfig,
)

block_size = None if sys.argv[1] == "None" else int(sys.argv[1])
prefix_caching = sys.argv[2] == "True"
model_config = ModelConfig(
    "moonshotai/Kimi-K3",
    trust_remote_code=True,
    skip_tokenizer_init=True,
    max_model_len=4096,
)

def make_config():
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            enable_prefix_caching=prefix_caching,
            mamba_block_size=block_size,
        ),
        attention_config=AttentionConfig(backend="CPU_MLA"),
        device_config=DeviceConfig(device="cpu"),
    )

if block_size == 16:
    with pytest.raises(ValueError, match="mamba-block-size"):
        make_config()
else:
    config = make_config()
    assert config.model_config.is_hybrid
    assert not config.cache_config.enable_prefix_caching
    assert not config.scheduler_config.enable_chunked_prefill
    assert config.cache_config.mamba_cache_mode == "none"
    assert config.cache_config.mamba_block_size == 4096
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(block_size), str(prefix_caching)],
        env={
            **os.environ,
            "VLLM_TARGET_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
