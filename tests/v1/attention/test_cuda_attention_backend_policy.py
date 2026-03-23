# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CUDA model-aware attention backend policy."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend


def setup_function():
    _get_backend_priorities.cache_clear()


def teardown_function():
    _get_backend_priorities.cache_clear()


def test_gpt_oss_sm89_prioritizes_triton_for_sink_attention():
    priorities = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(8, 9),
        model_architectures=("GptOssForCausalLM",),
        has_sink=True,
    )

    assert priorities == [
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


def test_gpt_oss_sm90_prioritizes_flash_attn_for_sink_attention():
    priorities = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(9, 0),
        model_architectures=("GptOssForCausalLM",),
        has_sink=True,
    )

    assert priorities == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


def test_gpt_oss_sm100_prioritizes_flashinfer_for_sink_attention():
    priorities = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(10, 0),
        model_architectures=("GptOssForCausalLM",),
        has_sink=True,
    )

    assert priorities == [
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


def test_non_gpt_oss_or_sinkless_configs_keep_generic_policy():
    priorities = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(8, 9),
        model_architectures=("GptOssForCausalLM",),
        has_sink=False,
    )

    assert priorities == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


def test_get_attn_backend_uses_declared_architectures_for_policy():
    sentinel = object()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            user_specified_block_size=False,
            block_size=16,
        ),
        model_config=SimpleNamespace(
            architectures=["GptOssForCausalLM"],
            architecture="StaleWrappedArchitecture",
        ),
        attention_config=SimpleNamespace(backend=None),
    )

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.v1.attention.selector._cached_get_attn_backend",
            return_value=sentinel,
        ) as mock_cached_get_attn_backend,
    ):
        backend = get_attn_backend(
            128,
            torch.bfloat16,
            None,
            has_sink=True,
        )

    assert backend is sentinel
    attn_selector_config = mock_cached_get_attn_backend.call_args.kwargs[
        "attn_selector_config"
    ]
    assert attn_selector_config.model_architectures == (
        "GptOssForCausalLM",
        "StaleWrappedArchitecture",
    )
