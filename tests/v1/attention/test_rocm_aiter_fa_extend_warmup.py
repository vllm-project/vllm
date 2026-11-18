# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER flash-attention backend warms the kernels only an extend reaches."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends import rocm_aiter_fa
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionExtendKernel

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

NO_WINDOW = (-1, -1)


def _keys(**overrides):
    kwargs = dict(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=NO_WINDOW,
        has_alibi=False,
        has_sinks=False,
    )
    kwargs.update(overrides)
    return AiterFlashAttentionExtendKernel().get_warmup_keys(**kwargs)


def test_sliding_window_layers_have_nothing_to_warm():
    """A window extend runs the same LSE-free call a new prompt does."""
    assert _keys(sliding_window=(1023, 0)) == []


def test_compile_runs_the_calls_extend_forward_makes(monkeypatch):
    calls = []
    ops = MagicMock()
    ops.flash_attn_varlen_func.side_effect = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setattr(rocm_aiter_fa, "rocm_aiter_ops", ops)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    kernel = AiterFlashAttentionExtendKernel()
    keys = kernel.get_warmup_keys(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=NO_WINDOW,
        has_alibi=True,
        has_sinks=True,
    )
    kernel.compile_many(keys)

    causal, context = calls
    for call, is_causal in ((causal, True), (context, False)):
        assert call["causal"] is is_causal
        assert call["return_lse"]
        assert call["min_seqlen_q"] == 0
        assert call["window_size"] == NO_WINDOW
        assert call["alibi_slopes"] is not None
        # AITER treats a one-token query as decode and drops `causal`.
        assert call["max_seqlen_q"] > 1
        assert call["q"].shape[1:] == (4, 64)
        assert call["k"].shape[1:] == (1, 64)
    assert causal["sink_ptr"] is not None and context["sink_ptr"] is None
    assert causal["max_seqlen_k"] == causal["max_seqlen_q"]
    assert context["max_seqlen_k"] > context["max_seqlen_q"]


@pytest.mark.parametrize("enable_jit_warmup", [True, False])
def test_builder_registers_the_extend_warmup(monkeypatch, enable_jit_warmup):
    """Selecting the backend registers the warmup the runner later performs."""
    from tests.v1.attention.utils import (
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    config = create_vllm_config(dtype="bfloat16")
    config.kernel_config.enable_jit_warmup = enable_jit_warmup
    model_config = config.model_config
    head_size = model_config.get_head_size()
    registry = JitWarmupRegistry(config)
    with set_current_vllm_config(config):
        Attention(
            model_config.get_num_attention_heads(config.parallel_config),
            head_size,
            head_size**-0.5,
            model_config.get_num_kv_heads(config.parallel_config),
            cache_config=config.cache_config,
            prefix="layer",
            attn_backend=rocm_aiter_fa.AiterFlashAttentionBackend,
        )
        with registry.activate():
            rocm_aiter_fa.AiterFlashAttentionMetadataBuilder(
                create_standard_kv_cache_spec(config),
                ["layer"],
                config,
                torch.device(current_platform.device_type),
            )

    calls = []
    ops = MagicMock()
    ops.flash_attn_varlen_func.side_effect = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setattr(rocm_aiter_fa, "rocm_aiter_ops", ops)
    registry.warmup()

    assert [call["causal"] for call in calls] == (
        [True, False] if enable_jit_warmup else []
    )
    assert all(call["return_lse"] for call in calls)
