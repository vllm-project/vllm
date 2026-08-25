# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import IquestMoeSinkAttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_vllm_config(
    *,
    fa3_sink_mode="auto",
    disable_sink_attention=False,
    backend=AttentionBackendEnum.FLASH_ATTN,
):
    hf_config = SimpleNamespace(enable_sink_attention=True, num_sink_tokens=1)
    model_config = SimpleNamespace(
        disable_cascade_attn=False,
        disable_sink_attention=disable_sink_attention,
        fa3_sink_mode=fa3_sink_mode,
        hf_config=hf_config,
        hf_text_config=hf_config,
    )
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=backend,
            flash_attn_version=3,
        ),
        model_config=model_config,
    )


def test_explicit_fa3_sink_mode_preserves_sink_and_disables_cascade():
    vllm_config = _make_vllm_config(fa3_sink_mode="unfused")

    IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)

    assert vllm_config.model_config.hf_config.enable_sink_attention is True
    assert vllm_config.model_config.hf_config.num_sink_tokens == 1
    assert vllm_config.model_config.disable_cascade_attn is True


def test_explicit_fa3_sink_mode_rejects_disabling_sink():
    vllm_config = _make_vllm_config(
        fa3_sink_mode="unfused", disable_sink_attention=True
    )

    with pytest.raises(ValueError, match="cannot be combined.*disable-sink"):
        IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)


def test_explicit_fa3_sink_mode_rejects_non_fa_backend():
    vllm_config = _make_vllm_config(
        fa3_sink_mode="unfused", backend=AttentionBackendEnum.FLEX_ATTENTION
    )

    with pytest.raises(ValueError, match="requires explicitly selecting.*FLASH_ATTN"):
        IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)


def test_explicit_fa3_sink_mode_requires_explicit_fa3():
    vllm_config = _make_vllm_config(fa3_sink_mode="unfused")
    vllm_config.attention_config.flash_attn_version = 2

    with pytest.raises(ValueError, match="flash_attn_version=3"):
        IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)


def test_explicit_fa3_sink_mode_requires_model_sink_attention():
    vllm_config = _make_vllm_config(fa3_sink_mode="unfused")
    vllm_config.model_config.hf_config.enable_sink_attention = False

    with pytest.raises(ValueError, match="model with sink attention enabled"):
        IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)


def test_explicit_fa3_sink_mode_requires_positive_sink_count():
    vllm_config = _make_vllm_config(fa3_sink_mode="unfused")
    vllm_config.model_config.hf_config.num_sink_tokens = 0

    with pytest.raises(ValueError, match="positive model-configured"):
        IquestMoeSinkAttentionConfig.verify_and_update_config(vllm_config)
