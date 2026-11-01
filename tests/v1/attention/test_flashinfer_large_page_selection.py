# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large-page support must account for both target and draft attention heads."""

import json

import pytest
from transformers import LlamaConfig

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import vllm.v1.attention.backends.flashinfer as fi
from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend

# The rule when using Blackwell and TRTLLM is:
# supports_large_pages <=> num_heads / kv_heads > 1
target_supports_large_pages = {"num_heads": 32, "num_kv_heads": 8}
draft_supports_large_pages = {"draft_num_heads": 16, "draft_num_kv_heads": 4}
draft_not_supports_large_pages = {"draft_num_heads": 16, "draft_num_kv_heads": 16}


def test_large_pages_supported_for_gqa_target(tmp_path, blackwell_with_trtllm):
    vllm_config = new_vllm_config(tmp_path, **target_supports_large_pages)
    assert FlashInferBackend.supports_large_pages(vllm_config)


def test_large_pages_not_supported_when_mha_draft_cannot_serve_them(
    tmp_path, blackwell_with_trtllm
):
    # E.g. GLM-5.3-Flash + DSpark
    # MLA/GQA target (ratio > 1) with an MHA drafter (ratio 1).
    vllm_config = new_vllm_config(
        tmp_path,
        **target_supports_large_pages,
        **draft_not_supports_large_pages,
    )
    assert not FlashInferBackend.supports_large_pages(vllm_config)


def test_large_pages_supported_when_draft_also_gqa(tmp_path, blackwell_with_trtllm):
    vllm_config = new_vllm_config(
        tmp_path,
        **target_supports_large_pages,
        **draft_supports_large_pages,
    )
    assert FlashInferBackend.supports_large_pages(vllm_config)


@pytest.fixture
def blackwell_with_trtllm(monkeypatch):
    """Neutralize platform and kernel availability checks."""
    monkeypatch.setattr(
        fi.current_platform,
        "is_device_capability_family",
        lambda family: family == 100,
    )
    monkeypatch.setattr(fi, "can_use_trtllm_attention", lambda *a, **kw: True)


def new_vllm_config(
    tmp_path, num_heads, num_kv_heads, draft_num_heads=None, draft_num_kv_heads=None
) -> VllmConfig:
    """Build a real VllmConfig with a GQA or MHA target and optional draft."""
    model_config = ModelConfig(
        model=_write_model(tmp_path / "target", num_heads, num_kv_heads),
        tokenizer_mode="skip",
    )
    if draft_num_heads is None:
        speculative_config = None
    else:
        speculative_config = SpeculativeConfig(
            model=_write_model(tmp_path / "draft", draft_num_heads, draft_num_kv_heads),
            method="dspark",
            num_speculative_tokens=8,
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(),
        )
    return VllmConfig(model_config=model_config, speculative_config=speculative_config)


def _write_model(path, num_heads, num_kv_heads) -> str:
    """Write a minimal offline HF config.json with the given head counts."""
    path.mkdir()
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=128,
        max_position_embeddings=128,
    )
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    return str(path)
