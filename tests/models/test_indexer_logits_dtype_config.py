# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the per-model sparse-indexer logits dtype default."""

from types import SimpleNamespace

import pytest

import vllm.utils.deep_gemm as deep_gemm
from vllm.config import AttentionConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP


@pytest.mark.parametrize(
    "arch,sm100_default",
    [
        ("DeepseekV32ForCausalLM", "bfloat16"),
        ("DeepseekV4ForCausalLM", "bfloat16"),
        ("DeepseekV41ForCausalLM", "bfloat16"),
        ("GlmMoeDsaForCausalLM", "float32"),
        ("Glm5NextForCausalLM", "float32"),
    ],
)
@pytest.mark.parametrize("sm100", [True, False])
@pytest.mark.parametrize("requested", ["auto", "float32", "bfloat16"])
def test_indexer_logits_dtype_follows_model_default_unless_set(
    monkeypatch, arch: str, sm100_default: str, sm100: bool, requested: str
) -> None:
    monkeypatch.setattr(deep_gemm, "mqa_logits_use_mx_scales", lambda: sm100)
    attention_config = AttentionConfig(
        indexer_logits_dtype=requested, indexer_sparse_logits=False
    )
    vllm_config = SimpleNamespace(
        attention_config=attention_config,
        # A non-auto KV dtype keeps the hooks off their platform-specific paths.
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=2048),
            hf_text_config=SimpleNamespace(index_topk=2048),
        ),
        parallel_config=SimpleNamespace(set_dcp_defaults=lambda **_: None),
        speculative_config=None,
    )

    MODELS_CONFIG_MAP[arch].verify_and_update_config(vllm_config)

    # The indexer layers resolve a remaining "auto" to float32.
    default = sm100_default if sm100 else "float32"
    expected = default if requested == "auto" else requested
    assert attention_config.resolve_indexer_logits_dtype("float32") == expected
