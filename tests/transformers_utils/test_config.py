# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    get_config,
    maybe_override_with_speculators,
    try_get_generation_config,
)


def _patch_config_dict(monkeypatch, config_module, config_dict):
    calls = []

    def fake_get_config_dict(model, revision=None, token=None, **kwargs):
        calls.append((model, revision, kwargs))
        return config_dict, {}

    monkeypatch.setattr(
        config_module.PretrainedConfig,
        "get_config_dict",
        fake_get_config_dict,
    )
    return calls


def _patch_config_source(monkeypatch, config_module, source):
    monkeypatch.setattr(
        config_module, "resolve_gguf_config_source", lambda *args, **kwargs: source
    )


def _patch_gguf_file(monkeypatch, config_module, filename):
    monkeypatch.setattr(
        config_module, "get_gguf_file_path_from_hf", lambda *args, **kwargs: filename
    )


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_remote_gguf_speculators_uses_resolved_config_source(monkeypatch):
    from vllm.transformers_utils import config as config_module

    _patch_config_source(monkeypatch, config_module, "org/base")
    calls = _patch_config_dict(
        monkeypatch,
        config_module,
        {
            "speculators_model_type": "eagle3",
            "transformer_layer_config": {},
            "speculators_config": {
                "proposal_methods": [{"speculative_tokens": 2}],
                "verifier": {"name_or_path": "org/target"},
            },
        },
    )

    model, tokenizer, speculative_config = maybe_override_with_speculators(
        "org/spec-GGUF:UD-IQ4_NL",
        tokenizer=None,
        trust_remote_code=False,
        revision="gguf-rev",
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert model == "org/target"
    assert tokenizer == "org/target"
    assert speculative_config == {
        "method": "eagle3",
        "num_speculative_tokens": 2,
        "model": "org/spec-GGUF:UD-IQ4_NL",
    }


def test_remote_gguf_speculators_uses_gguf_parser_fallback(monkeypatch):
    from vllm.transformers_utils import config as config_module

    _patch_config_source(monkeypatch, config_module, "org/spec-GGUF")
    _patch_gguf_file(monkeypatch, config_module, "spec-UD-IQ4_NL.gguf")
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: False,
    )
    calls = _patch_config_dict(
        monkeypatch,
        config_module,
        {"model_type": "qwen3_5_moe"},
    )

    model, tokenizer, speculative_config = maybe_override_with_speculators(
        "org/spec-GGUF:UD-IQ4_NL",
        tokenizer=None,
        trust_remote_code=False,
        revision="gguf-rev",
    )

    assert calls == [
        (
            "org/spec-GGUF",
            "gguf-rev",
            {
                "gguf_file": "spec-UD-IQ4_NL.gguf",
                "local_files_only": False,
            },
        )
    ]
    assert model == "org/spec-GGUF:UD-IQ4_NL"
    assert tokenizer is None
    assert speculative_config is None


def test_local_gguf_speculators_uses_explicit_hf_config_path(monkeypatch):
    from vllm.transformers_utils import config as config_module

    calls = _patch_config_dict(
        monkeypatch,
        config_module,
        {"model_type": "qwen3_5_moe"},
    )

    model, tokenizer, speculative_config = maybe_override_with_speculators(
        "/models/qwen.gguf",
        tokenizer="org/base",
        trust_remote_code=False,
        hf_config_path="org/base",
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert model == "/models/qwen.gguf"
    assert tokenizer == "org/base"
    assert speculative_config is None


def test_local_gguf_speculators_uses_resolved_config_source(monkeypatch):
    from vllm.transformers_utils import config as config_module

    _patch_config_source(monkeypatch, config_module, "org/base")
    monkeypatch.setattr(
        config_module,
        "check_gguf_file",
        lambda model: model == "/models/qwen.gguf",
    )
    calls = _patch_config_dict(
        monkeypatch,
        config_module,
        {"model_type": "qwen3_5_moe"},
    )

    model, tokenizer, speculative_config = maybe_override_with_speculators(
        "/models/qwen.gguf",
        tokenizer=None,
        trust_remote_code=False,
        revision="local-rev",
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert model == "/models/qwen.gguf"
    assert tokenizer is None
    assert speculative_config is None


def test_remote_gguf_get_config_uses_gguf_parser_fallback(monkeypatch):
    from transformers import PretrainedConfig

    from vllm.transformers_utils import config as config_module

    calls = []

    class FakeParser:
        def parse(
            self,
            model,
            trust_remote_code,
            revision=None,
            code_revision=None,
            hf_overrides=None,
            **kwargs,
        ):
            calls.append((model, revision, kwargs))
            return {}, PretrainedConfig(architectures=["FakeModel"])

    _patch_config_source(monkeypatch, config_module, "org/model-GGUF")
    _patch_gguf_file(monkeypatch, config_module, "model-UD-IQ4_NL.gguf")
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: False,
    )
    monkeypatch.setattr(
        config_module,
        "is_mistral_model_repo",
        lambda model_name_or_path, revision=None: False,
    )
    monkeypatch.setattr(
        config_module,
        "get_config_parser",
        lambda config_format: FakeParser(),
    )

    config = get_config(
        "org/model-GGUF:UD-IQ4_NL",
        trust_remote_code=False,
        revision="gguf-rev",
    )

    assert calls == [
        (
            "org/model-GGUF",
            "gguf-rev",
            {"gguf_file": "model-UD-IQ4_NL.gguf"},
        )
    ]
    assert config.architectures == ["FakeModel"]
