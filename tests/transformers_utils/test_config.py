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


def _patch_gguf_reader(monkeypatch, gguf_utils_module, metadata):
    class FakeField:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class FakeReader:
        def __init__(self, model):
            self.model = model

        def get_field(self, key):
            if key not in metadata:
                return None
            return FakeField(metadata[key])

    monkeypatch.setattr(gguf_utils_module.gguf, "GGUFReader", FakeReader)


def test_local_gguf_mtp_metadata_patches_qwen_config(monkeypatch):
    from transformers import PretrainedConfig

    from vllm.transformers_utils import gguf_utils

    gguf_utils._get_local_gguf_mtp_metadata.cache_clear()
    _patch_gguf_reader(
        monkeypatch,
        gguf_utils,
        {
            "general.architecture": "qwen35moe",
            "qwen35moe.nextn_predict_layers": [1],
        },
    )
    monkeypatch.setattr(gguf_utils, "check_gguf_file", lambda model: True)

    config = PretrainedConfig(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )

    patched_config = gguf_utils.maybe_patch_mtp_config_from_gguf(
        "/models/qwen-mtp.gguf",
        config,
    )

    assert patched_config.mtp_num_hidden_layers == 1
    gguf_utils._get_local_gguf_mtp_metadata.cache_clear()


def test_local_gguf_mtp_metadata_patches_gemma_config(monkeypatch):
    from transformers import PretrainedConfig

    from vllm.transformers_utils import gguf_utils

    gguf_utils._get_local_gguf_mtp_metadata.cache_clear()
    _patch_gguf_reader(
        monkeypatch,
        gguf_utils,
        {
            "general.architecture": "gemma4-assistant",
            "gemma4-assistant.nextn_predict_layers": [4],
            "gemma4-assistant.embedding_length": [1024],
            "gemma4-assistant.embedding_length_out": [2816],
            "gemma4-assistant.feed_forward_length": [8192],
            "gemma4-assistant.attention.head_count": [16],
            "gemma4-assistant.attention.head_count_kv": [8, 8, 8, 2],
            "gemma4-assistant.attention.key_length": [512],
            "gemma4-assistant.attention.key_length_swa": [256],
            "gemma4-assistant.attention.sliding_window": [1024],
            "gemma4-assistant.attention.sliding_window_pattern": [
                True,
                True,
                True,
                False,
            ],
        },
    )
    monkeypatch.setattr(gguf_utils, "check_gguf_file", lambda model: True)

    config = PretrainedConfig(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
    )

    patched_config = gguf_utils.maybe_patch_mtp_config_from_gguf(
        "/models/gemma-mtp.gguf",
        config,
    )

    assert patched_config.model_type == "gemma4_assistant"
    assert patched_config.num_hidden_layers == 4
    assert patched_config.hidden_size == 1024
    assert patched_config.backbone_hidden_size == 2816
    assert patched_config.intermediate_size == 8192
    assert patched_config.num_attention_heads == 16
    assert patched_config.num_key_value_heads == 8
    assert patched_config.num_global_key_value_heads == 2
    assert patched_config.head_dim == 256
    assert patched_config.global_head_dim == 512
    assert patched_config.sliding_window == 1024
    assert patched_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    gguf_utils._get_local_gguf_mtp_metadata.cache_clear()


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

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "org/spec-GGUF:UD-IQ4_NL",
            tokenizer=None,
            trust_remote_code=True,
            revision="gguf-rev",
        )
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert trust_remote_code is False
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

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "org/spec-GGUF:UD-IQ4_NL",
            tokenizer=None,
            trust_remote_code=False,
            revision="gguf-rev",
        )
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
    assert trust_remote_code is False
    assert model == "org/spec-GGUF:UD-IQ4_NL"
    assert tokenizer is None
    assert speculative_config is None


def test_remote_gguf_parser_fallback_without_speculators_disables_trust_remote_code(
    monkeypatch,
):
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

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "org/spec-GGUF:UD-IQ4_NL",
            tokenizer=None,
            trust_remote_code=True,
            revision="gguf-rev",
        )
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
    assert trust_remote_code is False
    assert model == "org/spec-GGUF:UD-IQ4_NL"
    assert tokenizer is None
    assert speculative_config is None


def test_remote_gguf_parser_speculators_disables_trust_remote_code(monkeypatch):
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
        {
            "speculators_model_type": "eagle3",
            "transformer_layer_config": {},
            "speculators_config": {
                "proposal_methods": [{"speculative_tokens": 2}],
                "verifier": {"name_or_path": "org/target"},
            },
        },
    )

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "org/spec-GGUF:UD-IQ4_NL",
            tokenizer=None,
            trust_remote_code=True,
            revision="gguf-rev",
        )
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
    assert trust_remote_code is False
    assert model == "org/target"
    assert tokenizer == "org/target"
    assert speculative_config == {
        "method": "eagle3",
        "num_speculative_tokens": 2,
        "model": "org/spec-GGUF:UD-IQ4_NL",
    }


def test_local_gguf_speculators_uses_explicit_hf_config_path(monkeypatch):
    from vllm.transformers_utils import config as config_module

    calls = _patch_config_dict(
        monkeypatch,
        config_module,
        {"model_type": "qwen3_5_moe"},
    )

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "/models/qwen.gguf",
            tokenizer="org/base",
            trust_remote_code=False,
            hf_config_path="org/base",
        )
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert trust_remote_code is False
    assert model == "/models/qwen.gguf"
    assert tokenizer == "org/base"
    assert speculative_config is None


def test_local_gguf_explicit_hf_config_path_preserves_trust_remote_code(
    monkeypatch,
):
    from vllm.transformers_utils import config as config_module

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

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "/models/qwen.gguf",
            tokenizer="org/base",
            trust_remote_code=True,
            hf_config_path="org/base",
        )
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert trust_remote_code is True
    assert model == "org/target"
    assert tokenizer == "org/target"
    assert speculative_config == {
        "method": "eagle3",
        "num_speculative_tokens": 2,
        "model": "/models/qwen.gguf",
    }


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

    model, tokenizer, speculative_config, trust_remote_code = (
        maybe_override_with_speculators(
            "/models/qwen.gguf",
            tokenizer=None,
            trust_remote_code=False,
            revision="local-rev",
        )
    )

    assert calls == [("org/base", None, {"local_files_only": False})]
    assert trust_remote_code is False
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


def test_remote_gguf_get_config_disables_trust_for_metadata_base(
    monkeypatch,
):
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
            calls.append((model, trust_remote_code, revision, kwargs))
            return {}, PretrainedConfig(architectures=["FakeModel"])

    _patch_config_source(monkeypatch, config_module, "org/base")
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: (
            model == "org/base"
            and config_name == config_module.HF_CONFIG_NAME
            and revision is None
        ),
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

    get_config(
        "org/model-GGUF:UD-IQ4_NL",
        trust_remote_code=True,
        revision="gguf-rev",
    )

    assert calls == [("org/base", False, None, {})]


def test_local_gguf_get_config_disables_trust_for_metadata_base(
    monkeypatch,
):
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
            calls.append((model, trust_remote_code, revision, kwargs))
            return {}, PretrainedConfig(architectures=["FakeModel"])

    _patch_config_source(monkeypatch, config_module, "org/base")
    monkeypatch.setattr(
        config_module,
        "is_gguf",
        lambda model: model == "/models/qwen.gguf",
    )
    monkeypatch.setattr(
        config_module,
        "check_gguf_file",
        lambda model: model == "/models/qwen.gguf",
    )
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: (
            model == "org/base"
            and config_name == config_module.HF_CONFIG_NAME
            and revision is None
        ),
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

    get_config(
        "/models/qwen.gguf",
        trust_remote_code=True,
        revision="local-rev",
    )

    assert calls == [("org/base", False, None, {})]


def test_local_gguf_get_config_patches_mtp_before_speculative_override(
    monkeypatch,
):
    from transformers import PretrainedConfig

    from vllm.config.speculative import SpeculativeConfig
    from vllm.transformers_utils import config as config_module

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
            return {}, PretrainedConfig(
                model_type="qwen3_5_moe",
                architectures=["Qwen3_5MoeForConditionalGeneration"],
            )

    _patch_config_source(monkeypatch, config_module, "/models")
    monkeypatch.setattr(
        config_module,
        "is_gguf",
        lambda model: model == "/models/qwen-mtp.gguf",
    )
    monkeypatch.setattr(
        config_module,
        "check_gguf_file",
        lambda model: model == "/models/qwen-mtp.gguf",
    )
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: (
            model == "/models" and config_name == config_module.HF_CONFIG_NAME
        ),
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

    def fake_patch_mtp_config(model, config):
        config.update({"mtp_num_hidden_layers": 1})
        return config

    monkeypatch.setattr(
        config_module,
        "maybe_patch_mtp_config_from_gguf",
        fake_patch_mtp_config,
    )

    config = get_config(
        "/models/qwen-mtp.gguf",
        trust_remote_code=False,
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )

    assert config.model_type == "qwen3_5_mtp"
    assert config.n_predict == 1
    assert config.architectures == ["Qwen3_5MoeMTP"]


def test_local_nested_gguf_mtp_uses_parent_config_before_speculative_override(
    monkeypatch,
):
    from pathlib import Path

    from transformers import PretrainedConfig

    from vllm.config.speculative import SpeculativeConfig
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
            calls.append((model, trust_remote_code, revision, kwargs))
            return {}, PretrainedConfig(
                model_type="gemma4",
                architectures=["Gemma4ForConditionalGeneration"],
            )

    gguf_file = "/models/gemma-gguf/MTP/gemma-mtp.gguf"
    gguf_repo = Path("/models/gemma-gguf/MTP")
    parent_repo = Path("/models/gemma-gguf")
    monkeypatch.setattr(
        config_module,
        "resolve_gguf_config_source",
        lambda *args, **kwargs: gguf_repo,
    )
    monkeypatch.setattr(
        config_module,
        "is_gguf",
        lambda model: model == gguf_file,
    )
    monkeypatch.setattr(
        config_module,
        "check_gguf_file",
        lambda model: model == gguf_file,
    )
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        lambda model, config_name, revision=None: (
            Path(model) == parent_repo and config_name == config_module.HF_CONFIG_NAME
        ),
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

    def fake_patch_mtp_config(model, config):
        config.model_type = "gemma4_assistant"
        return config

    monkeypatch.setattr(
        config_module,
        "maybe_patch_mtp_config_from_gguf",
        fake_patch_mtp_config,
    )

    config = get_config(
        gguf_file,
        trust_remote_code=False,
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )

    assert calls == [(parent_repo, False, None, {})]
    assert config.model_type == "gemma4_mtp"
    assert config.n_predict == 1
    assert config.architectures == ["Gemma4MTPModel"]
