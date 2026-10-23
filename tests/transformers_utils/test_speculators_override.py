# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``maybe_override_with_speculators`` config merging.

When a model is in speculators format, the function previously dropped any
user-provided ``--speculative-config`` overrides (e.g. ``attention_backend``
added in https://github.com/vllm-project/vllm/pull/39930), silently replacing
the dict with a freshly-extracted one. These tests pin the merge behavior so
runtime fields from the user are preserved while ``model`` stays auto-detected
from the speculators format.
"""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.transformers_utils.config import maybe_override_with_speculators

_SPECULATORS_CONFIG_DICT: dict[str, Any] = {
    "architectures": ["DFlashDraftModel"],
    "speculators_config": {
        "algorithm": "dflash",
        "default_proposal_method": "greedy",
        "proposal_methods": [
            {
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 8,
                "verifier_accept_k": 1,
            }
        ],
        "verifier": {
            "architectures": [],
            "name_or_path": "fake-org/fake-verifier",
        },
    },
    "speculators_model_type": "dflash",
    "transformer_layer_config": {},
}


def _patched_get_config_dict(config_dict):
    """Return a patch context that bypasses HF Hub for get_config_dict."""
    return patch(
        "vllm.transformers_utils.config.PreTrainedConfig.get_config_dict",
        return_value=(config_dict, {}),
    )


def test_speculators_extracts_baseline_config():
    """Sanity: extracted dict has method/num_speculative_tokens/model fields."""
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        model, tok, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=None,
        )

    assert model == "fake-org/fake-verifier"
    assert tok == "fake-org/fake-verifier"
    assert spec is not None
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 8
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_preserves_user_attention_backend_override():
    """User-supplied attention_backend must survive the speculators path.

    Regression for the case where vllm_speculative_config was silently
    discarded for speculators-format models.
    """
    user_overrides = {"attention_backend": "FLASH_ATTN"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["attention_backend"] == "FLASH_ATTN"
    # Auto-detected fields still present
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 8
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_user_cannot_override_model_field():
    """``model`` is dictated by the speculators format and must stay locked.

    Even if a user passes a different ``model`` in their override dict, the
    final config must use the speculators model path (so the drafter actually
    loads from the right weights).
    """
    user_overrides = {"model": "fake-org/wrong-model"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_user_cannot_override_method_field():
    user_overrides = {"method": "ngram"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["method"] == "dflash"


def test_speculators_user_can_override_num_speculative_tokens():
    """Runtime knobs (num_speculative_tokens, moe_backend, etc.) take user value."""
    user_overrides = {
        "num_speculative_tokens": 4,
        "moe_backend": "fake-backend",
    }
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["num_speculative_tokens"] == 4
    assert spec["moe_backend"] == "fake-backend"


def test_non_speculators_model_passes_user_config_through_unchanged():
    """When the model is NOT a speculators model, user config is returned as-is."""
    plain_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}
    user_overrides = {"method": "ngram", "num_speculative_tokens": 3}
    with _patched_get_config_dict(plain_config):
        model, tok, spec = maybe_override_with_speculators(
            model="fake-org/plain-llama",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert model == "fake-org/plain-llama"
    assert tok is None
    assert spec == user_overrides


@pytest.mark.parametrize(
    "verifier",
    [
        "/abs/path/to/other_tenant_private",
        "../private-model",
        "./private-model",
        "file:///private-model",
        "C:\\private-model",
        "",
        None,
        [],
        {},
        1,
    ],
)
def test_speculators_rejects_invalid_verifier_references(verifier):
    config = deepcopy(_SPECULATORS_CONFIG_DICT)
    config["speculators_config"]["verifier"]["name_or_path"] = verifier
    with (
        _patched_get_config_dict(config),
        pytest.raises(ValueError, match="Speculators verifier must"),
    ):
        maybe_override_with_speculators("fake-org/draft", None, False)


@pytest.mark.parametrize("verifier", ["private-model", "fake-org/fake-verifier"])
@pytest.mark.parametrize("symlink", [False, True])
def test_speculators_rejects_local_verifiers_with_valid_repo_names(
    tmp_path, monkeypatch, verifier, symlink
):
    monkeypatch.chdir(tmp_path)
    local_verifier = tmp_path / verifier
    local_verifier.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        local_verifier.symlink_to(tmp_path / "private-target", target_is_directory=True)
    else:
        local_verifier.mkdir()

    config = deepcopy(_SPECULATORS_CONFIG_DICT)
    config["speculators_config"]["verifier"]["name_or_path"] = verifier
    with (
        _patched_get_config_dict(config),
        pytest.raises(ValueError, match="Speculators verifier must"),
    ):
        maybe_override_with_speculators("fake-org/draft", None, False)


def test_non_speculators_local_model_remains_operator_selected(tmp_path):
    plain_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64}
    with _patched_get_config_dict(plain_config):
        model, tokenizer, _ = maybe_override_with_speculators(
            str(tmp_path),
            str(tmp_path),
            False,
            vllm_speculative_config={"model": "fake-org/draft", "method": "eagle3"},
        )
    assert model == tokenizer == str(tmp_path)


@pytest.mark.parametrize("engine_args_cls", [EngineArgs, AsyncEngineArgs])
def test_speculators_rejects_local_verifier_before_model_loading(
    tmp_path, engine_args_cls
):
    private_model = tmp_path / "private-model"
    private_model.mkdir()
    config = deepcopy(_SPECULATORS_CONFIG_DICT)
    config["speculators_config"]["verifier"]["name_or_path"] = str(private_model)
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Speculators verifier must"):
        engine_args_cls(model=str(draft)).create_engine_config()


def test_local_file_model_skips_speculators_detection(tmp_path):
    """A single-file checkpoint (e.g. a .gguf) is not parsed as a config."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    assert maybe_override_with_speculators(
        model=str(gguf), tokenizer="org/tok", trust_remote_code=False
    ) == (str(gguf), "org/tok", None)
