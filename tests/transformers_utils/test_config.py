# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import vllm.transformers_utils.config as config_module
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    maybe_override_with_speculators,
    try_get_generation_config,
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


def test_maybe_override_with_speculators_gguf_quant_modelscope_no_path_replace_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    model = "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_M"
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_get_config_dict(model_arg: Any, **kwargs: Any):
        if isinstance(model_arg, Path):
            raise TypeError(
                "Path.replace() takes 2 positional arguments but 3 were given"
            )
        calls.append((model_arg, kwargs))
        return {}, None

    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")
    monkeypatch.setattr(
        config_module.PretrainedConfig,
        "get_config_dict",
        staticmethod(fake_get_config_dict),
    )

    resolved_model, resolved_tokenizer, speculative_config = (
        maybe_override_with_speculators(
            model=model,
            tokenizer=None,
            trust_remote_code=False,
        )
    )

    assert len(calls) == 1
    model_arg, kwargs = calls[0]
    assert model_arg == (
        "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
    )
    assert "*.gguf" in kwargs["ignore_file_pattern"]
    assert resolved_model == model
    assert resolved_tokenizer is None
    assert speculative_config is None


def test_modelscope_remote_gguf_config_downloads_selected_file(
    monkeypatch: pytest.MonkeyPatch,
):
    model = "hesamation/model-GGUF:Q4_K_M"
    kwargs: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []

    def fake_get_gguf_file_path_from_hf(
        repo_id: str,
        quant_type: str,
        *,
        revision: str | None = None,
    ):
        assert repo_id == "hesamation/model-GGUF"
        assert quant_type == "Q4_K_M"
        assert revision == "master"
        return "model.Q4_K_M.gguf"

    def fake_snapshot_download(**snapshot_kwargs: Any):
        calls.append(snapshot_kwargs)
        return "/cache/model-GGUF"

    modelscope_module = types.ModuleType("modelscope")
    hub_module = types.ModuleType("modelscope.hub")
    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot_module.snapshot_download = fake_snapshot_download
    hub_module.snapshot_download = snapshot_module
    modelscope_module.hub = hub_module
    monkeypatch.setitem(sys.modules, "modelscope", modelscope_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub_module)
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        snapshot_module,
    )
    monkeypatch.setattr(
        config_module,
        "get_gguf_file_path_from_hf",
        fake_get_gguf_file_path_from_hf,
    )

    resolved = config_module._localize_modelscope_remote_gguf_for_config(
        model,
        "master",
        kwargs,
    )

    assert resolved == "/cache/model-GGUF"
    assert kwargs["gguf_file"] == "model.Q4_K_M.gguf"
    assert calls == [
        {
            "model_id": "hesamation/model-GGUF",
            "revision": "master",
            "local_files_only": False,
            "allow_patterns": "model.Q4_K_M.gguf",
        }
    ]
