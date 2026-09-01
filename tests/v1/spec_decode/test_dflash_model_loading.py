# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils


@pytest.mark.parametrize("load_fails", [False, True])
def test_dflash_draft_dtype_keeps_shared_cache_config(monkeypatch, load_fails):
    cache_config = SimpleNamespace(cache_dtype="auto", kv_cache_layout=None)
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(),
        cache_config=cache_config,
        speculative_config=SimpleNamespace(
            attention_backend=None,
            draft_model_config=draft_model_config,
            kv_cache_dtype="fp8",
        ),
    )

    captured = SimpleNamespace(cache_config=None, cache_dtype=None)

    class DraftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()

    def get_model(*, vllm_config, model_config):
        assert model_config is draft_model_config
        captured.cache_config = vllm_config.cache_config
        captured.cache_dtype = vllm_config.cache_config.cache_dtype
        if load_fails:
            raise RuntimeError("model load failed")
        return DraftModel()

    def replace(config, **updates):
        values = vars(config) | updates
        return SimpleNamespace(**values)

    monkeypatch.setattr(dflash_utils, "replace", replace)
    monkeypatch.setattr(dflash_utils, "get_model", get_model)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _hf_config: False,
    )
    monkeypatch.setattr(
        dflash_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(dflash_utils, "get_target_lm_head", lambda *_args: None)

    if load_fails:
        with pytest.raises(RuntimeError, match="model load failed"):
            dflash_utils.load_dflash_model(  # type: ignore[arg-type]
                nn.Module(), vllm_config
            )
    else:
        dflash_utils.load_dflash_model(  # type: ignore[arg-type]
            nn.Module(), vllm_config
        )

    assert captured.cache_config is cache_config
    assert captured.cache_dtype == "fp8"
    assert cache_config.cache_dtype == "auto"
    cache_config.kv_cache_layout = "NHD"
    assert captured.cache_config.kv_cache_layout == "NHD"
