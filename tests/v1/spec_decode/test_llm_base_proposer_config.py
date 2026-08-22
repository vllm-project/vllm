# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import VllmConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


@pytest.mark.cpu_test
def test_draft_vllm_config_uses_draft_model_load_and_quant_configs():
    target_model = SimpleNamespace(name="target")
    draft_model = SimpleNamespace(name="draft")
    target_load = SimpleNamespace(name="target-load")
    draft_load = SimpleNamespace(name="draft-load")
    draft_quant = SimpleNamespace(name="draft-quant")

    base = SimpleNamespace(
        model_config=target_model,
        load_config=target_load,
        quant_config=SimpleNamespace(name="target-quant"),
        kernel_config=SimpleNamespace(moe_backend=None),
        attention_config=SimpleNamespace(backend=None),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    speculative_config = SimpleNamespace(
        draft_load_config=draft_load,
        moe_backend=None,
        attention_backend=None,
        kv_cache_dtype=None,
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.vllm_config = base
    proposer.speculative_config = speculative_config
    proposer.draft_model_config = draft_model

    def fake_replace(instance, **updates):
        values = vars(instance).copy()
        values.update(updates)
        return SimpleNamespace(**values)

    with (
        patch(
            "vllm.v1.spec_decode.llm_base_proposer.replace",
            side_effect=fake_replace,
        ),
        patch.object(
            VllmConfig,
            "get_quantization_config",
            return_value=draft_quant,
        ) as get_quant_config,
    ):
        result = proposer._create_draft_vllm_config()

    get_quant_config.assert_called_once_with(draft_model, draft_load)
    assert result.model_config is draft_model
    assert result.load_config is draft_load
    assert result.quant_config is draft_quant
