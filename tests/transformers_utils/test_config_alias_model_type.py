# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading one config alias must not rewrite an earlier config's model_type."""

import json
import pickle
from pathlib import Path

import pytest
from transformers import DeepseekV3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from vllm.transformers_utils.config import get_config


def _write_config(root: Path, model_type: str) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": ["DeepseekV3ForCausalLM"],
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 256,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
            }
        )
    )


@pytest.mark.parametrize(
    ("first_alias", "second_alias"),
    [
        ("kimi_k2", "deepseek_v32"),
        ("deepseek_v32", "kimi_k2"),
        ("kimi_k2", "kimi_k2"),
    ],
)
def test_second_alias_keeps_first_serialized_model_type(
    tmp_path: Path, first_alias: str, second_alias: str
) -> None:
    """A later alias load must not change an earlier config's saved model_type."""
    original_model_type = DeepseekV3Config.model_type
    extra_content = dict(CONFIG_MAPPING._extra_content)
    try:
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        _write_config(first_root, first_alias)
        _write_config(second_root, second_alias)

        first = get_config(first_root, trust_remote_code=False)
        assert isinstance(first, DeepseekV3Config)
        assert first.model_type == first_alias
        assert first.to_dict()["model_type"] == first_alias

        second = get_config(second_root, trust_remote_code=False)
        assert second.to_dict()["model_type"] == second_alias
        assert first.model_type == first_alias
        assert first.to_dict()["model_type"] == first_alias
        assert DeepseekV3Config.model_type == original_model_type

        saved = tmp_path / "saved-first"
        first.save_pretrained(saved)
        saved_model_type = json.loads((saved / "config.json").read_text())["model_type"]
        assert saved_model_type == first_alias
        reloaded = get_config(saved, trust_remote_code=False)
        assert isinstance(reloaded, DeepseekV3Config)
        assert reloaded.to_dict()["model_type"] == first_alias
        assert first.to_dict()["model_type"] == first_alias
        restored = pickle.loads(pickle.dumps(first))
        assert restored.to_dict()["model_type"] == first_alias
    finally:
        DeepseekV3Config.model_type = original_model_type
        CONFIG_MAPPING._extra_content.clear()
        CONFIG_MAPPING._extra_content.update(extra_content)
