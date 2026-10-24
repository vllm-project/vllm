# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import Qwen3Config

from vllm.model_executor.models.qwen3_dflash import _dflash_layer_causal
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup

SLIDING = "sliding_attention"
FULL = "full_attention"


def _make_dflash_config(layer_types: list[str], speculators_model_type: str) -> dict:
    return {
        "speculators_model_type": speculators_model_type,
        "architectures": ["DFlashDraftModel"],
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [2, 5],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
            "sliding_window": 8,
        },
    }


def _served_causality(config_dict: dict) -> list[bool]:
    config = Qwen3Config(
        **SpeculatorsConfig.extract_transformers_pre_trained_config(config_dict)
    )
    return [_dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)]


@pytest.mark.parametrize(
    ("speculators_model_type", "layer_types", "sliding_window_non_causal", "expected"),
    [
        ("dflash", [SLIDING] * 3, False, [True] * 3),
        ("dflash", [SLIDING] * 3, True, [False] * 3),
        # Full-attention layers are trained bidirectionally whatever the flag
        ("dflash", [FULL] * 3, False, [False] * 3),
        ("dflash", [FULL, SLIDING, SLIDING], False, [False, True, True]),
        ("dflash2", [FULL, SLIDING, SLIDING], False, [False, True, True]),
        # Absent flag keeps the legacy bidirectional default
        ("dflash", [SLIDING] * 3, None, [False] * 3),
    ],
)
def test_dflash_served_causality_matches_training(
    speculators_model_type, layer_types, sliding_window_non_causal, expected
) -> None:
    config = _make_dflash_config(layer_types, speculators_model_type)
    if sliding_window_non_causal is not None:
        config["sliding_window_non_causal"] = sliding_window_non_causal
    assert _served_causality(config) == expected
