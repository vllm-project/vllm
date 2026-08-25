# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dspark_config(architecture: str) -> dict:
    return {
        "speculators_model_type": "dspark",
        "architectures": [architecture],
        "sample_from_anchor": True,
        "use_aux_hidden_state": True,
        "draft_vocab_size": 32,
        "target_hidden_size": 16,
        "mask_token_id": 31,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "block_size": 3,
        "aux_hidden_state_layer_ids": [2, 5],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }


def test_dspark_updater_preserves_qwen3_vl_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("Qwen3VLDSparkModel")
    )

    assert config["architectures"] == ["Qwen3VLDSparkModel"]
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False
    assert config["use_aux_hidden_state"] is True
    assert config["target_layer_ids"] == [1, 4]
    assert config["dflash_config"] == {
        "causal": False,
        "mask_token_id": 31,
        "target_layer_ids": [1, 4],
        "use_aux_hidden_state": True,
    }


def test_dspark_updater_keeps_legacy_qwen3_checkpoint_loadable() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("DSparkSpeculator")
    )

    assert config["architectures"] == ["Qwen3DSparkModel"]
