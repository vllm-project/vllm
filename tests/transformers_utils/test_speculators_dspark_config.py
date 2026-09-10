# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _dspark_config(architecture: str, *, sample_from_anchor: bool) -> dict:
    return {
        "speculators_model_type": "dspark",
        "architectures": [architecture],
        "sample_from_anchor": sample_from_anchor,
        "aux_hidden_state_layer_ids": [2, 5],
        "mask_token_id": 31,
        "transformer_layer_config": {"model_type": "qwen3"},
    }


def test_dspark_preserves_qwen3_vl_architecture_and_anchor_semantics() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _dspark_config("Qwen3VLDSparkModel", sample_from_anchor=True)
    )

    assert config["architectures"] == ["Qwen3VLDSparkModel"]
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False


def test_dspark_keeps_generic_qwen3_architecture() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _dspark_config("DSparkSpeculator", sample_from_anchor=False)
    )

    assert config["architectures"] == ["Qwen3DSparkModel"]
    assert config["dspark_bonus_anchor"] is True
