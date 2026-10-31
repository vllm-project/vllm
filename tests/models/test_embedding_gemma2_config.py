# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for EmbeddingGemma2 configuration."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import EmbeddingGemma2ModelConfig


def _update(**hf_config_fields) -> SimpleNamespace:
    hf_config = SimpleNamespace(**hf_config_fields)
    EmbeddingGemma2ModelConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=hf_config)
    )
    return hf_config


def test_sets_matryoshka_dimensions_from_model_card() -> None:
    assert _update().matryoshka_dimensions == [128, 256, 512, 768]


@pytest.mark.parametrize(
    "fields",
    [
        {"matryoshka_dimensions": [768, 256]},
        {"is_matryoshka": True},
    ],
)
def test_keeps_declared_matryoshka_config(fields) -> None:
    hf_config = _update(**fields)
    assert getattr(hf_config, "matryoshka_dimensions", None) == fields.get(
        "matryoshka_dimensions"
    )
