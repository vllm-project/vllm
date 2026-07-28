# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for pure config-helper utilities in transformers_utils.config.

These helpers are used by model loading / RoPE setup paths but previously had
no direct coverage. Tests use lightweight stand-ins so they run on CPU without
downloading Hugging Face models.
"""

from __future__ import annotations

from typing import Any

import pytest
from transformers.configuration_utils import ALLOWED_LAYER_TYPES

from vllm.transformers_utils.config import (
    is_rope_parameters_nested,
    thinker_uses_mrope,
    uses_xdrope_dim,
)


class FakeConfig:
    """Minimal stand-in for transformers.PretrainedConfig."""

    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    def get_text_config(self) -> FakeConfig:
        return getattr(self, "text_config", self)


@pytest.mark.parametrize(
    ("rope_parameters", "expected"),
    [
        ({}, False),
        ({"full_attention": {"type": "default"}}, True),
        (
            {
                "full_attention": {"type": "default"},
                "sliding_attention": {"type": "default"},
            },
            True,
        ),
        ({"not_a_layer_type": {}}, False),
        ({"full_attention": {}, "not_a_layer_type": {}}, False),
        # Sanity-check against the live transformers allow-list so a
        # future rename of a real layer type does not silently flip this.
        ({next(iter(ALLOWED_LAYER_TYPES)): {}}, True),
    ],
)
def test_is_rope_parameters_nested(rope_parameters, expected):
    assert is_rope_parameters_nested(rope_parameters) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (FakeConfig(), False),
        (FakeConfig(thinker_config=FakeConfig()), False),
        (
            FakeConfig(
                thinker_config=FakeConfig(
                    text_config=FakeConfig(rope_parameters={"type": "default"})
                )
            ),
            False,
        ),
        (
            FakeConfig(
                thinker_config=FakeConfig(
                    text_config=FakeConfig(
                        rope_parameters={"mrope_section": [16, 24, 24]}
                    )
                )
            ),
            True,
        ),
        (
            FakeConfig(
                thinker_config=FakeConfig(text_config=FakeConfig(rope_parameters=None))
            ),
            False,
        ),
    ],
)
def test_thinker_uses_mrope(config, expected):
    assert thinker_uses_mrope(config) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (FakeConfig(), 0),
        (FakeConfig(xdrope_section=[16, 24, 24]), 3),
        (FakeConfig(xdrope_section="not-a-list"), 0),
        (FakeConfig(rope_scaling=None), 0),
        (FakeConfig(rope_scaling={"type": "yarn"}), 0),
        (FakeConfig(rope_scaling={"xdrope_section": [8, 8]}), 2),
        (FakeConfig(rope_scaling={"xdrope_section": None}), 0),
        (FakeConfig(rope_scaling={"xdrope_section": "bad"}), 0),
        # Top-level xdrope_section wins over rope_scaling.
        (
            FakeConfig(
                xdrope_section=[1, 2, 3, 4],
                rope_scaling={"xdrope_section": [8, 8]},
            ),
            4,
        ),
    ],
)
def test_uses_xdrope_dim(config, expected):
    assert uses_xdrope_dim(config) == expected
