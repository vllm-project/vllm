# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for cohere-specific ``OnlineQuantizationConfig.from_config``.

This covers the checkpoint ``config.json`` ``quantization_config`` parsing
path, which is distinct from the upstream CLI/``OnlineQuantizationConfigArgs``
path exercised by ``tests/quantization/test_online.py``.

CPU-safe: the parsing path doesn't touch CUDA; lives under ``tests/cohere/cpu``
so it gets auto-discovered by ``run_cpu_tests`` in ``run_tests.sh``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config.quantization import OnlineQuantScheme  # noqa: E402
from vllm.model_executor.layers.quantization.online.base import (  # noqa: E402
    OnlineQuantizationConfig,
)


@pytest.mark.parametrize(
    "shorthand,expected_scheme",
    [
        ("fp8_per_tensor", OnlineQuantScheme.FP8_PER_TENSOR),
        ("fp8_per_block", OnlineQuantScheme.FP8_PER_BLOCK),
        ("mxfp8", OnlineQuantScheme.MXFP8),
        (
            "int8_per_channel_weight_only",
            OnlineQuantScheme.INT8_PER_CHANNEL_WEIGHT_ONLY,
        ),
    ],
)
def test_shorthand_quant_method_populates_global_scheme(
    shorthand: str, expected_scheme: OnlineQuantScheme
) -> None:
    cfg = OnlineQuantizationConfig.from_config({"quant_method": shorthand})

    assert cfg.args.global_scheme == expected_scheme
    assert cfg.args.linear_scheme_override is None
    assert cfg.args.moe_scheme_override is None
    assert cfg.ignored_layers == []


def test_online_quant_method_with_explicit_overrides() -> None:
    cfg = OnlineQuantizationConfig.from_config(
        {
            "quant_method": "online",
            "linear_scheme_override": "fp8_per_block",
            "moe_scheme_override": "fp8_per_tensor",
        }
    )

    assert cfg.args.global_scheme is None
    assert cfg.args.linear_scheme_override == OnlineQuantScheme.FP8_PER_BLOCK
    assert cfg.args.moe_scheme_override == OnlineQuantScheme.FP8_PER_TENSOR


def test_explicit_global_scheme_not_overwritten_by_shorthand() -> None:
    """If ``global_scheme`` is already set, ``quant_method`` shorthand
    should not clobber it."""
    cfg = OnlineQuantizationConfig.from_config(
        {
            "quant_method": "fp8_per_tensor",
            "global_scheme": "fp8_per_block",
        }
    )

    assert cfg.args.global_scheme == OnlineQuantScheme.FP8_PER_BLOCK


def test_ignore_aliases_are_merged() -> None:
    cfg = OnlineQuantizationConfig.from_config(
        {
            "quant_method": "fp8_per_block",
            "ignore": ["a", "re:.*self_attn.*"],
            "ignored_layers": ["b"],
            "modules_to_not_convert": ["c"],
        }
    )

    assert cfg.ignored_layers == ["a", "re:.*self_attn.*", "b", "c"]


def test_ignore_only_from_alias_when_primary_missing() -> None:
    cfg = OnlineQuantizationConfig.from_config(
        {
            "quant_method": "fp8_per_block",
            "ignored_layers": ["only_alias"],
        }
    )

    assert cfg.ignored_layers == ["only_alias"]


def test_activation_scheme_dynamic_is_accepted() -> None:
    cfg = OnlineQuantizationConfig.from_config(
        {
            "quant_method": "fp8_per_tensor",
            "activation_scheme": "dynamic",
        }
    )

    assert cfg.args.global_scheme == OnlineQuantScheme.FP8_PER_TENSOR


def test_activation_scheme_static_raises() -> None:
    with pytest.raises(ValueError, match="activation_scheme"):
        OnlineQuantizationConfig.from_config(
            {
                "quant_method": "fp8_per_tensor",
                "activation_scheme": "static",
            }
        )


def test_no_scheme_raises() -> None:
    """At least one of global / linear / moe scheme must be set."""
    with pytest.raises(ValueError, match="global_scheme"):
        OnlineQuantizationConfig.from_config({"quant_method": "online"})
