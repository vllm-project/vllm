# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AutoRound/INC per-expert bit-width resolution for MuseSpark (CPU only)."""

from types import SimpleNamespace

import pytest

from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    _hf_experts_prefix,
    _inc_experts_bits,
    _is_per_expert_weight,
)

PREFIX = "model.layers.{}.feed_forward.experts"
HF = "model.language_model.layers.{}.mlp.experts"


def _inc(extra):
    return SimpleNamespace(weight_bits=4, extra_config=extra)


def test_hf_experts_prefix():
    assert _hf_experts_prefix(PREFIX.format(7)) == HF.format(7)
    assert _hf_experts_prefix("other.prefix") == "other.prefix"


def test_inc_bits_default_and_override():
    extra = {
        f"{HF.format(0)}.{e}.{p}": {"bits": 16}
        for e in range(2)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    assert _inc_experts_bits(_inc(extra), PREFIX.format(0)) == 16
    assert _inc_experts_bits(_inc(extra), PREFIX.format(5)) == 4
    assert _inc_experts_bits(_inc(None), PREFIX.format(5)) == 4


def test_inc_bits_mixed_raises():
    extra = {
        f"{HF.format(3)}.0.gate_proj": {"bits": 16},
        f"{HF.format(3)}.1.gate_proj": {"bits": 8},
    }
    with pytest.raises(ValueError, match="mix bit widths"):
        _inc_experts_bits(_inc(extra), PREFIX.format(3))


@pytest.mark.parametrize(
    "name,expected",
    [("12.gate_proj.qweight", True), ("w13_weight", False), ("w2_weight_scale", False)],
)
def test_is_per_expert_weight(name, expected):
    assert _is_per_expert_weight(name) is expected
