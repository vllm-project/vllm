# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for QuarkConfig.maybe_update_config.

Fetches real HF configs (metadata only, no model weights) to verify
that dynamic_mxfp4_quant is only enabled for DeepSeek-V3-family models.

Run: pytest tests/quantization/test_quark_maybe_update_config.py -v
"""

import pytest
from transformers import AutoConfig

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig


def _make_quark_config() -> QuarkConfig:
    """Create a minimal QuarkConfig for testing."""
    return QuarkConfig(quant_config={}, kv_cache_group=[], pack_method="reorder")


# ---------------------------------------------------------------------------
# Non-deepseek models must not flip dynamic_mxfp4_quant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name",
    ["amd/MiniMax-M2.1-MXFP4"],
)
def test_non_deepseek_model_stays_false(model_name: str):
    """Non-deepseek_v3 models must not enable dynamic_mxfp4_quant."""
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    qcfg = _make_quark_config()

    qcfg.maybe_update_config(model_name, hf_config=hf_config)

    assert qcfg.dynamic_mxfp4_quant is False


# ---------------------------------------------------------------------------
# DeepSeek-V3 family + fp4 must enable dynamic_mxfp4_quant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name",
    ["amd/DeepSeek-R1-MXFP4-ASQ"],
)
def test_deepseek_family_fp4_enables_flag(model_name: str):
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    qcfg = _make_quark_config()

    qcfg.maybe_update_config(model_name, hf_config=hf_config)

    assert qcfg.dynamic_mxfp4_quant is True


# ---------------------------------------------------------------------------
# Missing hf_config → warn and stay False
# ---------------------------------------------------------------------------
def test_missing_hf_config_stays_false():
    qcfg = _make_quark_config()

    qcfg.maybe_update_config("some/model")

    assert qcfg.dynamic_mxfp4_quant is False
