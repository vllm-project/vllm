# SPDX-License-Identifier: Apache-2.0
"""Tests for configurable Fp8Config.ignored_layers_match_mode."""
import pytest

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)


def _cfg(**kw):
    base = dict(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=["mtp", "layers.0.attention_conv.kernel_projection"],
    )
    base.update(kw)
    return Fp8Config(**base)


def test_default_match_mode_is_exact():
    assert _cfg().ignored_layers_match_mode == "exact"


def test_from_config_reads_match_mode():
    cfg = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["mtp"],
            "ignored_layers_match_mode": "suffix",
        }
    )
    assert cfg.ignored_layers_match_mode == "suffix"
    assert cfg.ignored_layers == ["mtp"]


def test_from_config_defaults_exact():
    cfg = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["mtp"],
        }
    )
    assert cfg.ignored_layers_match_mode == "exact"


def test_invalid_match_mode_raises():
    with pytest.raises(ValueError):
        _cfg(ignored_layers_match_mode="bogus")


@pytest.mark.parametrize(
    "prefix,entry,mode,expected",
    [
        # exact misses a suffix entry on a nested runtime prefix
        ("model.layers.0.attention_conv.kernel_projection",
         "layers.0.attention_conv.kernel_projection", "exact", False),
        # suffix hits
        ("model.layers.0.attention_conv.kernel_projection",
         "layers.0.attention_conv.kernel_projection", "suffix", True),
        # MTP head: bare "mtp" entry vs "mtp.fc" runtime prefix.
        # suffix requires a component boundary ".mtp", so it does NOT match;
        # substring is needed for bare module-name entries.
        ("mtp.fc", "mtp", "exact", False),
        ("mtp.fc", "mtp", "suffix", False),
        ("mtp.fc", "mtp", "substring", True),
        # substring also works for abbreviated names
        ("model.layers.0.self_attn.q_proj", "q_proj", "substring", True),
    ],
)
def test_is_layer_skipped_modes(prefix, entry, mode, expected):
    assert is_layer_skipped(
        prefix=prefix,
        ignored_layers=[entry],
        fused_mapping={},
        match_mode=mode,
    ) is expected