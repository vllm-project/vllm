# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A missing config attribute should point at the checkpoint's config.json."""

import copy

import pytest

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

CONFIGS = [DeepseekV4Config, DeepseekV41Config]
# Fields the model packages read straight off the config while building
# layers. The set differs by architecture, platform and enabled feature, so
# the code deliberately does not curate one; these are here as representative
# examples of the failure being explained.
EXAMPLE_FIELDS = ["hc_mult", "hc_eps", "index_topk", "index_n_heads", "o_groups"]


@pytest.mark.parametrize("config_cls", CONFIGS)
@pytest.mark.parametrize("field", EXAMPLE_FIELDS)
def test_a_missing_field_names_itself_and_the_file(config_cls, field):
    """Otherwise this surfaces as a bare AttributeError from inside a layer."""
    with pytest.raises(AttributeError) as excinfo:
        getattr(config_cls(), field)

    message = str(excinfo.value)
    assert f"'{field}'" in message, message
    assert "config.json" in message, message
    assert config_cls.model_type in message, message


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_probing_is_unaffected(config_cls):
    """Hasattr and getattr-with-default decide fallbacks all over the stack."""
    config = config_cls()

    assert hasattr(config, "index_topk") is False
    assert getattr(config, "index_topk", "fallback") == "fallback"


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_a_config_that_has_the_field_is_untouched(config_cls):
    """The hook only fires on lookup failure, so normal use is unchanged."""
    config = config_cls(index_topk=512, hc_mult=4)

    assert config.index_topk == 512
    assert hasattr(config, "hc_mult")
    assert copy.deepcopy(config).to_dict()["index_topk"] == 512
