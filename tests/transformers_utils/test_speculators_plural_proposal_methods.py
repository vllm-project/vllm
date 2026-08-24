# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection among plural speculators proposal_methods.

The speculators schema declares proposal_methods as a list and names the
active one via default_proposal_method. vLLM drafts with a single method:
the config layer must honor default_proposal_method (falling back to the
first entry for configs that predate the field) and must not silently
discard the other declared methods.
"""

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _config(
    proposal_methods: list[dict], default_proposal_method: str | None = "eagle3"
) -> dict:
    spec_config = {
        "proposal_methods": proposal_methods,
        "verifier": {"name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
    }
    if default_proposal_method is not None:
        spec_config["default_proposal_method"] = default_proposal_method
    return {
        "speculators_model_type": "eagle3",
        "speculators_config": spec_config,
        "transformer_layer_config": {},
    }


def test_single_method_unchanged():
    result = SpeculatorsConfig.extract_vllm_speculative_config(
        _config([{"proposal_type": "eagle3", "speculative_tokens": 5}])
    )
    assert result["num_speculative_tokens"] == 5
    assert result["method"] == "eagle3"


def test_default_proposal_method_selects_non_first_entry():
    # The named default wins even when it is not entry [0].
    result = SpeculatorsConfig.extract_vllm_speculative_config(
        _config(
            [
                {"proposal_type": "suffix", "speculative_tokens": 8},
                {"proposal_type": "eagle3", "speculative_tokens": 5},
            ],
            default_proposal_method="eagle3",
        )
    )
    assert result["num_speculative_tokens"] == 5


def test_missing_default_falls_back_to_first(caplog_vllm):
    # A distinct ignored-method name keeps this emission out of
    # warning_once's dedup cache shared with the other tests.
    result = SpeculatorsConfig.extract_vllm_speculative_config(
        _config(
            [
                {"proposal_type": "eagle3", "speculative_tokens": 5},
                {"proposal_type": "ngram", "speculative_tokens": 8},
            ],
            default_proposal_method=None,
        )
    )
    assert result["num_speculative_tokens"] == 5
    assert "ignoring" in caplog_vllm.text
    assert "ngram" in caplog_vllm.text


def test_unmatched_default_falls_back_to_first(caplog_vllm):
    result = SpeculatorsConfig.extract_vllm_speculative_config(
        _config(
            [{"proposal_type": "eagle3", "speculative_tokens": 5}],
            default_proposal_method="greedy",
        )
    )
    assert result["num_speculative_tokens"] == 5
    assert "matches no entry" in caplog_vllm.text


def test_non_selected_method_without_speculative_tokens_tolerated():
    # speculative_tokens is a field of concrete method types, not the base
    # schema: a non-selected entry without it must not fail engine start.
    result = SpeculatorsConfig.extract_vllm_speculative_config(
        _config(
            [
                {"proposal_type": "eagle3", "speculative_tokens": 5},
                {"proposal_type": "suffix"},
            ],
            default_proposal_method="eagle3",
        )
    )
    assert result["num_speculative_tokens"] == 5


def test_entry_without_proposal_type_rejected():
    with pytest.raises(ValueError, match="proposal_type"):
        SpeculatorsConfig.extract_vllm_speculative_config(
            _config(
                [
                    {"proposal_type": "eagle3", "speculative_tokens": 5},
                    {"speculative_tokens": 8},
                ]
            )
        )


def test_empty_methods_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        SpeculatorsConfig.extract_vllm_speculative_config(_config([]))


def test_selected_method_missing_speculative_tokens_rejected():
    with pytest.raises(ValueError, match="speculative_tokens"):
        SpeculatorsConfig.extract_vllm_speculative_config(
            _config([{"proposal_type": "eagle3"}])
        )
