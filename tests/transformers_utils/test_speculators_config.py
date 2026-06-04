# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for #40382: a speculators-format drafts declared
``speculative_tokens`` must be forwarded onto the draft config (as
``num_lookahead_tokens``) so ``num_speculative_tokens`` can be defaulted when the
draft is loaded via ``--speculative-config '{\"model\": <speculators repo>}'``.
"""
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


def test_speculative_tokens_forwarded_as_num_lookahead():
    config_dict = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {"model_type": "llama", "head_dim": 256},
        "mask_token_id": 4,
        "aux_hidden_state_layer_ids": [1, 2],
        "speculators_config": {
            "proposal_methods": [{"speculative_tokens": 8}],
            "verifier": {"name_or_path": "some/target"},
        },
    }
    pre = SpeculatorsConfig.extract_transformers_pre_trained_config(config_dict)
    assert pre["num_lookahead_tokens"] == 8
