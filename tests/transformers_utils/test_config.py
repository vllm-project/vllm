# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_speculators_draft_exposes_num_lookahead_tokens():
    # A speculators-format draft declares its proposal count under
    # speculators_config.proposal_methods. SpeculativeConfig defaults
    # num_speculative_tokens from getattr(draft_hf_config, "num_lookahead_tokens"),
    # so the constructed draft config must expose the proposal count there.
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
    config = SpeculatorsConfig(
        **SpeculatorsConfig.extract_transformers_pre_trained_config(config_dict)
    )
    assert config.num_lookahead_tokens == 8
