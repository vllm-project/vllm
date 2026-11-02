# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculators-format config extraction for DSpine drafts.

DSpine checkpoints (e.g. tencent/Qwen3-8B-DSpine-b16) carry a code bridge
under ``dflash_config.selected_prefix_*`` keys plus an ``exact_sender``
last-write refinement head. The ``dspine`` speculators updater maps those
fields into ``dflash_config``, pins causal draft attention, and is served by
the ``dflash`` runtime method.
"""

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dspine_config() -> dict:
    return {
        "speculators_model_type": "dspine",
        "architectures": ["DSpineDraftModel"],
        "mask_token_id": 151669,
        "block_size": 16,
        "draft_vocab_size": 151936,
        "target_hidden_size": 4096,
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "selected_prefix_code_bridge": {
            "code_size": 1024,
            "rank": 512,
            "eps": 1e-6,
        },
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_last_write_refine": {
            "kind": "exact_sender",
            "head_topk": 16,
            "version": 1,
        },
        "code_bridge_second_pass": {
            "mode": "serial_table",
            "first_exact": 4,
            "head_topk": 16,
        },
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_hidden_layers": 5,
        },
        "speculators_config": {
            "proposal_methods": [{"speculative_tokens": 15}],
            "verifier": {"name_or_path": "Qwen/Qwen3-8B"},
        },
    }


def test_dspine_updater_sets_causal_serial_bridge_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspine_config()
    )

    assert config["architectures"] == ["DSpineDraftModel"]
    assert config["is_causal"] is True
    assert config["block_size"] == 16

    dflash_config = config["dflash_config"]
    assert dflash_config["causal"] is True
    assert dflash_config["draft_block_causal"] is True
    assert dflash_config["selected_prefix_terminal"] is False
    assert dflash_config["selected_prefix_feature_write_mode"] == "independent"
    assert dflash_config["selected_prefix_feature_writes"] is True
    assert dflash_config["selected_prefix_feature_layers"] == [1, 2, 3, 4, 5]
    assert dflash_config["selected_prefix_code_bridge"] == {
        "code_size": 1024,
        "rank": 512,
        "eps": 1e-6,
    }
    refine = dflash_config["selected_prefix_last_write_refine"]
    assert refine["kind"] == "exact_sender"
    assert refine["head_topk"] == 16
    assert config["code_bridge_second_pass"]["mode"] == "serial_table"
    assert config["code_bridge_second_pass"]["head_topk"] == 16
    # DFlash plumbing shared by every speculator on the dflash runtime.
    assert dflash_config["mask_token_id"] == 151669
    assert dflash_config["target_layer_ids"] == [1, 9, 17, 25, 33]


def test_dspine_updater_defaults_bridge_flags_and_feature_layers() -> None:
    """Minimal speculators dict: bridge/refine defaults fill dflash_config."""
    config_dict = _make_dspine_config()
    del config_dict["selected_prefix_code_bridge"]
    del config_dict["selected_prefix_feature_layers"]
    del config_dict["selected_prefix_last_write_refine"]
    del config_dict["code_bridge_second_pass"]
    config_dict["dspine_config"] = {"transfer_rank": 1024, "message_size": 512}
    config_dict["head_topk"] = 16

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(config_dict)

    dflash_config = config["dflash_config"]
    assert dflash_config["selected_prefix_feature_layers"] == [1, 2, 3, 4, 5]
    assert dflash_config["selected_prefix_code_bridge"] == {
        "code_size": 1024,
        "rank": 512,
        "eps": 1e-6,
    }
    assert dflash_config["selected_prefix_last_write_refine"]["head_topk"] == 16
    assert dflash_config["selected_prefix_terminal"] is False
    assert "code_bridge_second_pass" not in config


def test_dspine_extracts_dflash_vllm_speculative_config() -> None:
    """Method 'dspine' is served by the dflash runtime like 'dflash2'."""
    spec = SpeculatorsConfig.extract_vllm_speculative_config(_make_dspine_config())

    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 15


def test_dspine_config_preserves_architecture() -> None:
    """EAGLEConfig(method='dflash') must not rename DSpineDraftModel."""
    from transformers import Qwen3Config

    from vllm.transformers_utils.configs.eagle import EAGLEConfig

    config = Qwen3Config(architectures=["DSpineDraftModel"])
    wrapped = EAGLEConfig(config, method="dflash")
    assert wrapped.architectures == ["DSpineDraftModel"]
