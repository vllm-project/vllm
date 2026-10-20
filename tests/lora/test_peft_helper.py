# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import shutil

import pytest

from vllm.config.lora import LoRAConfig
from vllm.lora.peft_helper import PEFTHelper

ERROR_CASES = [
    (
        "test_rank",
        {"r": 1024},
        "is greater than max_lora_rank",
    ),
    ("test_dora", {"use_dora": True}, "does not yet support DoRA"),
    (
        "test_modules_to_save",
        {"modules_to_save": ["lm_head"]},
        "Unsupported modules_to_save",
    ),
    ("test_rank_zero", {"r": 0}, "must be a positive integer"),
    ("test_rank_negative", {"r": -8}, "must be a positive integer"),
    ("test_lora_bias", {"lora_bias": True}, "does not support LoRA bias"),
    ("test_pissa", {"init_lora_weights": "pissa"}, "init_lora_weights='pissa'"),
    (
        "test_pissa_niter",
        {"init_lora_weights": "pissa_niter_4"},
        "init_lora_weights='pissa_niter_4'",
    ),
    ("test_olora", {"init_lora_weights": "olora"}, "init_lora_weights='olora'"),
    ("test_corda", {"init_lora_weights": "corda"}, "init_lora_weights='corda'"),
    ("test_loftq", {"init_lora_weights": "loftq"}, "init_lora_weights='loftq'"),
    (
        "test_unknown_init",
        {"init_lora_weights": "future_init"},
        "init_lora_weights='future_init'",
    ),
    ("test_alora", {"alora_invocation_tokens": [1, 2]}, "Activated LoRA"),
    ("test_layer_replication", {"layer_replication": [[0, 2]]}, "layer_replication"),
    ("test_bdlora", {"use_bdlora": {"nblocks": 2}}, "BD-LoRA"),
    ("test_qalora", {"use_qalora": True}, "QALoRA"),
]


def test_peft_helper_pass(llama32_lora_files, tmp_path):
    peft_helper = PEFTHelper.from_local_dir(
        llama32_lora_files, max_position_embeddings=4096
    )
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    peft_helper.validate_legal(lora_config)
    assert peft_helper.r == 8
    assert peft_helper.lora_alpha == 32
    target_modules = sorted(peft_helper.target_modules)

    assert target_modules == [
        "down_proj",
        "embed_tokens",
        "gate_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert peft_helper.vllm_max_position_embeddings == 4096

    # test RSLoRA
    rslora_config = dict(use_rslora=True)
    test_dir = tmp_path / "test_rslora"
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(rslora_config)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)

    peft_helper = PEFTHelper.from_local_dir(test_dir, max_position_embeddings=4096)
    peft_helper.validate_legal(lora_config)
    scaling = peft_helper.lora_alpha / math.sqrt(peft_helper.r)
    assert abs(peft_helper.vllm_lora_scaling_factor - scaling) < 1e-3


@pytest.mark.parametrize("test_name,config_change,expected_error", ERROR_CASES)
def test_peft_helper_error(
    llama32_lora_files,
    tmp_path,
    test_name: str,
    config_change: dict,
    expected_error: str,
):
    test_dir = tmp_path / test_name
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(config_change)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    # Test loading the adapter
    with pytest.raises(ValueError, match=expected_error):
        PEFTHelper.from_local_dir(
            test_dir, max_position_embeddings=4096
        ).validate_legal(lora_config)


@pytest.mark.parametrize("bad_rank", [0, -1, -8])
def test_peft_helper_invalid_rank_direct(bad_rank: int):
    """Regression test: constructing a PEFTHelper with a non-positive rank
    must raise a clear ValueError instead of crashing with an unrelated
    ZeroDivisionError (r=0) or silently succeeding with a sign-flipped
    scaling factor that validate_legal() never catches (r<0, since its only
    rank check is the upper bound against max_lora_rank).

    Network-free: constructs PEFTHelper directly rather than going through
    from_local_dir(), which needs an on-disk adapter_config.json.
    """
    with pytest.raises(ValueError, match="must be a positive integer"):
        PEFTHelper(r=bad_rank, lora_alpha=16, target_modules=["q_proj"])


@pytest.mark.parametrize(
    "init_lora_weights",
    [True, False, "gaussian", "eva", "orthogonal", "mica", "lora_ga"],
)
def test_peft_helper_init_lora_weights_supported(init_lora_weights):
    """A saved adapter with these inits loads in PEFT as a plain LoRA."""
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    PEFTHelper.from_dict(
        {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "init_lora_weights": init_lora_weights,
            "alora_invocation_tokens": None,
            "layer_replication": None,
            "use_bdlora": None,
        }
    ).validate_legal(lora_config)
