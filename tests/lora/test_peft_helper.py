# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import shutil

import pytest

from vllm.config import LoRAConfig
from vllm.lora.peft_helper import PEFTHelper

ERROR_CASES = [
    (
        "test_rank",
        {
            "r": 1024
        },
        "is greater than max_lora_rank",
    ),
    (
        "test_bias",
        {
            "bias": "all"
        },
        "Adapter bias cannot be used without bias_enabled",
    ),
    ("test_dora", {
        "use_dora": True
    }, "does not yet support DoRA"),
    (
        "test_modules_to_save",
        {
            "modules_to_save": ["lm_head"]
        },
        "only supports modules_to_save being None",
    ),
]


def test_peft_helper_pass(long_context_lora_files_16k_1, tmp_path):
    peft_helper = PEFTHelper.from_local_dir(long_context_lora_files_16k_1,
                                            max_position_embeddings=4096)
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    peft_helper.validate_legal(lora_config)
    assert peft_helper.r == 8
    assert peft_helper.lora_alpha == 16
    assert peft_helper.target_modules == [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    assert peft_helper.context_length == 16384
    assert peft_helper.vllm_max_position_embeddings == 4096
    assert peft_helper.vllm_long_context_scaling_factor == float(
        math.ceil(peft_helper.context_length /
                  peft_helper.vllm_max_position_embeddings))
    # test RSLoRA
    rslora_config = dict(use_rslora=True)
    test_dir = tmp_path / "test_rslora"
    shutil.copytree(long_context_lora_files_16k_1, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(rslora_config)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)

    peft_helper = PEFTHelper.from_local_dir(test_dir,
                                            max_position_embeddings=4096)
    peft_helper.validate_legal(lora_config)
    scaling = peft_helper.lora_alpha / math.sqrt(peft_helper.r)
    assert abs(peft_helper.vllm_lora_scaling_factor - scaling) < 1e-3


@pytest.mark.parametrize("test_name,config_change,expected_error", ERROR_CASES)
def test_peft_helper_error(
    sql_lora_files,
    tmp_path,
    test_name: str,
    config_change: dict,
    expected_error: str,
):
    test_dir = tmp_path / test_name
    shutil.copytree(sql_lora_files, test_dir)

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
            test_dir, max_position_embeddings=4096).validate_legal(lora_config)
