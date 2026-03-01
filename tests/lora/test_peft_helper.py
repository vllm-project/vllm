# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import shutil
import time
from unittest import mock

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
        "only supports modules_to_save being None",
    ),
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


@pytest.mark.skip_global_cleanup
def test_load_lora_config_retry(tmp_path):
    """_load_lora_config retries transient errors and respects timeout."""
    # Create a local adapter_config.json instead of downloading from HF Hub
    config_data = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }
    config_dir = tmp_path / "adapter"
    config_dir.mkdir()
    src = str(config_dir / "adapter_config.json")
    with open(src, "w") as f:
        json.dump(config_data, f)

    # --- immediate success (file already exists) ---
    result = PEFTHelper._load_lora_config(src, timeout=1.0)
    assert result["r"] == config_data["r"]
    assert result["lora_alpha"] == config_data["lora_alpha"]

    # --- timeout on missing file ---
    missing = str(tmp_path / "no_such_dir" / "adapter_config.json")
    t0 = time.monotonic()
    with pytest.raises(FileNotFoundError):
        PEFTHelper._load_lora_config(missing, timeout=0.2)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.2

    # --- timeout on truncated JSON ---
    trunc_dir = tmp_path / "truncated"
    trunc_dir.mkdir()
    trunc_path = str(trunc_dir / "adapter_config.json")
    with open(trunc_path, "w") as f:
        f.write('{"r": 8, "lora_al')
    t0 = time.monotonic()
    with pytest.raises(json.JSONDecodeError):
        PEFTHelper._load_lora_config(trunc_path, timeout=0.2)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.2

    # --- retry succeeds when transient error clears ---
    real_open = open
    attempt = {"count": 0}

    def flaky_open(path, *args, **kwargs):
        if str(path) == src and attempt["count"] < 3:
            attempt["count"] += 1
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=flaky_open):
        result = PEFTHelper._load_lora_config(src, timeout=5.0)
    assert result["r"] == config_data["r"]
    assert attempt["count"] == 3
