# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config, get_hf_image_processor_config


def test_rwkv7_pth_url_builds_config_from_blinkdl_filename():
    config = get_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
        trust_remote_code=False,
    )

    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.model_type == "rwkv7"
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24
    assert config.head_size == 64
    assert config.vocab_size == 65536
    assert config.max_position_embeddings == 8192


def test_rwkv7_local_pth_builds_config_from_filename(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-2.9b-20260526-ctx8192.pth"
    checkpoint.touch()

    config = get_config(checkpoint, trust_remote_code=False)

    assert config.hidden_size == 2560
    assert config.num_hidden_layers == 32


def test_unknown_rwkv7_pth_filename_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-custom-ctx8192.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="Unsupported RWKV7 raw .pth checkpoint"):
        get_config(checkpoint, trust_remote_code=False)


def test_rwkv7_pth_url_has_no_image_processor_config():
    config = get_hf_image_processor_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
    )

    assert config == {}
