# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

@pytest.fixture(scope="session")
def zephyr_lora_files():
    """Download zephyr LoRA files once per test session."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id="typeof/zephyr-7b-beta-lora")


@pytest.fixture(scope="session")
def zephyr_lora_added_tokens_files(zephyr_lora_files):
    """Create zephyr LoRA files with added tokens once per test session."""
    import shutil
    from tempfile import TemporaryDirectory

    from transformers import AutoTokenizer

    tmp_dir = TemporaryDirectory()
    tmp_model_dir = f"{tmp_dir.name}/zephyr"
    shutil.copytree(zephyr_lora_files, tmp_model_dir)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # Copy tokenizer to adapter and add some unique tokens
    # 32000, 32001, 32002
    added = tokenizer.add_tokens(["vllm1", "vllm2", "vllm3"],
                                 special_tokens=True)
    assert added == 3
    tokenizer.save_pretrained(tmp_model_dir)
    yield tmp_model_dir
    tmp_dir.cleanup()
