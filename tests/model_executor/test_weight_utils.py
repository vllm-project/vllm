# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile

import huggingface_hub.constants
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    enable_hf_transfer,
    filter_duplicate_safetensors_files,
)


def test_hf_transfer_auto_activation():
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
        # in case it is already set, we can't test the auto activation
        pytest.skip("HF_HUB_ENABLE_HF_TRANSFER is set, can't test auto activation")
    enable_hf_transfer()
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa

        HF_TRANSFER_ACTIVE = True
    except ImportError:
        HF_TRANSFER_ACTIVE = False
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER == HF_TRANSFER_ACTIVE


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


def test_filter_duplicate_safetensors_files_with_subfolder(tmp_path):
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    kept_file = llm_dir / "model-00001-of-00002.safetensors"
    kept_file.write_bytes(b"0")
    dropped_file = tmp_path / "other.safetensors"
    dropped_file.write_bytes(b"0")

    index_path = llm_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors"}})
    )

    filtered = filter_duplicate_safetensors_files(
        [str(kept_file), str(dropped_file)],
        str(tmp_path),
        "llm/model.safetensors.index.json",
    )

    assert filtered == [str(kept_file)]


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()
