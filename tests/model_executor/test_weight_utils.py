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


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()


def test_filter_duplicate_safetensors_files_missing_shard():
    """Verify that missing shard files referenced in the index raise
    RuntimeError (GitHub issue #34859)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an index file referencing two shard files
        index = {
            "weight_map": {
                "model.layer.weight": "model-00001-of-00002.safetensors",
                "model.head.weight": "model-00002-of-00002.safetensors",
            }
        }
        index_file = "model.safetensors.index.json"
        with open(os.path.join(tmpdir, index_file), "w") as f:
            json.dump(index, f)

        # Only create one shard file — the other is "missing"
        shard1 = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        with open(shard1, "w") as f:
            pass

        with pytest.raises(RuntimeError, match="missing from disk"):
            filter_duplicate_safetensors_files([shard1], tmpdir, index_file)


def test_filter_duplicate_safetensors_files_all_present():
    """Verify that a complete checkpoint passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = {
            "weight_map": {
                "model.layer.weight": "model-00001-of-00002.safetensors",
                "model.head.weight": "model-00002-of-00002.safetensors",
            }
        }
        index_file = "model.safetensors.index.json"
        with open(os.path.join(tmpdir, index_file), "w") as f:
            json.dump(index, f)

        shard1 = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        for path in [shard1, shard2]:
            with open(path, "w") as f:
                pass

        result = filter_duplicate_safetensors_files(
            [shard1, shard2], tmpdir, index_file
        )
        assert set(result) == {shard1, shard2}
