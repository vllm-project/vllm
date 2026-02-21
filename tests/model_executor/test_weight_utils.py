# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import huggingface_hub.constants
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    enable_hf_transfer,
)


def test_hf_transfer_auto_activation():
    if hasattr(huggingface_hub.constants, "HF_XET_HIGH_PERFORMANCE"):
        # hub>=1.x path: xet acceleration.
        original_value = huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE
        try:
            huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE = False
            enable_hf_transfer()
            assert huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE is True
        finally:
            huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE = original_value
        return

    if not hasattr(huggingface_hub.constants, "HF_HUB_ENABLE_HF_TRANSFER"):
        pytest.skip("No known transfer acceleration flag on this hub version")

    # hub<1.x path: hf_transfer acceleration (if installed).
    original_value = huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER
    try:
        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False
        enable_hf_transfer()
        try:
            import hf_transfer  # type: ignore # noqa

            hf_transfer_active = True
        except ImportError:
            hf_transfer_active = False
        assert hf_transfer_active == huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER
    finally:
        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = original_value


def test_download_weights_from_hf():
    original_offline = huggingface_hub.constants.HF_HUB_OFFLINE
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
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
        finally:
            huggingface_hub.constants.HF_HUB_OFFLINE = original_offline


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()
