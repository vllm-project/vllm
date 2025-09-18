# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import tempfile

import huggingface_hub.constants

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)
from vllm.transformers_utils.runai_utils import (is_runai_obj_uri,
                                                 list_safetensors)


def test_is_runai_obj_uri():
    assert is_runai_obj_uri("gs://some-gcs-bucket/path")
    assert is_runai_obj_uri("s3://some-s3-bucket/path")
    assert not is_runai_obj_uri("nfs://some-nfs-path")


def test_runai_list_safetensors_local():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf("openai-community/gpt2",
                                 allow_patterns=["*.safetensors", "*.json"],
                                 cache_dir=tmpdir)
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0
        parentdir = [
            os.path.dirname(safetensor) for safetensor in safetensors
        ][0]
        files = list_safetensors(parentdir)
        assert len(safetensors) == len(files)


if __name__ == "__main__":
    test_is_runai_obj_uri()
    test_runai_list_safetensors_local()
