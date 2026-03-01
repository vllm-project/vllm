# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import hashlib
import os
import tempfile

import huggingface_hub.constants

from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.transformers_utils.runai_utils import (
    ObjectStorageModel,
    is_runai_obj_uri,
    list_safetensors,
)


def test_is_runai_obj_uri():
    assert is_runai_obj_uri("gs://some-gcs-bucket/path")
    assert is_runai_obj_uri("s3://some-s3-bucket/path")
    assert not is_runai_obj_uri("nfs://some-nfs-path")


def test_runai_list_safetensors_local():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2",
            allow_patterns=["*.safetensors", "*.json"],
            cache_dir=tmpdir,
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0
        parentdir = [os.path.dirname(safetensor) for safetensor in safetensors][0]
        files = list_safetensors(parentdir)
        assert len(safetensors) == len(files)


def test_runai_pull_files_gcs(monkeypatch):
    monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS", "true")
    # Bypass default project lookup by setting GOOGLE_CLOUD_PROJECT
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    filename = "LT08_L1GT_074061_20130309_20170505_01_T2_MTL.txt"
    gcs_bucket = "gs://gcp-public-data-landsat/LT08/01/074/061/LT08_L1GT_074061_20130309_20170505_01_T2/"
    gcs_url = f"{gcs_bucket}/{filename}"
    model = ObjectStorageModel(gcs_url)
    model.pull_files(gcs_bucket, allow_pattern=[f"*{filename}"])
    # To re-generate / change URLs:
    #   gsutil ls -L gs://<gcs-url> | grep "Hash (md5)" | tr -d ' ' \
    #     | cut -d":" -f2 | base64 -d | xxd -p
    expected_checksum = "f60dea775da1392434275b311b31a431"
    hasher = hashlib.new("md5")
    with open(os.path.join(model.dir, filename), "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    actual_checksum = hasher.hexdigest()
    assert actual_checksum == expected_checksum
