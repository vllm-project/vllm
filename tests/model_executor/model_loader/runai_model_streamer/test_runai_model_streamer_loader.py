# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import snapshot_download
from runai_model_streamer.safetensors_streamer.streamer_mock import StreamerPatcher

from vllm import SamplingParams
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader

load_format = "runai_streamer"
test_model = "openai-community/gpt2"
# TODO(amacaskill): Replace with a GKE owned GCS bucket.
test_gcs_model = "gs://vertex-model-garden-public-us/codegemma/codegemma-2b/"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def get_runai_model_loader():
    load_config = LoadConfig(load_format=load_format)
    return get_model_loader(load_config)


def test_get_model_loader_with_runai_flag():
    model_loader = get_runai_model_loader()
    assert model_loader.__class__.__name__ == "RunaiModelStreamerLoader"


def test_runai_model_loader_download_files(vllm_runner):
    with vllm_runner(test_model, load_format=load_format) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs


def test_runai_model_loader_download_files_gcs(
    vllm_runner, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS", "true")
    monkeypatch.setenv(
        "CLOUD_STORAGE_EMULATOR_ENDPOINT", "https://storage.googleapis.com"
    )
    with vllm_runner(test_gcs_model, load_format=load_format) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs


STREAMER_MODULE_PATH = "runai_model_streamer"
VLLM_MODEL_LOADER_MODULE = "vllm.transformers_utils.runai_utils"


@patch(f"{VLLM_MODEL_LOADER_MODULE}.runai_pull_files")
@patch(f"{VLLM_MODEL_LOADER_MODULE}.runai_list_safetensors")
@patch(f"{STREAMER_MODULE_PATH}.SafetensorsStreamer")
def test_runai_model_loader_download_files_s3_mocked_with_patch(
    mock_streamer_class,
    mock_list_safetensors,
    mock_pull_files,
    vllm_runner,
    tmp_path: Path,
):
    test_mock_s3_model = "s3://my-mock-bucket/gpt2/"

    # Download model from HF
    mock_model_dir = f"{tmp_path}/gpt2"
    snapshot_download(repo_id=test_model, local_dir=mock_model_dir)

    # Initialize the Patcher to replace the S3 path with the local path
    local_bucket_root = str(tmp_path)
    patcher = StreamerPatcher(local_bucket_root)

    mock_list_safetensors.side_effect = patcher.shim_list_safetensors
    mock_pull_files.side_effect = patcher.shim_pull_files
    mock_streamer_class.side_effect = patcher.create_mock_streamer

    with vllm_runner(test_mock_s3_model, load_format=load_format) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs
