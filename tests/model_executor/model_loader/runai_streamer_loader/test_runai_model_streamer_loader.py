# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

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


@pytest.skip("Cannot access model on GCS")
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
