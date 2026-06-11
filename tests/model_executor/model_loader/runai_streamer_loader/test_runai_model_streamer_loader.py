# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

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


def test_runai_model_loader_selects_mistral_consolidated_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    model_loader = get_runai_model_loader()
    model_dir = tmp_path / "mistral"
    model_dir.mkdir()
    weight_files = [
        str(model_dir / "consolidated.safetensors"),
        str(model_dir / "model-00001-of-00002.safetensors"),
        str(model_dir / "model-00002-of-00002.safetensors"),
    ]
    for weight_file in weight_files:
        Path(weight_file).touch()

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.runai_streamer_loader.list_safetensors",
        lambda path: weight_files,
    )

    model_config = SimpleNamespace(
        model=str(model_dir),
        model_weights="",
        revision=None,
        config_format="auto",
    )

    assert model_loader._prepare_weights(str(model_dir), model_config) == [
        str(model_dir / "consolidated.safetensors")
    ]


def test_runai_model_loader_selects_mistral_consolidated_object_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    model_loader = get_runai_model_loader()
    model_dir = tmp_path / "mistral"
    model_dir.mkdir()
    object_uri = "s3://bucket/mistral"
    weight_files = [
        f"{object_uri}/consolidated.safetensors",
        f"{object_uri}/model-00001-of-00002.safetensors",
        f"{object_uri}/model-00002-of-00002.safetensors",
    ]

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.runai_streamer_loader.list_safetensors",
        lambda path: weight_files,
    )

    model_config = SimpleNamespace(
        model=str(model_dir),
        model_weights=object_uri,
        revision=None,
        config_format="hf",
    )

    assert model_loader._prepare_weights(object_uri, model_config) == [
        f"{object_uri}/consolidated.safetensors"
    ]


def test_runai_model_loader_keeps_object_weights_without_mistral_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    model_loader = get_runai_model_loader()
    model_dir = tmp_path / "hf"
    model_dir.mkdir()
    object_uri = "s3://bucket/hf"
    weight_files = [
        f"{object_uri}/model-00001-of-00002.safetensors",
        f"{object_uri}/model-00002-of-00002.safetensors",
    ]

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.runai_streamer_loader.list_safetensors",
        lambda path: weight_files,
    )

    model_config = SimpleNamespace(
        model=str(model_dir),
        model_weights=object_uri,
        revision=None,
        config_format="hf",
    )

    assert model_loader._prepare_weights(object_uri, model_config) == weight_files


def test_runai_model_loader_download_files(vllm_runner):
    with vllm_runner(test_model, load_format=load_format) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs


@pytest.mark.skip(
    reason="Temporarily disabled due to GCS access issues. "
    "TODO: Re-enable this test once the underlying issue is resolved."
)
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
