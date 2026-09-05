# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import types
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm import SamplingParams
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader import runai_streamer_loader as rsl
from vllm.utils.mem_constants import GiB_bytes

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


def test_runai_passes_revision_by_name():
    # revision must reach download_safetensors_index_file_from_hf as the
    # ``revision`` keyword, not the positional ``subfolder`` slot.
    fake_self = types.SimpleNamespace(
        load_config=types.SimpleNamespace(download_dir="/cache", ignore_patterns=[])
    )
    with (
        patch.object(rsl, "is_runai_obj_uri", return_value=False),
        patch.object(rsl, "download_weights_from_hf", return_value="/folder"),
        patch.object(
            rsl, "list_safetensors", return_value=["/folder/model.safetensors"]
        ),
        patch.object(rsl, "download_safetensors_index_file_from_hf") as mock_idx,
    ):
        rsl.RunaiModelStreamerLoader._prepare_weights(
            cast(rsl.RunaiModelStreamerLoader, fake_self), "org/model", "myrev"
        )

    mock_idx.assert_called_once()
    assert mock_idx.call_args.kwargs.get("revision") == "myrev"
    assert "myrev" not in mock_idx.call_args.args


def _runai_loader(extra):
    return rsl.RunaiModelStreamerLoader(
        LoadConfig(load_format="runai_streamer", model_loader_extra_config=extra)
    )


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"typo_key": 1}, "Unexpected extra config"),
        ({"distributed": "yes"}, "distributed must be a bool"),
        ({"concurrency": "16"}, "concurrency must be a positive integer"),
        ({"concurrency": -1}, "concurrency must be a positive integer"),
    ],
)
def test_runai_rejects_invalid_extra_config(extra, match):
    # The loader used to silently drop unknown keys / wrong types / negatives.
    with pytest.raises(ValueError, match=match):
        _runai_loader(extra)


def test_runai_accepts_valid_extra_config():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RUNAI_STREAMER_CONCURRENCY", None)
        os.environ.pop("RUNAI_STREAMER_MEMORY_LIMIT", None)
        loader = _runai_loader(
            {"distributed": True, "concurrency": 16, "memory_limit": 1024}
        )
        assert loader._is_distributed is True
        assert os.environ["RUNAI_STREAMER_CONCURRENCY"] == "16"
        assert os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] == "1024"


def test_runai_invalid_extra_config_leaves_environ_untouched():
    # A later invalid key must not leave an earlier valid key applied to
    # os.environ (all values are validated before any global mutation).
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RUNAI_STREAMER_CONCURRENCY", None)
        with pytest.raises(ValueError, match="memory_limit must be an integer >= -1"):
            _runai_loader({"concurrency": 16, "memory_limit": -5})
        assert "RUNAI_STREAMER_CONCURRENCY" not in os.environ


@pytest.mark.parametrize(
    "model_config,expected_source",
    [
        # Object storage: model_weights holds the URI, model the pulled config dir.
        (
            types.SimpleNamespace(
                model="/tmp/pulled-config-files",
                model_weights="s3://bucket/weights",
                revision="myrev",
            ),
            "s3://bucket/weights",
        ),
        # HF repo or local path: model_weights is empty, model is the source.
        (
            types.SimpleNamespace(
                model="org/model", model_weights="", revision="myrev"
            ),
            "org/model",
        ),
    ],
    ids=["object_storage", "hf_repo"],
)
def test_runai_get_all_weights_resolves_source(model_config, expected_source):
    fake_self = types.SimpleNamespace(
        _get_weights_iterator=lambda path, revision: iter(
            [(f"{path}@{revision}", None)]
        )
    )

    weights = list(
        rsl.RunaiModelStreamerLoader.get_all_weights(
            fake_self, model_config, model=None
        )
    )

    assert weights == [(f"{expected_source}@myrev", None)]


@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        (
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
        ),
    ],
)
def test_runai_deep_sleep_reload_weights(base_model, mul_model, add_model, vllm_runner):
    free, total = torch.accelerator.get_memory_info()
    used_bytes_baseline = total - free

    def sleep_and_reload(llm, path):
        # Level 2 discards the parameter memory, so after wake_up the weights
        # can only come from the reload: a tensor the streamer fails to
        # deliver changes the output instead of keeping a stale valid value.
        llm.get_llm().sleep(level=2)
        free, total = torch.accelerator.get_memory_info()
        assert total - free - used_bytes_baseline < 3 * GiB_bytes
        llm.get_llm().wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights", kwargs={"weights_path": path})
        free, total = torch.accelerator.get_memory_info()
        assert total - free - used_bytes_baseline < 4 * GiB_bytes
        llm.get_llm().wake_up(tags=["kv_cache"])

    with vllm_runner(
        model_name=base_model,
        load_format=load_format,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        base_output = llm.generate_greedy(["3 4 ="], max_tokens=4)

        sleep_and_reload(llm, mul_model)
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        sleep_and_reload(llm, add_model)
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp

        # Round trip to the original weights over discarded memory must
        # reproduce the pre-sleep output.
        sleep_and_reload(llm, base_model)
        assert llm.generate_greedy(["3 4 ="], max_tokens=4) == base_output
