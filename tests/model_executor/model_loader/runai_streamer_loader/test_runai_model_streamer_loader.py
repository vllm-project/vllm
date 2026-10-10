# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import types
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

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
        rsl.RunaiModelStreamerLoader._prepare_weights(fake_self, "org/model", "myrev")

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
        _get_weights_iterator=lambda path, revision, is_unused_weight=None: iter(
            [(f"{path}@{revision}", None)]
        )
    )

    weights = list(
        rsl.RunaiModelStreamerLoader.get_all_weights(
            fake_self, model_config, model=None
        )
    )

    assert weights == [(f"{expected_source}@myrev", None)]


@pytest.mark.parametrize("selection", ["draft", "unadapted", "reject_all"])
def test_runai_selects_shards_before_streaming(tmp_path, monkeypatch, selection):
    """Unused shards must not reach RunAI; mixed shards retain their tensors."""
    contents = [
        {"main.weight": torch.tensor([1.0])},
        {"main.bias": torch.tensor([2.0]), "draft.weight": torch.tensor([3.0])},
        {"draft.bias": torch.tensor([4.0])},
    ]
    files = []
    for i, tensors in enumerate(contents):
        path = str(tmp_path / f"part-{i}.safetensors")
        save_file(tensors, path)
        files.append(path)
    monkeypatch.setattr(rsl, "list_safetensors", lambda path: files)

    opened = []
    real_stream = rsl.runai_safetensors_weights_iterator

    def stream(paths, *args):
        opened.extend(paths)
        yield from real_stream(paths, *args)

    monkeypatch.setattr(rsl, "runai_safetensors_weights_iterator", stream)
    model = torch.nn.Module()
    if selection == "draft":
        model.is_unused_checkpoint_weight = lambda name: not name.startswith("draft.")
    elif selection == "reject_all":
        model.is_unused_checkpoint_weight = lambda name: True
    config = types.SimpleNamespace(model=str(tmp_path), model_weights="", revision=None)
    loaded = dict(_runai_loader({}).get_all_weights(config, model))

    expected = files[1:] if selection == "draft" else files
    assert opened == expected
    assert set(loaded) == {
        name
        for tensors in (contents[1:] if selection == "draft" else contents)
        for name in tensors
    }
    for name, tensor in {**contents[1], **contents[2]}.items():
        torch.testing.assert_close(loaded[name], tensor)


def _check_distributed_selection(rank, rendezvous, scenario):
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    root = "s3://bucket/weights"
    if scenario == "different_roots":
        root += str(rank)
    files = [root + "/main.safetensors", root + "/draft.safetensors"]

    def pull_files(source, destination, **kwargs):
        if scenario == "metadata_failure" and rank == 1:
            raise OSError("Index temporarily unavailable")
        (Path(destination) / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "main.weight": "main.safetensors",
                        "draft.weight": "draft.safetensors",
                    }
                }
            )
        )

    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(rsl, "list_safetensors", lambda path: files)
            monkeypatch.setattr(rsl, "runai_pull_files", pull_files)
            monkeypatch.setattr(
                rsl,
                "get_world_group",
                lambda: types.SimpleNamespace(
                    world_size=2, cpu_group=torch.distributed.group.WORLD
                ),
                raising=False,
            )
            # Exercise real selection and Gloo; observe the distributed stream input.
            monkeypatch.setattr(
                rsl,
                "runai_safetensors_weights_iterator",
                lambda paths, *args: iter((path, None) for path in paths),
            )
            model = torch.nn.Module()
            if not (scenario == "missing_hook" and rank == 1):
                needed = (
                    "main." if scenario == "different_roles" and rank == 1 else "draft."
                )
                model.is_unused_checkpoint_weight = lambda name: (
                    not name.startswith(needed)
                )
            config = types.SimpleNamespace(model=root, model_weights="", revision=None)
            selected = [
                path
                for path, _ in _runai_loader({"distributed": True}).get_all_weights(
                    config, model
                )
            ]
            expected = files[1:] if scenario == "healthy" else files
            assert selected == expected
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "healthy",
        "metadata_failure",
        "different_roles",
        "missing_hook",
        "different_roots",
    ],
)
def test_distributed_runai_preserves_every_ranks_required_shards(tmp_path, scenario):
    """All ranks must stream the same union even if one cannot filter safely."""
    torch.multiprocessing.spawn(
        _check_distributed_selection,
        args=((tmp_path / "rendezvous").as_uri(), scenario),
        nprocs=2,
    )


@pytest.mark.parametrize("scheme", ["s3", "gs", "az"])
@pytest.mark.parametrize(
    "index_kind", ["valid", "missing", "malformed", "unavailable", "unadapted"]
)
def test_runai_remote_selection_uses_actual_weight_source(
    tmp_path, monkeypatch, scheme, index_kind
):
    root = f"{scheme}://bucket/weights"
    files = [
        root + "/main/part.safetensors",
        root + "/draft/part.safetensors",
        root + "/mixed.safetensors",
        root + "/extra.safetensors",
    ]
    monkeypatch.setattr(rsl, "list_safetensors", lambda path: files)
    pulled = []

    def pull_files(source, destination, allow_pattern, ignore_pattern=None):
        pulled.append((source.rstrip("/"), allow_pattern))
        if index_kind == "unavailable":
            # Storage SDKs do not necessarily derive errors from OSError.
            raise Exception("Object metadata unavailable")
        index = Path(destination) / "model.safetensors.index.json"
        if index_kind == "valid":
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "main.weight": "main/part.safetensors",
                            "draft.weight": "draft/part.safetensors",
                            "main.bias": "mixed.safetensors",
                            "draft.bias": "mixed.safetensors",
                        }
                    }
                )
            )
        elif index_kind == "malformed":
            index.write_text("not JSON")

    monkeypatch.setattr(rsl, "runai_pull_files", pull_files, raising=False)
    opened = []

    def stream(paths, *args):
        opened.extend(paths)
        return iter(())

    monkeypatch.setattr(rsl, "runai_safetensors_weights_iterator", stream)
    model = torch.nn.Module()
    if index_kind != "unadapted":
        model.is_unused_checkpoint_weight = lambda name: not name.startswith("draft.")
    # The config directory is not the checkpoint: using its index could drop
    # tensors from an independent model_weights source.
    (tmp_path / "model.safetensors.index.json").write_text("not the weights index")
    config = types.SimpleNamespace(
        model=str(tmp_path), model_weights=root, revision=None
    )
    list(_runai_loader({}).get_all_weights(config, model))

    assert opened == (files[1:] if index_kind == "valid" else files)
    assert pulled == (
        []
        if index_kind == "unadapted"
        else [
            (
                root,
                [
                    "weights/model.safetensors.index.json",
                    "model.safetensors.index.json",
                ],
            )
        ]
    )


@pytest.mark.parametrize(
    "weight_map",
    [None, {}, [], {"draft.weight": None}, {"draft.weight": "../part.safetensors"}],
)
def test_runai_invalid_weight_map_keeps_original_files(monkeypatch, weight_map):
    files = ["s3://bucket/model/part.safetensors"]
    monkeypatch.setattr(rsl, "list_safetensors", lambda path: files)

    def pull_files(source, directory, **kwargs):
        (Path(directory) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    monkeypatch.setattr(rsl, "runai_pull_files", pull_files)
    monkeypatch.setattr(
        rsl, "runai_safetensors_weights_iterator", lambda paths, *args: iter(paths)
    )
    model = torch.nn.Module()
    model.is_unused_checkpoint_weight = lambda name: True
    config = types.SimpleNamespace(
        model="s3://bucket/model", model_weights="", revision=None
    )
    assert list(_runai_loader({}).get_all_weights(config, model)) == files


def test_runai_model_rule_error_is_not_hidden_by_metadata_fallback(monkeypatch):
    files = ["s3://bucket/model/part.safetensors"]
    monkeypatch.setattr(rsl, "list_safetensors", lambda path: files)

    def pull_files(source, directory, **kwargs):
        (Path(directory) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"draft.weight": "part.safetensors"}})
        )

    monkeypatch.setattr(rsl, "runai_pull_files", pull_files)
    model = torch.nn.Module()

    def invalid_rule(name):
        raise ValueError("Invalid model rule")

    model.is_unused_checkpoint_weight = invalid_rule
    config = types.SimpleNamespace(
        model="s3://bucket/model", model_weights="", revision=None
    )
    with pytest.raises(ValueError, match="Invalid model rule"):
        list(_runai_loader({}).get_all_weights(config, model))


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
