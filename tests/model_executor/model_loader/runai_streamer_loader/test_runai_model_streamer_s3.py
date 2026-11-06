# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from runai_model_streamer.safetensors_streamer.streamer_mock import StreamerPatcher
from safetensors.torch import load_file, save_file

from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.repo_utils import hf_api

from .conftest import RunaiDummyExecutor

load_format = "runai_streamer"
test_model = "openai-community/gpt2"


def test_runai_model_loader_download_files_s3_mocked_with_patch(
    vllm_runner,
    tmp_path: Path,
    monkeypatch,
):
    patcher = StreamerPatcher(str(tmp_path))

    test_mock_s3_model = "s3://my-mock-bucket/gpt2/"

    # Download model from HF
    mock_model_dir = f"{tmp_path}/gpt2"
    hf_api().snapshot_download(repo_id=test_model, local_dir=mock_model_dir)

    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_list_safetensors",
        patcher.shim_list_safetensors,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_pull_files",
        patcher.shim_pull_files,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.SafetensorsStreamer",
        patcher.create_mock_streamer,
    )

    engine_args = EngineArgs(
        model=test_mock_s3_model,
        load_format=load_format,
        tensor_parallel_size=1,
    )

    vllm_config = engine_args.create_engine_config()

    executor = RunaiDummyExecutor(vllm_config)
    executor.driver_worker.load_model()


def test_runai_reload_weights_path_from_mocked_s3(tmp_path: Path, monkeypatch):
    # An engine started from object storage keeps the bucket URI in
    # model_config.model_weights. A reload with weights_path must stream the
    # new checkpoint, not re-stream the one the engine started from.
    patcher = StreamerPatcher(str(tmp_path))
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_list_safetensors",
        patcher.shim_list_safetensors,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_pull_files",
        patcher.shim_pull_files,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.SafetensorsStreamer",
        patcher.create_mock_streamer,
    )

    base_dir = tmp_path / "gpt2"
    hf_api().snapshot_download(
        repo_id=test_model,
        local_dir=str(base_dir),
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    # Second checkpoint under another prefix: the final layer norm scaled, so a
    # reload that silently re-streams the original bucket is distinguishable.
    tensors = load_file(base_dir / "model.safetensors")
    original = tensors["ln_f.weight"].clone()
    modified = original * 2
    tensors["ln_f.weight"] = modified
    mod_dir = tmp_path / "gpt2-mod"
    mod_dir.mkdir()
    save_file(tensors, mod_dir / "model.safetensors", metadata={"format": "pt"})

    vllm_config = EngineArgs(
        model="s3://my-mock-bucket/gpt2/",
        load_format=load_format,
        tensor_parallel_size=1,
    ).create_engine_config()
    assert vllm_config.model_config.model_weights == "s3://my-mock-bucket/gpt2/"

    executor = RunaiDummyExecutor(vllm_config)
    executor.driver_worker.load_model()
    model = executor.driver_worker.model_runner.get_model()
    (param,) = [p for n, p in model.named_parameters() if n.endswith("ln_f.weight")]

    def current() -> torch.Tensor:
        return param.detach().float().cpu()

    torch.testing.assert_close(current(), original, rtol=1e-2, atol=1e-2)

    # No weights_path: re-stream the checkpoint the engine started from, i.e.
    # the bucket kept in model_weights. Zero the live tensor first so a reload
    # that did nothing is distinguishable from one that streamed the bucket.
    param.data.zero_()
    executor.driver_worker.reload_weights()
    torch.testing.assert_close(current(), original, rtol=1e-2, atol=1e-2)

    # weights_path: the new prefix must win over the stale model_weights.
    executor.driver_worker.reload_weights(weights_path="s3://my-mock-bucket/gpt2-mod/")
    torch.testing.assert_close(current(), modified, rtol=1e-2, atol=1e-2)

    # After the override, a bare reload means the new checkpoint, not the
    # one the engine was started with.
    param.data.zero_()
    executor.driver_worker.reload_weights()
    torch.testing.assert_close(current(), modified, rtol=1e-2, atol=1e-2)
