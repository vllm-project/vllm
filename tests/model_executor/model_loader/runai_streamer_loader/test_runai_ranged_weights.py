# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest

from vllm.model_executor.model_loader import runai_streamer_loader as rsl


def make_loader(monkeypatch, streamed: list[list[str]]):
    loader = rsl.RunaiModelStreamerLoader.__new__(rsl.RunaiModelStreamerLoader)
    loader.load_config = SimpleNamespace(use_tqdm_on_load=False)
    loader._is_distributed = False
    monkeypatch.setattr(
        loader,
        "_prepare_weights",
        lambda *args: [
            "s3://bucket/a.safetensors",
            "s3://bucket/b.safetensors",
        ],
    )
    monkeypatch.setattr(
        rsl,
        "runai_safetensors_weights_iterator",
        lambda files, *args, **kwargs: streamed.append(list(files)) or iter(()),
    )
    return loader


def make_model(filled: dict, extras: list | None = None):
    def load(names, source):
        filled.update(names=names, source=source)
        return extras or []

    return SimpleNamespace(
        is_ranged_checkpoint_weight=lambda name: name.startswith("engram."),
        load_ranged_checkpoint_weights=load,
    )


def write_index(metadata_dir, weight_map):
    (metadata_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_ranged_weights_are_excluded_and_filled(tmp_path, monkeypatch):
    write_index(
        tmp_path,
        {
            "engram.weight": "a.safetensors",
            "other.weight": "b.safetensors",
        },
    )
    streamed: list[list[str]] = []
    filled: dict = {}
    extras = [("other.weight", "tensor")]
    loader = make_loader(monkeypatch, streamed)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        model_weights="s3://bucket",
        revision=None,
    )

    assert (
        list(loader.get_all_weights(model_config, make_model(filled, extras))) == extras
    )

    assert streamed == [["s3://bucket/b.safetensors"]]
    assert filled == {
        "names": {"engram.weight": "a.safetensors"},
        "source": "s3://bucket",
    }


def test_ranged_weights_can_be_disabled(tmp_path, monkeypatch):
    write_index(tmp_path, {"engram.weight": "a.safetensors"})
    monkeypatch.setenv("VLLM_RUNAI_RANGED_WEIGHTS", "0")
    streamed: list[list[str]] = []
    filled: dict = {}
    loader = make_loader(monkeypatch, streamed)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        model_weights="s3://bucket",
        revision=None,
    )

    list(loader.get_all_weights(model_config, make_model(filled)))

    assert streamed == [["s3://bucket/a.safetensors", "s3://bucket/b.safetensors"]]
    assert filled == {}


def test_missing_index_streams_everything(tmp_path, monkeypatch, caplog):
    streamed: list[list[str]] = []
    filled: dict = {}
    loader = make_loader(monkeypatch, streamed)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        model_weights="s3://bucket",
        revision=None,
    )

    with caplog.at_level("WARNING"):
        list(loader.get_all_weights(model_config, make_model(filled)))

    assert streamed == [["s3://bucket/a.safetensors", "s3://bucket/b.safetensors"]]
    assert filled == {}
    assert "streaming every checkpoint tensor" in caplog.text


@pytest.mark.parametrize("weight_map", [None, [], {"name": 1}])
def test_invalid_index_is_ignored(tmp_path, monkeypatch, caplog, weight_map):
    if weight_map is not None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    streamed: list[list[str]] = []
    loader = make_loader(monkeypatch, streamed)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        model_weights="s3://bucket",
        revision=None,
    )

    with caplog.at_level("WARNING"):
        list(loader.get_all_weights(model_config, make_model({})))

    assert len(streamed) == 1 and len(streamed[0]) == 2
