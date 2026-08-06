# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import io
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.model_loader import tensorizer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_serialize_extra_artifacts_forwards_revision(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeHFAPI:
        def snapshot_download(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr(tensorizer, "hf_api", lambda: FakeHFAPI())
    tensorizer.serialize_extra_artifacts(
        SimpleNamespace(tensorizer_dir="/tmp/out", stream_kwargs={}),
        "repo/model",
        revision="pinned-revision",
    )

    assert seen["args"] == ("repo/model",)
    assert seen["kwargs"]["revision"] == "pinned-revision"


def test_serialize_vllm_model_passes_model_revision_to_artifacts(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, Any] = {}

    @contextlib.contextmanager
    def fake_open_stream(*args, **kwargs):
        yield io.BytesIO()

    class FakeSerializer:
        def __init__(self, *args, **kwargs):
            pass

        def write_module(self, model):
            pass

        def close(self):
            pass

    def fake_serialize_extra_artifacts(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

    monkeypatch.setattr(tensorizer, "open_stream", fake_open_stream)
    monkeypatch.setattr(tensorizer, "TensorSerializer", FakeSerializer)
    monkeypatch.setattr(
        tensorizer, "serialize_extra_artifacts", fake_serialize_extra_artifacts
    )

    model = torch.nn.Module()
    tensorizer_config = SimpleNamespace(
        _construct_tensorizer_args=lambda: SimpleNamespace(
            tensorizer_uri="/tmp/model.tensors", stream_kwargs={}
        ),
        encryption_keyfile=None,
        _is_sharded=False,
        serialization_kwargs=None,
    )
    model_config = SimpleNamespace(
        served_model_name="repo/model",
        revision="pinned-revision",
    )

    tensorizer.serialize_vllm_model(model, tensorizer_config, model_config)

    assert seen["args"][1] == "repo/model"
    assert seen["kwargs"]["revision"] == "pinned-revision"
