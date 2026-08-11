# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.model_loader import tensorizer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_serialize_vllm_model_forwards_model_revision(
    monkeypatch: pytest.MonkeyPatch,
):
    @contextlib.contextmanager
    def fake_open_stream(*args, **kwargs):
        yield io.BytesIO()

    snapshot_download = Mock()
    monkeypatch.setattr(tensorizer, "open_stream", fake_open_stream)
    monkeypatch.setattr(tensorizer, "TensorSerializer", Mock())
    monkeypatch.setattr(
        tensorizer,
        "hf_api",
        lambda: SimpleNamespace(snapshot_download=snapshot_download),
    )

    model = torch.nn.Module()
    tensorizer_config = SimpleNamespace(
        _construct_tensorizer_args=lambda: SimpleNamespace(
            tensorizer_uri="/tmp/model.tensors",
            tensorizer_dir="/tmp/out",
            stream_kwargs={},
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

    assert snapshot_download.call_args.args == ("repo/model",)
    assert snapshot_download.call_args.kwargs["revision"] == "pinned-revision"
