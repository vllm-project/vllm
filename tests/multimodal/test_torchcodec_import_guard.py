# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A torchcodec whose native library cannot load (OSError) counts as unavailable."""

import builtins
import importlib
import sys

import pytest

import vllm.multimodal.media.audio as audio
import vllm.multimodal.video_decoders.torchcodec as video_torchcodec
from vllm.utils.import_utils import check_torchcodec_available

NATIVE_LOAD_ERROR = "libnvrtc.so.13: cannot open shared object file: No such file"


@pytest.fixture
def torchcodec_cannot_load(monkeypatch):
    """Importing torchcodec raises OSError, as a build for another CUDA does."""
    for name in [m for m in sys.modules if m.split(".")[0] == "torchcodec"]:
        monkeypatch.delitem(sys.modules, name)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == "torchcodec":
            raise OSError(NATIVE_LOAD_ERROR)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield
    monkeypatch.undo()
    importlib.reload(audio)
    importlib.reload(video_torchcodec)


def test_check_torchcodec_available_reports_native_load_error(torchcodec_cannot_load):
    with pytest.raises(RuntimeError, match="cannot load its native library"):
        check_torchcodec_available()


def test_media_modules_import_when_torchcodec_cannot_load(torchcodec_cannot_load):
    importlib.reload(audio)
    importlib.reload(video_torchcodec)
    assert audio.AudioDecoder is None
    assert isinstance(audio._torchcodec_import_exc, OSError)
    with pytest.raises(ImportError):
        video_torchcodec.VideoDecoder()
