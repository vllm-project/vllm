# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.amd import model as rocm_model


def test_deepseek_v4_rocm_aux_streams_enabled(monkeypatch):
    streams = [object(), object(), object()]

    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(rocm_model.torch.cuda, "Stream", streams.pop)

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 3


def test_deepseek_v4_rocm_aux_streams_xpu_fallback(monkeypatch):
    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: True)

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is None


def test_deepseek_v4_aux_streams_cuda_behavior_unchanged(monkeypatch):
    streams = [object(), object(), object()]

    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(rocm_model.torch.cuda, "Stream", streams.pop)

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 3
