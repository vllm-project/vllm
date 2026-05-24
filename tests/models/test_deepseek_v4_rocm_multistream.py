# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.amd import model as rocm_model
from vllm.models.deepseek_v4.nvidia.ops import attention as dsv4_attention


def test_deepseek_v4_rocm_aux_streams_enabled(monkeypatch):
    streams = [object(), object(), object(), object(), object()]

    def make_stream(**kwargs):
        assert kwargs == {"priority": 0}
        return streams.pop()

    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(rocm_model.torch.cuda, "Stream", make_stream)
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", "1")
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MS_STRATEGY", "sglang")

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 5


def test_deepseek_v4_rocm_aux_streams_disabled_by_default(monkeypatch):
    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.delenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", raising=False)

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is None


def test_deepseek_v4_rocm_aux_streams_strategy_off(monkeypatch):
    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", "1")
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MS_STRATEGY", "off")

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is None


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


class _Metadata:
    def __init__(self, num_decodes: int, num_decode_tokens: int):
        self.num_decodes = num_decodes
        self.num_decode_tokens = num_decode_tokens


def test_deepseek_v4_num_decodes_uses_batch_count_not_layer_sum(monkeypatch):
    monkeypatch.setattr(dsv4_attention, "DeepseekSparseSWAMetadata", _Metadata)
    attn_metadata = {
        "layer_0.swa": _Metadata(num_decodes=4, num_decode_tokens=4),
        "layer_1.swa": _Metadata(num_decodes=4, num_decode_tokens=4),
        "layer_2.swa": _Metadata(num_decodes=4, num_decode_tokens=4),
    }

    assert dsv4_attention._deepseek_v4_num_decodes(attn_metadata) == 4
