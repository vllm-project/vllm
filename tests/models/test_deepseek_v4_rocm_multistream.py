# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.amd import model as dsv4_model
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only DeepSeek-V4 tests"
)


def test_deepseek_v4_rocm_aux_streams_enabled(monkeypatch):
    streams = [object(), object(), object(), object(), object()]

    def make_stream(**kwargs):
        assert kwargs == {"priority": -1}
        return streams.pop()

    monkeypatch.setattr(dsv4_model.torch.cuda, "Stream", make_stream)

    aux_streams = dsv4_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 5


class _Metadata:
    def __init__(
        self,
        num_decodes: int,
        num_decode_tokens: int,
        num_prefill_tokens: int = 0,
    ):
        self.num_decodes = num_decodes
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens


def _rocm_ms_strategy_for_decodes(
    monkeypatch,
    num_decodes: int,
    num_prefill_tokens: int = 0,
) -> str:
    class _ForwardContext:
        cudagraph_runtime_mode = dsv4_attention.CUDAGraphMode.PIECEWISE

    class _Wrapper:
        aux_stream_list = [object(), object(), object(), object(), object()]
        indexer = object()

    monkeypatch.setattr(dsv4_attention, "DeepseekSparseSWAMetadata", _Metadata)
    attn_metadata = {
        "layer_0.swa": _Metadata(num_decodes, num_decodes, num_prefill_tokens)
    }

    wrapper_cls = dsv4_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    method = wrapper_cls._rocm_csa_ms_strategy_for_step
    return method(_Wrapper(), _ForwardContext(), attn_metadata)


def test_deepseek_v4_rocm_multistream_all_decode_counts(monkeypatch):
    assert _rocm_ms_strategy_for_decodes(monkeypatch, 4) == "overlap"
    assert _rocm_ms_strategy_for_decodes(monkeypatch, 64) == "overlap"
    assert _rocm_ms_strategy_for_decodes(monkeypatch, 128) == "overlap"
    assert _rocm_ms_strategy_for_decodes(monkeypatch, 512) == "overlap"


def test_deepseek_v4_rocm_multistream_prefill_stays_off(monkeypatch):
    strategy = _rocm_ms_strategy_for_decodes(monkeypatch, 4, num_prefill_tokens=1)

    assert strategy == "off"
