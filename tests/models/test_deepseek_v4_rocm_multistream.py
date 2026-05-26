# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.amd import model as rocm_model
from vllm.models.deepseek_v4.nvidia.ops import attention as dsv4_attention


def test_deepseek_v4_rocm_aux_streams_enabled(monkeypatch):
    streams = [object(), object(), object(), object(), object()]

    def make_stream(**kwargs):
        assert kwargs == {"priority": -1}
        return streams.pop()

    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(rocm_model.torch.cuda, "Stream", make_stream)
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", "1")
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MS_STRATEGY", "overlap")

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 5


def test_deepseek_v4_rocm_aux_streams_enabled_by_default(monkeypatch):
    streams = [object(), object(), object(), object(), object()]

    def make_stream(**kwargs):
        assert kwargs == {"priority": -1}
        return streams.pop()

    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(rocm_model.torch.cuda, "Stream", make_stream)
    monkeypatch.delenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", raising=False)

    aux_streams = rocm_model.make_deepseek_v4_aux_streams()

    assert aux_streams is not None
    assert len(aux_streams) == 5


def test_deepseek_v4_rocm_aux_streams_disabled_by_env(monkeypatch):
    monkeypatch.setattr(rocm_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_model.current_platform, "is_xpu", lambda: False)
    monkeypatch.setenv("VLLM_ROCM_DSV4_CSA_MULTISTREAM", "0")

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


def test_deepseek_v4_rocm_strategy_cuda_behavior_unchanged(monkeypatch):
    class _ForwardContext:
        cudagraph_runtime_mode = dsv4_attention.CUDAGraphMode.PIECEWISE

    class _Wrapper:
        aux_stream_list = [object(), object(), object()]
        indexer = object()

    monkeypatch.setattr(dsv4_attention.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(dsv4_attention.envs, "VLLM_ROCM_DSV4_CSA_MULTISTREAM", True)
    monkeypatch.setattr(
        dsv4_attention.envs, "VLLM_ROCM_DSV4_CSA_MS_STRATEGY", "overlap"
    )

    wrapper_cls = dsv4_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    method = wrapper_cls._rocm_csa_ms_strategy_for_step

    assert method(_Wrapper(), _ForwardContext(), None) == "off"


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

    monkeypatch.setattr(dsv4_attention.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(dsv4_attention.envs, "VLLM_ROCM_DSV4_CSA_MULTISTREAM", True)
    monkeypatch.setattr(
        dsv4_attention.envs, "VLLM_ROCM_DSV4_CSA_MS_STRATEGY", "overlap"
    )
    monkeypatch.setattr(
        dsv4_attention.envs,
        "VLLM_ROCM_DSV4_CSA_MS_GRAPH_MODES",
        {"none", "piecewise"},
    )
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
