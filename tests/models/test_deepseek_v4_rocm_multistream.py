# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only DeepSeek-V4 tests"
)


def _swa_metadata(
    num_decode_tokens: int,
    num_prefill_tokens: int = 0,
) -> dsv4_attention.DeepseekSparseSWAMetadata:
    return dsv4_attention.DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_size=256,
        num_decodes=num_decode_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
    )


def _use_rocm_multistream(
    cudagraph_runtime_mode: dsv4_attention.CUDAGraphMode,
    metadata: dsv4_attention.DeepseekSparseSWAMetadata,
) -> bool:
    class _ForwardContext:
        pass

    class _Wrapper:
        aux_stream_list = [object()]

    forward_context = _ForwardContext()
    forward_context.cudagraph_runtime_mode = cudagraph_runtime_mode
    attn_metadata = {"layer_0.swa": metadata}

    wrapper_cls = dsv4_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    method = wrapper_cls._use_rocm_csa_multistream
    return method(_Wrapper(), forward_context, attn_metadata)


def test_deepseek_v4_rocm_multistream_decode_policy():
    decode_metadata = _swa_metadata(num_decode_tokens=4)

    assert (
        _use_rocm_multistream(dsv4_attention.CUDAGraphMode.NONE, decode_metadata)
        is True
    )
    assert (
        _use_rocm_multistream(dsv4_attention.CUDAGraphMode.PIECEWISE, decode_metadata)
        is True
    )
    assert (
        _use_rocm_multistream(dsv4_attention.CUDAGraphMode.FULL, decode_metadata)
        is False
    )

    mixed_metadata = _swa_metadata(num_decode_tokens=4, num_prefill_tokens=1)
    assert (
        _use_rocm_multistream(dsv4_attention.CUDAGraphMode.PIECEWISE, mixed_metadata)
        is False
    )


def test_deepseek_v4_rocm_post_rmsnorm_stream_mapping(monkeypatch):
    calls = []
    streams = [object()]

    def fake_maybe_execute_in_parallel(
        default_fn,
        aux_fn,
        start_event,
        done_event,
        aux_stream=None,
    ):
        assert aux_stream is streams[0]

        q = default_fn()
        compressor_result = aux_fn()
        return q, compressor_result

    monkeypatch.setattr(
        dsv4_attention,
        "maybe_execute_in_parallel",
        fake_maybe_execute_in_parallel,
    )

    class _WqB:
        def __call__(self, qr):
            calls.append("wq_b")
            return torch.empty((qr.shape[0], 3))

    class _Indexer:
        def __call__(
            self,
            hidden_states,
            qr,
            indexer_kv_score,
            indexer_weights,
            positions,
            rotary_emb,
            use_aux_stream=True,
        ):
            calls.append(("indexer", use_aux_stream))
            return object()

    class _Compressor:
        def __call__(self, kv_score, positions, rotary_emb):
            calls.append("compressor")
            return object()

    wrapper = SimpleNamespace(
        wq_b=_WqB(),
        indexer=_Indexer(),
        compressor=_Compressor(),
        n_local_heads=1,
        head_dim=3,
        indexer_rotary_emb=object(),
        rotary_emb=object(),
        ln_events=[object(), object(), object()],
    )

    def fake_kv_insert(q, kv, positions, attn_metadata):
        calls.append("kv_insert")
        return q

    def fail_project_compressor_kv_score(hidden_states, compressor):
        raise AssertionError("kv_score should be reused in this test")

    wrapper._fused_qnorm_rope_kv_insert = fake_kv_insert
    wrapper._project_compressor_kv_score = fail_project_compressor_kv_score

    hidden_states = torch.empty((2, 3))
    qr = torch.empty((2, 3))
    kv = torch.empty((2, 3))
    positions = torch.empty(2, dtype=torch.int64)
    kv_score = torch.empty((2, 3))
    indexer_kv_score = torch.empty((2, 3))
    indexer_weights = torch.empty((2, 3))

    method = dsv4_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    q = method._post_rmsnorm_prepare(
        wrapper,
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        None,
        streams,
        True,
    )

    assert q.shape == (2, 1, 3)
    assert calls == ["wq_b", "kv_insert", "compressor", ("indexer", False)]
