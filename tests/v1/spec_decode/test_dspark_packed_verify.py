# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side checks for DSpark target verification on UltraQuant.

DSpark scores 1 + num_speculative_tokens rows per request in the target. With
k=7 that is width 8, the widest verify the UltraQuant decode kernel takes in
one KV read. These tests cover the routing, the ragged mixed batch, and the
partial-output pool size for a verify step. The kernel itself is covered by
tests/kernels/ultraquant.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
from vllm.v1.attention.backends.ultraquant_attn import (
    UltraQuantAttentionImpl,
    UltraQuantMetadataBuilder,
)
from vllm.v1.attention.ops import flydsl_ultraquant_decode

pytestmark = pytest.mark.cpu_test


def _verify_builder(monkeypatch) -> UltraQuantMetadataBuilder:
    from vllm.v1.attention.backends import turboquant_attn
    from vllm.v1.kv_cache_interface import AttentionSpec

    monkeypatch.setattr(
        turboquant_attn, "is_workspace_manager_initialized", lambda: False
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
        ),
        model_config=SimpleNamespace(
            max_model_len=8192,
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda parallel_config: 8,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        attention_config=SimpleNamespace(tq_max_kv_splits_for_cuda_graph=4),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            max_num_new_slots_for_drafting=0,
        ),
    )
    spec = AttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        state_content_bytes=272,
    )
    return UltraQuantMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layers.0.self_attn.attn"],
        vllm_config=config,
        device=torch.device("cpu"),
    )


def _cam(query_lens, seq_lens):
    qsl = [0]
    for length in query_lens:
        qsl.append(qsl[-1] + length)
    qsl_cpu = torch.tensor(qsl, dtype=torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl_cpu,
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq,
        num_reqs=len(query_lens),
        num_actual_tokens=qsl[-1],
        max_query_len=max(query_lens),
        max_seq_len=int(max(seq_lens)),
        block_table_tensor=torch.zeros(len(query_lens), 2, dtype=torch.int32),
        slot_mapping=torch.zeros(qsl[-1], dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq.clone(),
    )


def test_pure_dspark_verify_routes_as_decode(monkeypatch):
    builder = _verify_builder(monkeypatch)
    # Two DSpark verifies of width 8: seven drafted tokens plus the bonus.
    meta = builder.build(0, _cam([8, 8], [128, 96]))
    assert meta.num_decodes == 2
    assert meta.is_prefill is False
    assert meta.max_query_len == 8


def test_mixed_dspark_verify_keeps_real_width(monkeypatch):
    builder = _verify_builder(monkeypatch)
    meta = builder.build(0, _cam([8, 6, 64], [128, 96, 64]))
    assert meta.num_decodes == 2
    assert meta.is_prefill is True

    impl = UltraQuantAttentionImpl.__new__(UltraQuantAttentionImpl)
    impl.num_heads = 8
    impl.head_size = 4
    impl.num_kv_heads = 1
    seen: dict[str, object] = {}

    def decode(query, kv_cache, attn_metadata, *args, **kwargs):
        del kv_cache, args, kwargs
        seen["decode_max_q"] = int(attn_metadata.max_query_len)
        seen["decode_rows"] = query.shape[0]
        seen["decode_reqs"] = int(attn_metadata.seq_lens.shape[0])
        seen["qsl"] = attn_metadata.query_start_loc.tolist()
        return torch.zeros(query.shape[0], impl.num_heads, impl.head_size)

    def prefill(query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        del key, value, kv_cache, args, kwargs
        seen["prefill_rows"] = query.shape[0]
        return torch.zeros(query.shape[0], impl.num_heads, impl.head_size)

    impl._ensure_on_device = lambda layer, device: None
    impl._decode_attention = decode
    impl._prefill_attention = prefill

    n = meta.num_actual_tokens
    layer = SimpleNamespace(_tq_Pi=None, _tq_PiT=None, _tq_centroids=None)
    impl.forward(
        layer,
        torch.zeros(n, impl.num_heads * impl.head_size),
        torch.zeros(n, impl.num_kv_heads * impl.head_size),
        torch.zeros(n, impl.num_kv_heads * impl.head_size),
        torch.zeros(2, 4, 8),
        meta,
    )
    assert seen == {
        "decode_max_q": 8,
        "decode_rows": 14,
        "decode_reqs": 2,
        "qsl": [0, 8, 14],
        "prefill_rows": 64,
    }

    base = TurboQuantAttentionImpl.__new__(TurboQuantAttentionImpl)
    assert base._decode_slice_max_query_len(meta, meta.num_decodes) == 1


def _pool_config(num_speculative_tokens: int, query_len: int) -> SimpleNamespace:
    speculative = (
        SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens
        else None
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8, 16]),
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=8192),
        uniform_decode_query_len=query_len,
        speculative_config=speculative,
    )


def test_partial_output_pool_covers_a_k7_verify_step(monkeypatch):
    """A width-8 verify has 8 rows per request. The first allocation must hold
    them, because a later grow frees buffers a CUDA graph already captured."""
    import vllm.config

    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _pool_config(7, 8)
    )
    assert flydsl_ultraquant_decode._detect_max_capture_B() == 32 * 8


def test_partial_output_pool_without_speculation_is_unchanged(monkeypatch):
    import vllm.config

    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _pool_config(0, 1)
    )
    assert flydsl_ultraquant_decode._detect_max_capture_B() == 32
