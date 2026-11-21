# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative verify requests stay on the decode path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.attention.backends.ultraquant_attn import (
    UltraQuantAttentionImpl,
    UltraQuantMetadataBuilder,
    _split_multi_query_decode,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import reference_ultraquant_attention
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP device")
def test_ragged_verify_batch_matches_causal_reference():
    device = torch.device("cuda")
    set_random_seed(0)

    query_lens = [3, 1, 4]
    B, Hk, Hq, N, D = len(query_lens), 1, 6, 128, 256
    block_size = 32
    blocks_per_seq = N // block_size
    scale = D**-0.5

    key = torch.randn(B, N, Hk, D, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        B * blocks_per_seq,
        block_size,
        Hk,
        slot_size(D),
        dtype=torch.uint8,
        device=device,
    )
    ultraquant_store(
        key.reshape(-1, Hk, D),
        value.reshape(-1, Hk, D),
        kv_cache,
        torch.arange(B * N, dtype=torch.int64, device=device),
    )
    block_table = torch.arange(
        B * blocks_per_seq, dtype=torch.int32, device=device
    ).view(B, blocks_per_seq)

    # One zero-length padded request and two padded query rows, as the model
    # runner produces for a CUDA-graph batch.
    num_rows = sum(query_lens) + 2
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist(), sum(query_lens)],
        dtype=torch.int32,
        device=device,
    )
    metadata = _split_multi_query_decode(
        num_rows,
        TurboQuantMetadata(
            seq_lens=torch.tensor([N] * B + [0], dtype=torch.int32, device=device),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
            block_table=torch.cat([block_table, torch.zeros_like(block_table[:1])]),
            query_start_loc=query_start_loc,
            max_seq_len=N,
        ),
    )
    query = torch.randn(num_rows, Hq, D, dtype=torch.bfloat16, device=device)
    actual = ultraquant_unified_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=metadata.block_table,
        seq_lens=metadata.seq_lens,
        query_start_loc=metadata.query_start_loc,
        scale=scale,
        max_query_len=1,
        max_seq_len=N,
    )

    row = 0
    for req, q_len in enumerate(query_lens):
        for j in range(q_len):
            ctx = N - q_len + 1 + j
            expected = reference_ultraquant_attention(
                query=query[row : row + 1],
                key=key[req : req + 1, :ctx],
                value=value[req : req + 1, :ctx],
                scale=scale,
            )
            cos = torch.nn.functional.cosine_similarity(
                actual[row].float().reshape(-1),
                expected.float().reshape(-1),
                dim=0,
            ).item()
            assert cos > 0.995, (req, j, cos)
            row += 1
    assert torch.isfinite(actual[row:]).all()


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
            get_num_attention_heads=lambda parallel_config: 6,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        attention_config=SimpleNamespace(tq_max_kv_splits_for_cuda_graph=4),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
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


def test_pure_verify_batch_routes_as_decode(monkeypatch):
    builder = _verify_builder(monkeypatch)
    # Two verify requests, no prefill. max_query_len > 1 must not set is_prefill.
    meta = builder.build(0, _cam([6, 4], [128, 96]))
    assert meta.num_decodes == 2
    assert meta.is_prefill is False

    mixed = builder.build(0, _cam([6, 64], [128, 64]))
    assert mixed.num_decodes == 1
    assert mixed.is_prefill is True


def test_mixed_verify_slice_keeps_real_width(monkeypatch):
    """Prefill stays on prefill attention. The verify half keeps its width.

    Query lengths [2, 4, 64] are two verify requests plus one prefill.
    The verify slice is ragged, and its longest request is 4, so the decode
    call carries max_query_len 4. TurboQuant's own helper stays at 1.
    """
    builder = _verify_builder(monkeypatch)
    meta = builder.build(0, _cam([2, 4, 64], [128, 96, 64]))
    assert meta.num_decodes == 2
    assert meta.num_decode_tokens == 6
    assert meta.is_prefill is True

    impl = UltraQuantAttentionImpl.__new__(UltraQuantAttentionImpl)
    impl.num_heads = 6
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
        seen["prefill"] = bool(attn_metadata.is_prefill)
        return torch.zeros(query.shape[0], impl.num_heads, impl.head_size)

    impl._ensure_on_device = lambda layer, device: None
    impl._decode_attention = decode
    impl._prefill_attention = prefill

    n = meta.num_actual_tokens
    flat_q = impl.num_heads * impl.head_size
    flat_kv = impl.num_kv_heads * impl.head_size
    layer = SimpleNamespace(_tq_Pi=None, _tq_PiT=None, _tq_centroids=None)
    impl.forward(
        layer,
        torch.zeros(n, flat_q),
        torch.zeros(n, flat_kv),
        torch.zeros(n, flat_kv),
        torch.zeros(2, 4, 8),
        meta,
    )

    assert seen == {
        "decode_max_q": 4,
        "decode_rows": 6,
        "decode_reqs": 2,
        "qsl": [0, 2, 6],
        "prefill_rows": 64,
        "prefill": True,
    }

    one = builder.build(0, _cam([1, 64], [32, 64]))
    assert impl._decode_slice_max_query_len(one, one.num_decodes) == 1

    wide = builder.build(0, _cam([2, 4, 64], [128, 96, 64]))
    wide.query_start_loc_cpu = torch.tensor([0, 9, 73], dtype=torch.int32)
    assert impl._decode_slice_max_query_len(wide, 1) == 9

    base = TurboQuantAttentionImpl.__new__(TurboQuantAttentionImpl)
    assert base._decode_slice_max_query_len(meta, meta.num_decodes) == 1


def _pool_cfg(max_seqs, max_tokens, capture, nspec, query_len):
    spec = None if nspec is None else SimpleNamespace(num_speculative_tokens=nspec)
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=max_tokens,
        ),
        speculative_config=spec,
        uniform_decode_query_len=query_len,
    )


def test_partial_output_pool_covers_verify_rows(monkeypatch):
    """The first pool allocation covers a verify step, not just one row per request."""
    from vllm.v1.attention.ops import flydsl_ultraquant_decode as mod

    def install(cfg):
        monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: cfg)

    install(_pool_cfg(256, 8192, [1, 2, 512], None, 1))
    assert mod._detect_max_capture_B() == 512

    # k=5 is width 6, and the kernel also accepts widths up to 8.
    install(_pool_cfg(256, 8192, [1, 2, 512], 5, 6))
    assert mod._detect_max_capture_B() == 256 * 8

    # max_num_batched_tokens is a hard cap on rows in one step.
    install(_pool_cfg(256, 1000, [1, 2, 512], 5, 6))
    assert mod._detect_max_capture_B() == 1000

    # Wider than the kernel's verify range still has one row per token.
    install(_pool_cfg(256, 8192, [512], 16, 17))
    assert mod._detect_max_capture_B() == 256 * 17
