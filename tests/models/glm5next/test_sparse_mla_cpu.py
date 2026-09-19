# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.forward_context import ForwardContext
from vllm.models.glm5next.cpu.mla import (
    Glm5NextCPUIndexerMetadataBuilder,
    Glm5NextCPUSparseImpl,
)
from vllm.models.glm5next.cpu.sparse_indexer import (
    SparseAttnIndexerKpool,
    _dequantize_cache_vector,
    _expand_pool_ids,
    _pool_compress,
    _quantize_cache_vector,
    _weighted_indexer_score,
    fwht128_quant_fp8,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)


def _reference_fwht(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    width = 1
    while width < 128:
        grouped = y.reshape(-1, 128 // (2 * width), 2, width)
        a, b = grouped.unbind(dim=2)
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        width *= 2
    return y * (128.0**-0.5)


def test_cpu_fwht_quant_uses_reference_transform():
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    quant, scale = fwht128_quant_fp8(x)

    expected = _reference_fwht(x).to(torch.bfloat16).float()
    dequant = quant.float() * scale
    torch.testing.assert_close(dequant, expected, atol=1.5, rtol=0.08)
    assert quant.dtype == torch.float8_e4m3fn
    assert scale.shape == (3, 1)


def test_cpu_pool_compress_matches_independent_softmax_reference():
    keys = torch.randn(4, 128, dtype=torch.bfloat16)
    gate = torch.randn(4, 128)
    ape = torch.randn(4, 128)

    actual = _pool_compress(keys, gate, ape)
    probs = torch.softmax(gate + ape, dim=0)
    expected = (keys.float() * probs).sum(dim=0).to(keys.dtype)
    torch.testing.assert_close(actual, expected)


def test_cpu_cache_quantization_returns_glm_record():
    values, scale = _quantize_cache_vector(torch.randn(128))

    assert values.dtype == torch.uint8
    assert values.shape == (128,)
    assert scale.shape == ()
    assert torch.isfinite(scale)


def test_cpu_sparse_mqa_matches_selected_row_reference():
    impl = object.__new__(Glm5NextCPUSparseImpl)
    impl.num_heads = 2
    impl.v_head_dim = 3
    impl.kv_lora_rank = 4
    impl.qk_nope_head_dim = 2
    impl.qk_rope_head_dim = 0
    impl.scale = 0.5
    impl.topk_indices_buffer = torch.tensor([[0, 1, -1]], dtype=torch.int32)

    q = torch.randn(1, 2, 4)
    cache = torch.randn(1, 2, 4)
    metadata = type(
        "Metadata",
        (),
        {
            "req_id_per_token": torch.tensor([0]),
            "block_size": 2,
            "block_table": torch.tensor([[0]], dtype=torch.int32),
        },
    )()

    actual, lse = impl.forward_mqa(q, cache, metadata, None)
    logits = torch.einsum("nd,sd->ns", q[0], cache[0, :2]) * impl.scale
    expected = torch.einsum("hs,sd->hd", logits.softmax(-1), cache[0, :2])
    torch.testing.assert_close(actual[0], expected)
    assert lse is None


def test_indexer_score_flattens_per_head_weights():
    key = torch.ones(2, 128)
    query = torch.ones(2, 128)
    weights = torch.tensor([[2.0], [3.0]])

    actual = _weighted_indexer_score(key, query, weights)
    assert actual.item() == 5.0 * 128.0


def test_cpu_indexer_metadata_expands_requests_without_triton():
    builder = object.__new__(Glm5NextCPUIndexerMetadataBuilder)
    builder.device = torch.device("cpu")

    common = type(
        "CommonMetadata",
        (),
        {
            "num_actual_tokens": 5,
            "num_reqs": 2,
            "query_start_loc_cpu": torch.tensor([0, 3, 5], dtype=torch.int32),
            "query_start_loc": torch.tensor([0, 3, 5], dtype=torch.int32),
            "seq_lens": torch.tensor([6, 2], dtype=torch.int32),
            "max_seq_len": 6,
            "slot_mapping": torch.arange(5),
            "block_table_tensor": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        },
    )()

    metadata = builder.build(0, common)
    assert metadata.num_decode_tokens == 5
    torch.testing.assert_close(
        metadata.decode.seq_lens,
        torch.tensor([4, 5, 6, 1, 2], dtype=torch.int32),
    )
    assert metadata.decode.block_table.shape == (5, 2)


def test_cpu_indexer_forward_writes_pool_and_expands_decode_topk(monkeypatch):
    class Cache:
        prefix = "glm.indexer.k_cache"

        def __init__(self):
            self.kv_cache = torch.zeros(1, 1, 132, dtype=torch.uint8)

    cache = Cache()
    topk = torch.full((1, 8), -1, dtype=torch.int32)
    indexer = SparseAttnIndexerKpool(
        cache,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=8,
        head_dim=128,
        max_model_len=16,
        max_total_seq_len=16,
        topk_indices_buffer=topk,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            decode_lens=torch.tensor([1], dtype=torch.int32),
            requires_padding=False,
            schedule_metadata=torch.empty((0, 2), dtype=torch.int32),
        ),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={cache.prefix: metadata},
        slot_mapping={},
    )
    monkeypatch.setattr(
        "vllm.models.glm5next.cpu.sparse_indexer.get_forward_context",
        lambda context=context: context,
    )

    result = indexer(
        hidden_states=torch.zeros(1, 4),
        q_quant=torch.zeros(1, 1, 128, dtype=torch.float8_e4m3fn),
        k=torch.ones(1, 128, dtype=torch.bfloat16),
        weights=torch.ones(1, 1),
        gate_score=torch.zeros(1, 128),
        compress_ape=torch.zeros(1, 128),
        index_kpool=1,
        positions=torch.tensor([0]),
    )

    assert result.data_ptr() == topk.data_ptr()
    assert result[0, 0].item() == 0
    assert torch.any(cache.kv_cache != 0)


@pytest.mark.parametrize("steps", [(7,), (3, 1, 3), (1, 1, 1, 1, 1, 1, 1)])
def test_forward_preserves_pool_and_tail_across_steps(monkeypatch, steps):
    cache = SimpleNamespace(
        prefix="index", kv_cache=torch.zeros(3, 32, 132, dtype=torch.uint8)
    )
    tail = SimpleNamespace(
        prefix="tail", kv_cache=torch.zeros(3, 2, 4, 128, dtype=torch.bfloat16)
    )
    output = torch.full((7, 7), -1, dtype=torch.int32)
    op = SparseAttnIndexerKpool(
        cache, 128, "ue8m0", 4, 128, 128, 128, output, tail_cache=tail
    )
    keys = torch.arange(1, 8).float()[:, None].expand(7, 128).bfloat16()
    start = 0
    for count in steps:
        positions = torch.arange(start, start + count)
        lengths = (positions + 1) // 4
        slots = torch.where((positions + 1) % 4 == 0, 64 + positions // 4, -1)
        metadata = DeepseekV32IndexerMetadata(
            seq_lens=lengths[-1:],
            max_seq_len=int(lengths[-1]),
            slot_mapping=slots,
            num_decodes=1,
            num_decode_tokens=count,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=DeepSeekV32IndexerDecodeMetadata(
                block_table=torch.full((count, 1), 2, dtype=torch.int32),
                seq_lens=lengths,
                decode_lens=torch.ones(count, dtype=torch.int32),
                requires_padding=False,
                schedule_metadata=torch.empty(0, 2, dtype=torch.int32),
            ),
        )
        context = ForwardContext(
            no_compile_layers={},
            slot_mapping={},
            attn_metadata={
                "index": metadata,
                "tail": SimpleNamespace(slot_mapping=4 + positions % 4),
            },
        )
        monkeypatch.setattr(
            "vllm.models.glm5next.cpu.sparse_indexer.get_forward_context",
            lambda context=context: context,
        )
        result = op(
            torch.zeros(count, 4),
            torch.zeros(count, 1, 128),
            keys[start : start + count],
            torch.ones(count, 1),
            gate_score=torch.zeros(count, 128),
            compress_ape=torch.zeros(4, 128),
            index_kpool=4,
            positions=positions,
        )
        for row, position in enumerate(positions.tolist()):
            expected_ids = list(range(position + 1)) + [-1] * (6 - position)
            assert result[row].tolist() == expected_ids
        start += count
    expected = _reference_fwht(keys[:4].float().mean(0)[None])[0]
    torch.testing.assert_close(
        _dequantize_cache_vector(cache.kv_cache[2, 0]),
        expected,
        atol=0.3,
        rtol=0.08,
    )
    assert not cache.kv_cache[:2].any()
    assert not tail.kv_cache[0].any()


def test_cpu_indexer_completes_pool_from_tail_and_expands_tokens():
    class Cache:
        prefix = "glm.indexer.k_cache"

        def __init__(self):
            self.kv_cache = torch.zeros(1, 1, 132, dtype=torch.uint8)

    class Tail:
        prefix = "glm.indexer.tail_cache"

        def __init__(self):
            self.kv_cache = torch.zeros(1, 2, 4, 128, dtype=torch.bfloat16)

    cache = Cache()
    tail = Tail()
    indexer = SparseAttnIndexerKpool(
        cache,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=8,
        head_dim=128,
        max_model_len=16,
        max_total_seq_len=16,
        topk_indices_buffer=torch.full((1, 8), -1, dtype=torch.int32),
        tail_cache=tail,
    )
    indexer.index_kpool = 4
    positions = torch.arange(4, dtype=torch.int32)
    keys = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 128)
    gates = torch.zeros(4, 128)
    metadata = type(
        "Metadata", (), {"slot_mapping": torch.zeros(4, dtype=torch.int64)}
    )()
    tail_metadata = type(
        "TailMetadata", (), {"slot_mapping": torch.arange(4, dtype=torch.int64)}
    )()

    indexer._write_pools(
        keys.to(torch.bfloat16),
        gates,
        torch.zeros(4, 128),
        metadata.slot_mapping,
        4,
        positions,
        {cache.prefix: metadata, tail.prefix: tail_metadata},
    )

    expected = _reference_fwht(keys.mean(dim=0).reshape(1, 128))[0]
    torch.testing.assert_close(
        _dequantize_cache_vector(cache.kv_cache[0, 0]),
        expected,
        atol=0.2,
        rtol=0.08,
    )
    torch.testing.assert_close(tail.kv_cache[0, 0, :, 0], keys[:, 0].to(torch.bfloat16))

    decode = type(
        "Decode",
        (),
        {
            "requires_padding": False,
            "seq_lens": torch.tensor([1], dtype=torch.int32),
            "block_table": torch.tensor([[0]], dtype=torch.int32),
        },
    )()
    metadata.decode = decode
    metadata.num_decodes = 1
    metadata.num_decode_tokens = 1
    indexer._decode_topk(
        torch.zeros(1, 1, 128, dtype=torch.float8_e4m3fn),
        torch.ones(1, 1),
        type("IndexerMetadata", (), {"decode": decode})(),
        4,
        torch.tensor([3]),
    )
    torch.testing.assert_close(
        indexer.topk_indices_buffer[0, :4],
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("pool_ids", "seq_len", "expected"),
    [
        ([[0]], 3, [[0, 1, 2, -1, -1, -1, -1, -1]]),
        ([[0]], 5, [[0, 1, 2, 3, 4, -1, -1, -1]]),
    ],
)
def test_pool_to_token_expansion_appends_only_valid_tail(pool_ids, seq_len, expected):
    ids = torch.tensor(pool_ids, dtype=torch.int32)
    actual = _expand_pool_ids(ids, torch.tensor([seq_len]), 3, max_tokens=8)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))
