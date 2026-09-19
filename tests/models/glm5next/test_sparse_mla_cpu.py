# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.models.glm5next.cpu.mla import (
    Glm5NextCPUIndexerMetadataBuilder,
    Glm5NextCPUSparseImpl,
)
from vllm.models.glm5next.cpu.sparse_indexer import (
    _expand_pool_ids,
    _pool_compress,
    _quantize_cache_vector,
    _weighted_indexer_score,
    fwht128_quant_fp8,
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
            "block_table_tensor": torch.tensor(
                [[0, 1], [2, 3]], dtype=torch.int32
            ),
        },
    )()

    metadata = builder.build(0, common)
    assert metadata.num_decode_tokens == 5
    torch.testing.assert_close(
        metadata.decode.seq_lens,
        torch.tensor([4, 5, 6, 1, 2], dtype=torch.int32),
    )
    assert metadata.decode.block_table.shape == (5, 2)


@pytest.mark.parametrize(
    ("pool_ids", "seq_len", "expected"),
    [
        ([[0]], 3, [[0, 1, 2, -1, -1, -1, -1, -1]]),
        ([[0]], 5, [[0, 1, 2, 3, 4, -1, -1, -1]]),
    ],
)
def test_pool_to_token_expansion_appends_only_valid_tail(
    pool_ids, seq_len, expected
):
    ids = torch.tensor(pool_ids, dtype=torch.int32)
    actual = _expand_pool_ids(ids, torch.tensor([seq_len]), 3, max_tokens=8)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))
