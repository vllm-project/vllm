# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    builder.kv_cache_spec = SimpleNamespace(tokens_per_state=1)

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


class _SparseRuntime:
    """Execute real CPU metadata/indexer/cache/MLA with deterministic inputs."""

    def __init__(self):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        self.index = SimpleNamespace(
            prefix="index", kv_cache=torch.zeros(8, 32, 132, dtype=torch.uint8)
        )
        self.tail = SimpleNamespace(
            prefix="tail", kv_cache=torch.zeros(8, 2, 4, 128, dtype=torch.bfloat16)
        )
        self.latent = torch.zeros(8, 128, 4)
        self.topk = torch.full((256, 7), -1, dtype=torch.int32)
        self.indexer = SparseAttnIndexerKpool(
            self.index,
            128,
            "ue8m0",
            4,
            128,
            1024,
            1024,
            self.topk,
            tail_cache=self.tail,
        )
        self.builder = Glm5NextCPUIndexerMetadataBuilder(
            MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=4,
            ),
            ["index"],
            None,
            torch.device("cpu"),
        )
        self.impl = object.__new__(Glm5NextCPUSparseImpl)
        self.impl.num_heads = 2
        self.impl.kv_lora_rank = 4
        self.impl.scale = 0.5
        self.impl.topk_indices_buffer = self.topk

    def run(self, requests):
        from vllm.forward_context import override_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.mla.indexer import (
            compute_kpool_tail_slot_mapping,
        )

        # Entries: (request data seed, first token, count, physical pages, tail page).
        positions: list[int] = []
        seeds: list[int] = []
        req_ids: list[int] = []
        slots: list[int] = []
        starts = [0]
        lengths: list[int] = []
        pages, tails = [], []
        for req, (seed, start, count, blocks, tail_block) in enumerate(requests):
            positions.extend(range(start, start + count))
            seeds.extend([seed] * count)
            req_ids.extend([req] * count)
            slots.extend(
                blocks[pos // 128] * 128 + pos % 128
                for pos in range(start, start + count)
            )
            starts.append(starts[-1] + count)
            lengths.append(start + count)
            pages.append(blocks)
            tails.append([tail_block])
        pos = torch.tensor(positions)
        seed = torch.tensor(seeds)
        starts = torch.tensor(starts, dtype=torch.int32)
        table = torch.tensor(pages, dtype=torch.int32)
        slots = torch.tensor(slots)
        common = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            num_reqs=len(requests),
            num_actual_tokens=len(positions),
            max_query_len=max(r[2] for r in requests),
            max_seq_len=max(lengths),
            block_table_tensor=table,
            slot_mapping=slots,
            positions=pos,
        )
        metadata = self.builder.build(0, common)
        tail_slots = compute_kpool_tail_slot_mapping(
            slots,
            torch.tensor(tails, dtype=torch.int32),
            starts,
            pos,
            len(positions),
            len(requests),
            4,
        )
        channels = torch.arange(128).float()
        keys = torch.sin(
            (pos[:, None] + seed[:, None] * 3 + channels) * 0.13
        ).bfloat16()
        gates = torch.cos((pos[:, None] + channels) * 0.17).bfloat16()
        ape = torch.arange(512).reshape(4, 128).float() * 0.001
        query = torch.cos(channels * 0.07)[None, None].expand(len(pos), 2, -1)
        context = ForwardContext(
            no_compile_layers={},
            slot_mapping={},
            attn_metadata={
                "index": metadata,
                "tail": SimpleNamespace(slot_mapping=tail_slots),
            },
        )
        with override_forward_context(context):
            self.indexer(
                torch.zeros(len(pos), 4),
                query,
                keys,
                torch.ones(len(pos), 2),
                gate_score=gates,
                compress_ape=ape,
                index_kpool=4,
                positions=pos,
            )
        values = torch.sin((pos[:, None] + seed[:, None] + torch.arange(4)) * 0.3)
        self.impl.do_kv_cache_update(
            values,
            torch.empty(len(pos), 1, 0),
            self.latent,
            slots,
            "auto",
            torch.ones(1),
        )
        q = torch.cos((pos[:, None, None] + torch.arange(8).reshape(1, 2, 4)) * 0.2)
        attn_meta = SimpleNamespace(
            req_id_per_token=torch.tensor(req_ids),
            block_size=128,
            block_table=table,
        )
        output, _ = self.impl.forward_mqa(q, self.latent, attn_meta, None)
        # Independent request-local dense attention over exactly the selected ids.
        for row, ids in enumerate(self.topk[: len(pos)]):
            chosen = ids[ids >= 0].long()
            assert (chosen <= pos[row]).all()
            assert chosen.unique().numel() == chosen.numel()
            tail_start = (int(pos[row]) + 1) // 4 * 4
            assert set(range(tail_start, int(pos[row]) + 1)) <= set(chosen.tolist())
            pool_count = (int(pos[row]) + 1) // 4
            if pool_count:
                history_pos = torch.arange(pool_count * 4)[:, None]
                history_keys = (
                    torch.sin((history_pos + seed[row] * 3 + channels) * 0.13)
                    .bfloat16()
                    .float()
                    .reshape(pool_count, 4, 128)
                )
                history_gates = (
                    torch.cos((history_pos + channels) * 0.17)
                    .bfloat16()
                    .float()
                    .reshape(pool_count, 4, 128)
                )
                pooled = (
                    (history_keys * (history_gates + ape).softmax(1)).sum(1).bfloat16()
                )
                transformed = _reference_fwht(pooled).bfloat16().float()
                scales = torch.exp2(
                    torch.ceil(
                        torch.log2(
                            transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4)
                            / 448
                        )
                    )
                )
                quantized = (
                    (transformed / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                logical = torch.arange(pool_count)
                physical = table[req_ids[row], logical // 32]
                records = self.index.kv_cache[physical, logical % 32]
                torch.testing.assert_close(
                    records[:, :128], quantized.view(torch.uint8)
                )
                torch.testing.assert_close(
                    records[:, 128:].contiguous().view(torch.float32), scales
                )
                scores = (
                    (query[row].float() @ (quantized.float() * scales).T).relu().sum(0)
                )
                selected_pool = int(chosen[0]) // 4
                torch.testing.assert_close(scores[selected_pool], scores.max())
            v = torch.sin((chosen[:, None] + seed[row] + torch.arange(4)) * 0.3)
            expected = (q[row] @ v.T * 0.5).softmax(-1) @ v
            torch.testing.assert_close(output[row], expected)
        return self.topk[: len(pos)].clone(), output.clone()


def test_mixed_prefill_decode_matches_isolated_requests():
    mixed, a, b = _SparseRuntime(), _SparseRuntime(), _SparseRuntime()
    mixed.run([(1, 0, 7, [5, 2], 6)])
    a.run([(1, 0, 7, [3, 4], 1)])
    ids, output = mixed.run([(1, 7, 1, [5, 2], 6), (9, 0, 11, [1, 7], 3)])
    ai, ao = a.run([(1, 7, 1, [3, 4], 1)])
    bi, bo = b.run([(9, 0, 11, [2, 5], 4)])
    torch.testing.assert_close(ids, torch.cat([ai, bi]))
    torch.testing.assert_close(output, torch.cat([ao, bo]))
    # Request ordering changes on the next step.
    ids, output = mixed.run([(9, 11, 2, [1, 7], 3), (1, 8, 1, [5, 2], 6)])
    bi, bo = b.run([(9, 11, 2, [2, 5], 4)])
    ai, ao = a.run([(1, 8, 1, [3, 4], 1)])
    torch.testing.assert_close(ids, torch.cat([bi, ai]))
    torch.testing.assert_close(output, torch.cat([bo, ao]))


def test_mixed_batch_pool_boundary_is_request_local():
    mixed, new_request, continuing = (
        _SparseRuntime(),
        _SparseRuntime(),
        _SparseRuntime(),
    )
    mixed.run([(9, 0, 3, [1, 7], 3)])
    continuing.run([(9, 0, 3, [2, 5], 4)])
    # Flattened positions [0, 1, 2, 3] look like one complete pool, but the
    # first three tokens and the final token belong to different requests.
    ids, output = mixed.run([(1, 0, 3, [5, 2], 6), (9, 3, 1, [1, 7], 3)])
    ni, no = new_request.run([(1, 0, 3, [3, 4], 1)])
    ci, co = continuing.run([(9, 3, 1, [2, 5], 4)])
    torch.testing.assert_close(ids, torch.cat([ni, ci]))
    torch.testing.assert_close(output, torch.cat([no, co]))


def test_slot_reuse_does_not_read_previous_request():
    reused, fresh = _SparseRuntime(), _SparseRuntime()
    reused.run([(11, 0, 135, [5, 2], 6)])
    for start, count in [(0, 3), (3, 1), (4, 5), (9, 122)]:
        actual = reused.run([(2, start, count, [5, 2], 6)])
        expected = fresh.run([(2, start, count, [5, 2], 6)])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)


@pytest.mark.parametrize("prefix", [7, 128, 131])
def test_prefix_restore_relocated_pages_matches_uninterrupted(prefix):
    live, restored = _SparseRuntime(), _SparseRuntime()
    live.run([(3, 0, prefix, [5, 2], 6)])
    # Restore all three state components to different physical owners.
    restored.index.kv_cache[[1, 7]] = live.index.kv_cache[[5, 2]].clone()
    restored.latent[[1, 7]] = live.latent[[5, 2]].clone()
    if prefix % 4:
        restored.tail.kv_cache[3] = live.tail.kv_cache[6].clone()
    else:
        # Prefix-cache hits are block-aligned; circular tail is not cached.
        restored.tail.kv_cache[3].fill_(99)
    for start, count in [(prefix, 1), (prefix + 1, 4)]:
        actual = restored.run([(3, start, count, [1, 7], 3)])
        expected = live.run([(3, start, count, [5, 2], 6)])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(
        restored.index.kv_cache[[1, 7]], live.index.kv_cache[[5, 2]]
    )


def test_indexer_score_applies_relu_before_head_weights():
    key = torch.ones(128)
    query = torch.stack([-torch.ones(128), torch.ones(128)])
    weights = torch.tensor([2.0, -3.0])
    assert _weighted_indexer_score(key, query, weights).item() == -384.0


@pytest.mark.parametrize("kernel_size", [None, 64])
def test_indexer_builder_compresses_slots_and_preserves_replay_mask(kernel_size):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    builder = Glm5NextCPUIndexerMetadataBuilder(
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=4,
        ),
        ["index"],
        None,
        torch.device("cpu"),
    )
    if kernel_size:
        builder.set_kernel_block_size(kernel_size)
    starts = torch.tensor([0, 2, 6], dtype=torch.int32)
    blocks = [[5, 2], [3, 7]]
    if kernel_size:
        blocks = [[10, 11, 4, 5], [6, 7, 14, 15]]
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([129, 4], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=4,
        max_seq_len=129,
        block_table_tensor=torch.tensor(blocks, dtype=torch.int32),
        slot_mapping=torch.tensor([767, 256, 384, 385, 386, -1]),
    )
    actual = builder.build(0, common)
    assert actual.slot_mapping.tolist() == [191, -1, -1, -1, -1, -1]
    assert actual.decode.seq_lens.tolist() == [32, 32, 0, 0, 0, 1]
    assert actual.decode.block_table.tolist() == [[5, 2]] * 2 + [[3, 7]] * 4


@pytest.mark.parametrize("sparse", [False, True])
def test_cpu_config_preserves_glm_sparse_chunking_and_prefix_cache(monkeypatch, sparse):
    import os
    from unittest.mock import patch

    from vllm.config import VllmConfig
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: False)
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)
    with patch.dict(os.environ):
        config = VllmConfig()
        config.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text",
                index_topk=4 if sparse else None,
                index_kpool=4,
            ),
            use_mla=True,
            max_model_len=256,
        )
        config.scheduler_config.enable_chunked_prefill = True
        config.cache_config.enable_prefix_caching = True
        CpuPlatform.check_and_update_config(config)
        assert config.scheduler_config.enable_chunked_prefill == sparse
        assert config.cache_config.enable_prefix_caching == sparse
