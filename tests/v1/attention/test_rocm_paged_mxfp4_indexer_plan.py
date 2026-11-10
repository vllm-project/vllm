# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm MXFP4 indexer reads the paged cache in place, so each prefill
chunk is split back into its requests' rows, which one launch finds through
the chunk's query_start_loc. A chunk is whole requests or one query slice of a
long request, and both have to come back with the right rows and the right
block-table rows.
"""

import types

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import rocm_paged_mxfp4_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    DeepseekV41RocmMxfp4IndexerMetadataBuilder,
    native_decode,
    plan_gather_launches,
    plan_prefill_chunks,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    RocmPagedMxfp4CacheLayout,
    _fill_topk_rows,
    rocm_mxfp4_consumer_rows,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _chunk(block_table, token_start, token_end, row_ends):
    ends = torch.tensor(row_ends, dtype=torch.int32)
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=block_table,
        cu_seqlen_ks=torch.full_like(ends, 7),
        cu_seqlen_ke=ends + 7,
        cu_seq_lens=torch.zeros(1, dtype=torch.int32),
        token_to_seq=torch.zeros(1, dtype=torch.int32),
        total_seq_lens=1,
        max_seqlen_k=int(ends.max()),
        token_start=token_start,
        token_end=token_end,
        num_reqs=block_table.shape[0],
    )


# A decode (1 token), then prefills of 3, 3, 5 and 6 query tokens; the last
# one is sliced into two chunks.
QUERY_START_LOC = [0, 1, 4, 7, 12, 18]
SEQ_LENS = torch.tensor([40, 10, 12, 21, 64])
BLOCK_TABLE = torch.arange(20, dtype=torch.int32).view(5, 4)
CHUNKS = [
    _chunk(BLOCK_TABLE[1:4], 1, 12, list(range(11))),
    _chunk(BLOCK_TABLE[4:5], 12, 15, [29, 30, 30]),
    _chunk(BLOCK_TABLE[4:5], 15, 18, [31, 31, 32]),
]


def _plans(min_gather_width, block=8):
    return plan_prefill_chunks(
        CHUNKS,
        QUERY_START_LOC,
        SEQ_LENS,
        SEQ_LENS.int() // 2,
        2,
        block,
        min_gather_width,
        torch.tensor(QUERY_START_LOC, dtype=torch.int32),
    )


def test_chunks_split_into_requests():
    plans = _plans(20.0)

    assert [p.requests for p in plans] == [
        [(0, 3, 0), (3, 6, 1), (6, 11, 2)],
        [(0, 3, 0)],
        [(0, 3, 0)],
    ]
    # every chunk launches once, with offsets local to the chunk
    assert [p.query_start_loc.tolist() for p in plans] == [
        [0, 3, 6, 11],
        [0, 3],
        [0, 3],
    ]
    # the logits are as wide as the longest compressed context in the chunk
    assert [p.width for p in plans] == [10, 32, 32]
    assert [p.context_lens.tolist() for p in plans] == [[5, 6, 10], [32], [32]]
    torch.testing.assert_close(plans[2].row_ends, torch.tensor([31, 31, 32]).int())
    torch.testing.assert_close(plans[2].block_ends, torch.tensor([4, 4, 4]).int())
    assert [p.use_gather for p in plans] == [False, True, True]
    assert [p.first_request for p in plans] == [1, 4, 4]
    # no candidate blocks, as in the model-neutral builder
    for plan in _plans(None, block=0):
        assert plan.block_ends is None and not plan.use_gather


def test_launches_follow_the_logits_budget(monkeypatch):
    """Each launch's fp32 logits fit in VLLM_SPARSE_INDEXER_MAX_LOGITS_MB. A dense
    chunk's logits are [rows, longest context in the chunk]: K is read in place,
    not gathered, so requests share a launch until their rows no longer fit, and
    a request that does not fit alone is split on its query rows. A consumer's
    logits are [rows, pool]."""
    budget = 4 * 1000 * 30
    contexts, queries = torch.tensor([1000, 600, 200]), torch.tensor([10, 10, 10])
    assert split_prefill_chunks(contexts, queries, budget, 2) == [
        (slice(2, 5), slice(0, 30))
    ]
    # the third request would take the chunk past the budget
    queries = torch.tensor([10, 10, 11])
    assert split_prefill_chunks(contexts, queries, budget) == [
        (slice(0, 2), slice(0, 20)),
        (slice(2, 3), slice(0, 11)),
    ]
    assert split_prefill_chunks(torch.tensor([1000]), torch.tensor([100]), budget) == [
        (slice(0, 1), slice(lo, min(lo + 30, 100))) for lo in range(0, 100, 30)
    ]
    # the default 512 MiB holds 128 rows of a 1M context, or 8192 of a 16K pool
    assert split_prefill_chunks(
        torch.tensor([1 << 20]), torch.tensor([200]), 512 << 20
    ) == [(slice(0, 1), slice(0, 128)), (slice(0, 1), slice(128, 200))]
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    assert rocm_mxfp4_consumer_rows(16384) == 8192


def test_gather_launches_rejoin_a_requests_rows():
    """The consumers' logits are [rows, pool] whatever the context, so a
    request's rows go back together across the chunks that cut them, and a
    launch never mixes requests."""
    # the first chunk is too narrow to gather, the last request's slices rejoin
    (joined,) = plan_gather_launches(CHUNKS, _plans(20.0), 100)
    assert (joined.token_start, joined.token_end) == (12, 18)
    # every chunk gathers: the first chunk's three requests launch apart
    launches = plan_gather_launches(CHUNKS, _plans(0.0), 100)
    assert [(g.token_start, g.token_end) for g in launches] == [
        (1, 4),
        (4, 7),
        (7, 12),
        (12, 18),
    ]
    assert plan_gather_launches(CHUNKS, _plans(None), 100) == []


def test_decode_launches_native_on_uniform_steps():
    """A decode step goes to the kernel as next_n-row sequences only when its
    shape says so: each request has next_n rows, and cudagraph padding (query
    length 0) comes only after them. A FULL graph captured on a uniform batch
    then replays the same launch on a padded one."""
    rows = torch.zeros(8, dtype=torch.int32)
    row_block_table = torch.arange(32, dtype=torch.int32).view(8, 4)
    context_lens = torch.tensor([9, 4, 7, 0, 5], dtype=torch.int32)

    native = native_decode(rows, row_block_table, [2, 2, 2, 0], 2, context_lens)
    assert native is not None and native.next_n == 2
    assert native.context_lens.tolist() == [9, 4, 7, 0]
    # every request's first flattened row, as a view
    assert native.block_table.tolist() == row_block_table[::2].tolist()
    assert native.block_table.data_ptr() == row_block_table.data_ptr()

    assert native_decode(rows, row_block_table, [2, 0, 2, 2], 2, context_lens) is None
    # a ragged step, rows padded past the requests, and no speculation
    assert native_decode(rows[:5], row_block_table, [2, 1, 2], 2, context_lens) is None
    assert native_decode(rows, row_block_table, [2, 2, 2], 2, context_lens) is None
    assert native_decode(rows[:4], row_block_table, [1] * 4, 1, context_lens) is None


@pytest.mark.parametrize(
    "decode_rows, k, first_filled",
    [(7, 512, 7), (128, 512, 0), (7, 2048, 0), (None, 512, 0)],
)
def test_topk_fill_skips_the_rows_vllms_topk_writes(decode_rows, k, first_filled):
    num_tokens = 130
    buffer = torch.full((num_tokens + 1, k), 7, dtype=torch.int32)
    metadata = types.SimpleNamespace(decode=None, decode_row_lens=None)
    if decode_rows is not None:
        metadata.decode = object()
        metadata.decode_row_lens = torch.zeros(decode_rows, dtype=torch.int32)
    layer = types.SimpleNamespace(metadata=metadata, topk_buffer=buffer, topk_tokens=k)

    _fill_topk_rows(layer, num_tokens)
    assert bool((buffer[:first_filled] == 7).all())
    assert bool((buffer[first_filled:num_tokens] == -1).all())
    assert bool((buffer[num_tokens:] == 7).all())


class _TorchPathBuilder(DeepseekV41RocmMxfp4IndexerMetadataBuilder):
    fused_decode_metadata = False


def _step(device_lens, prefill_lens=(), host_lens=None, seq=None, rows=None):
    host_lens = device_lens if host_lens is None else host_lens
    gen = torch.Generator().manual_seed(sum(device_lens) * 31 + len(device_lens))
    if seq is None:
        seq = [
            q + int(torch.randint(0, 200000, (1,), generator=gen)) if q else 0
            for q in device_lens
        ]
    seq = list(seq) + [q + 1000 for q in prefill_lens]
    num_reqs = len(seq)

    def offsets(lens):
        return torch.tensor([0, *lens, *prefill_lens], dtype=torch.int32).cumsum(0)

    qsl_cpu = offsets(host_lens).int()
    num_tokens = int(qsl_cpu[-1]) + (rows - sum(device_lens) if rows else 0)
    seq_lens = torch.tensor(seq, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=offsets(device_lens).int().cuda(),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_lens.cuda(),
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=int(torch.diff(qsl_cpu).max()),
        max_seq_len=max(seq),
        block_table_tensor=torch.randint(
            0, 1 << 20, (num_reqs, 8192), generator=gen, dtype=torch.int32
        ).cuda(),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64).cuda(),
        causal=True,
    )


AV_LENS = [*range(1, 8), *range(1, 7), *[1] * 7]
UNIFORM_STEPS = [
    dict(device_lens=[7]),
    dict(device_lens=[7, 7, 7, 7], seq=[90, 4000, 0, 0]),
    dict(device_lens=[7, 0, 0], rows=21),
    dict(device_lens=[7, 7], prefill_lens=[40, 100]),
    dict(device_lens=[], prefill_lens=[20, 64]),
]
ADAPTIVE_STEPS = [
    dict(device_lens=AV_LENS, host_lens=[3] * 16 + [2] * 4, rows=64),
    dict(device_lens=AV_LENS[::-1], host_lens=[3] * 16 + [2] * 4, rows=64),
    dict(device_lens=[1, 7] * 24, host_lens=[4] * 48),
    dict(device_lens=[5, 0, 7, 2], host_lens=[4, 4, 4, 2], rows=20),
    dict(device_lens=[2, 5], prefill_lens=[30], host_lens=[4, 3]),
    dict(device_lens=[0, 1, 0, 0], host_lens=[1, 0, 0, 0], rows=2),
    dict(device_lens=[], prefill_lens=[20, 64]),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("adaptive", [False, True], ids=["uniform", "adaptive"])
def test_decode_metadata_matches_torch_path(monkeypatch, adaptive):
    """The one-launch decode metadata is the torch path's, padding rows aside."""
    module = rocm_paged_mxfp4_indexer
    monkeypatch.setattr(
        module,
        "rocm_paged_mxfp4_cache_layout",
        lambda *a: RocmPagedMxfp4CacheLayout(16, 16, 4),
    )
    monkeypatch.setattr(module, "rocm_mxfp4_decode_schedule_words", lambda *a: 4)
    monkeypatch.setattr(module, "build_rocm_mxfp4_decode_schedule", lambda *a: None)
    config = create_vllm_config(
        model_name="vllm-project/tiny-random-LlamaForCausalLM",
        max_model_len=8192,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        hf_config_override=dict(
            index_n_heads=32, index_head_dim=128, candidate_block_size=8
        ),
    )
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=6, enable_adaptive_verification=adaptive
    )
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=2,
    )
    device = torch.device("cuda", torch.accelerator.current_device_index())
    builders = [
        cls(spec, ["indexer"], config, device, block_table_width=8192)
        for cls in (_TorchPathBuilder, DeepseekV41RocmMxfp4IndexerMetadataBuilder)
    ]
    names = [
        "decode_seq_lens_buffer",
        "expanded_block_table_buffer",
        "decode_lens_buffer",
        "per_req_decode_lens_buffer",
        "context_lens_buffer",
        "decode_block_ends_buffer",
    ]
    ref_buf, out_buf = ([getattr(b, n) for n in names] for b in builders)
    for step in ADAPTIVE_STEPS if adaptive else UNIFORM_STEPS:
        cm = _step(**step)
        ref, out = (b.build(0, cm) for b in builders)
        assert torch.equal(out_buf[4][: cm.num_reqs], ref_buf[4][: cm.num_reqs])
        if ref.decode is None:
            assert out.decode is None
            continue
        assert out.decode is not None
        live = int(cm.query_start_loc_cpu[out.num_decodes])
        assert torch.equal(out_buf[0], ref_buf[0])
        assert torch.count_nonzero(out_buf[1][live : out.num_decode_tokens]) == 0
        for a, b, buf in (
            (out.decode.seq_lens, ref.decode.seq_lens, out_buf[0]),
            (out.decode.decode_lens, ref.decode.decode_lens, out_buf[2]),
            (
                out.decode.per_req_decode_lens,
                ref.decode.per_req_decode_lens,
                out_buf[3],
            ),
            (out.decode_block_ends, ref.decode_block_ends, out_buf[5]),
            (out.decode.block_table[:live], ref.decode.block_table[:live], out_buf[1]),
        ):
            assert a is not None and b is not None
            assert a.shape == b.shape and a.stride() == b.stride()
            assert torch.equal(a, b) and a.data_ptr() == buf.data_ptr()
        assert out.decode.block_table.shape == ref.decode.block_table.shape
        assert out.decode.requires_padding == ref.decode.requires_padding
