# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the e4m3 KV-cache bridge in the FlashAttention sparse MLA
backend (no GPU required).

The sparse varlen kernel only consumes bf16 rows, so under an e4m3 cache the
backend materializes the selected rows first: decode tokens gather their
top-k rows into a compact workspace indexed by a row-offset table, while each
prefill chunk upconverts its requests' resident context once and remaps its
top-k indices to workspace offsets (never per top-k row). These tests pin
that arithmetic with pure-torch references for the two kernels the path
calls: the request-index -> slot/offset conversion and the varlen attention.

The fused Triton gather-dequant runs for real via TRITON_INTERPRET (CPU
interpreter), including an exhaustive sweep of all 256 e4m3 byte values
against the unfused torch chain.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

# isort: off
import vllm.v1.attention.backends.mla.flashattn_mla_sparse as fa_sparse_mod  # noqa: E402
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (  # noqa: E402
    FlashAttnMLASparseBackend,
    FlashAttnMLASparseImpl,
    FlashAttnMLASparseMetadata,
    FlashAttnMLASparseMetadataBuilder,
    _gather_dequant_rows,
    _upconvert_chunk_context,
)
# isort: on

BLOCK_SIZE = 64
HEAD = 512
TOPK = 8


def make_cache(num_blocks, seed=0):
    vals = torch.randn(
        num_blocks * BLOCK_SIZE,
        HEAD,
        generator=torch.Generator().manual_seed(seed),
    )
    return vals.to(torch.float8_e4m3fn).view(num_blocks, BLOCK_SIZE, HEAD)


def dequant(cache, k_scale=1.0):
    rows = cache.reshape(-1, HEAD).to(torch.bfloat16)
    return rows if k_scale == 1.0 else rows * k_scale


def slots_for(positions, req_blocks):
    """Slot ids of req-relative token positions under a block table whose
    row ``req_blocks`` lists the request's physical blocks in order."""
    return [
        req_blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
        for pos in positions
    ]


def ref_convert(
    req_id,
    block_table,
    token_indices,
    BLOCK_SIZE=64,
    NUM_TOPK_TOKENS=2048,
    HAS_PREFILL_WORKSPACE=False,
    prefill_workspace_request_ids=None,
    prefill_workspace_starts=None,
    return_valid_counts=False,
    **_,
):
    """Pure-torch mirror of triton_convert_req_index_to_global_index: valid
    entries compacted to a row prefix, -1 elsewhere."""
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    out = torch.full_like(token_indices, -1)
    counts = torch.zeros(token_indices.shape[0], dtype=torch.int32)
    for t in range(token_indices.shape[0]):
        vals = []
        for j in range(token_indices.shape[1]):
            pos = int(token_indices[t, j])
            if pos == -1:
                continue
            if HAS_PREFILL_WORKSPACE:
                r = int(prefill_workspace_request_ids[t])
                if r != -1:
                    vals.append(int(prefill_workspace_starts[r]) + pos)
                    continue
            req = int(req_id[t])
            blk = int(block_table[req, pos // BLOCK_SIZE])
            if blk < 0:
                continue
            vals.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
        out[t, : len(vals)] = torch.tensor(vals, dtype=out.dtype)
        counts[t] = len(vals)
    if return_valid_counts:
        return out, counts
    return out


def ref_fa_varlen(
    *,
    q,
    k,
    v,
    q_v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    seqused_k,
    block_table,
    softmax_scale,
    causal=True,
    fa_version=3,
    only_qv=False,
    **_,
):
    """Mirror of the sparse varlen call: per token, attend to the first
    seqused_k rows gathered through block_table; scores from q_v @ V."""
    num_tokens, num_heads, _ = q_v.shape
    v_rows = v.reshape(-1, v.shape[-1])
    out = torch.empty(num_tokens, num_heads, v.shape[-1], dtype=v.dtype)
    for t in range(num_tokens):
        cnt = int(seqused_k[t])
        rows = v_rows[block_table[t, :cnt].to(torch.int64)].float()
        for h in range(num_heads):
            scores = (rows @ q_v[t, h].float()) * softmax_scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = (probs @ rows).to(v.dtype)
    return out


def ref_sparse_attention(q_nope, slots_per_token, v_rows):
    """Ground truth: softmax over each token's selected rows (bf16)."""
    num_tokens, num_heads, _ = q_nope.shape
    out = torch.empty(num_tokens, num_heads, HEAD, dtype=torch.bfloat16)
    rows = v_rows.float()
    for t, slots in enumerate(slots_per_token):
        sel = rows[torch.tensor(slots, dtype=torch.int64)]
        for h in range(num_heads):
            scores = (sel @ q_nope[t, h].float()) * (HEAD**-0.5)
            probs = torch.softmax(scores, dim=0)
            out[t, h] = (probs @ sel).to(torch.bfloat16)
    return out


def make_impl(kv_cache_dtype="fp8_e4m3", workspace_rows=4096, num_heads=2):
    impl = object.__new__(FlashAttnMLASparseImpl)
    impl.num_heads = num_heads
    impl.head_size = HEAD
    impl.scale = HEAD**-0.5
    impl.kv_lora_rank = HEAD
    impl.qk_rope_head_dim = 0
    impl.kv_cache_dtype = kv_cache_dtype
    impl.use_fp8_kv_cache = kv_cache_dtype in ("fp8", "fp8_e4m3")
    impl.prefill_bf16_workspace = torch.zeros(
        workspace_rows, HEAD, dtype=torch.bfloat16
    )
    return impl


def make_q(num_tokens, num_heads=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, num_heads, HEAD, generator=g).to(torch.bfloat16)


def make_block_table(rows_of_blocks):
    bt = torch.zeros(len(rows_of_blocks), 16, dtype=torch.int32)
    for r, blocks in enumerate(rows_of_blocks):
        bt[r, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return bt


def topk_tensor(rows):
    topk = torch.full((len(rows), TOPK), -1, dtype=torch.int32)
    for t, row in enumerate(rows):
        topk[t, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return topk


@pytest.fixture
def patched_kernels(monkeypatch):
    monkeypatch.setattr(fa_sparse_mod, "flash_attn_varlen_func", ref_fa_varlen)
    monkeypatch.setattr(
        fa_sparse_mod, "triton_convert_req_index_to_global_index", ref_convert
    )


def test_decode_bridge_matches_reference(patched_kernels):
    cache = make_cache(num_blocks=8, seed=3)
    block_table = make_block_table([[3], [5]])
    # token0 (req0, ctx 10): 8 selected positions; token1 (req1, ctx 5): only
    # 5 valid, the -1 tail must be masked rather than gathered.
    topk_rows = [[7, 3, 1, 9, 0, 2, 5, 8], [4, 0, 2, 3, 1, -1, -1, -1]]
    q_nope = make_q(2)

    impl = make_impl()
    impl.topk_indices_buffer = topk_tensor(topk_rows)
    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=block_table,
        block_size=BLOCK_SIZE,
        num_decode_tokens=2,
        fp8_prefill=None,
    )
    out, lse = impl.forward_mqa(
        (q_nope, q_nope.new_empty(2, impl.num_heads, 0)),
        cache,
        attn_metadata,
        SimpleNamespace(_k_scale_float=1.0),
    )
    assert lse is None

    expected_slots = [
        slots_for(topk_rows[0], [3]),
        slots_for([p for p in topk_rows[1] if p != -1], [5]),
    ]
    ref = ref_sparse_attention(q_nope, expected_slots, dequant(cache))
    torch.testing.assert_close(out, ref)


def test_decode_bridge_applies_k_scale(patched_kernels):
    cache = make_cache(num_blocks=4, seed=5)
    k_scale = 0.25
    topk_rows = [[2, 6, 4, 1, 0, 3, 5, 7]]
    q_nope = make_q(1, seed=7)

    impl = make_impl(kv_cache_dtype="fp8")
    impl.topk_indices_buffer = topk_tensor(topk_rows)
    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        block_table=make_block_table([[2]]),
        block_size=BLOCK_SIZE,
        num_decode_tokens=1,
        fp8_prefill=None,
    )
    out, _ = impl.forward_mqa(
        (q_nope, q_nope.new_empty(1, impl.num_heads, 0)),
        cache,
        attn_metadata,
        SimpleNamespace(_k_scale_float=k_scale),
    )

    expected_slots = [slots_for(topk_rows[0], [2])]
    ref = ref_sparse_attention(q_nope, expected_slots, dequant(cache, k_scale))
    torch.testing.assert_close(out, ref)


def test_prefill_chunks_match_reference(patched_kernels, monkeypatch):
    cache = make_cache(num_blocks=8, seed=11)
    # Snapshot each chunk's upconverted segment before the next chunk
    # reuses the workspace buffer.
    upconvert_snapshots = []
    real_upconvert = fa_sparse_mod._upconvert_chunk_context

    def snapshot_upconvert(cache_, bt_, lens_, max_len_, dst_, k_scale_):
        real_upconvert(cache_, bt_, lens_, max_len_, dst_, k_scale_)
        upconvert_snapshots.append(dst_.clone())

    monkeypatch.setattr(fa_sparse_mod, "_upconvert_chunk_context", snapshot_upconvert)

    # token 0: decode (req0, block 1, ctx 10). Prefill req1 (blocks [3, 4],
    # ctx 100 spanning two physical blocks, 4 query tokens) and req2 (blocks
    # [6, 7], ctx 80, 3 query tokens) are split into two chunks to exercise
    # per-chunk workspace rebasing and multi-block slot math.
    decode_topk = [9, 7, 5, 3, 1, 0, 2, 4]
    prefill_topk = [
        [99, 50, 77, 12, 0, 96, 30, 8],  # req1 token 0
        [98, 51, 10, 62, 5, 90, 41, 3],  # req1 token 1
        [97, 52, 20, 72, 6, 91, 42, 13],  # req1 token 2
        [96, 53, 30, 82, 7, 92, 43, 23],  # req1 token 3
        [79, 40, 66, 11, 0, 70, 25, 9],  # req2 token 0
        [78, 41, 12, 55, 4, 60, 31, 14],  # req2 token 1
        [77, 42, 22, 65, 5, 61, 32, 15],  # req2 token 2
    ]
    num_tokens = 8
    q_nope = make_q(num_tokens)
    block_table = make_block_table([[1], [3, 4], [6, 7]])

    impl = make_impl()
    impl.topk_indices_buffer = topk_tensor([decode_topk] + prefill_topk)
    # Full workspace starts, rebased per chunk: req1's chunk keeps [0], req2's
    # chunk is rebased from 100 to 0. Indexing by global prefill request id
    # (0 for req1, 1 for req2) yields each chunk's local offset.
    fp8_prefill = FlashAttnMLASparseMetadata.FP8Prefill(
        request_ids=torch.tensor([-1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32),
        workspace_starts=torch.tensor([0, 0], dtype=torch.int32),
        chunks=[
            FlashAttnMLASparseMetadata.FP8Prefill.Chunk(
                tokens_slice=slice(1, 5),
                block_table=block_table[1:2],
                seq_lens=torch.tensor([100], dtype=torch.int32),
                tot_seqlen=100,
                max_seq_len=100,
            ),
            FlashAttnMLASparseMetadata.FP8Prefill.Chunk(
                tokens_slice=slice(5, 8),
                block_table=block_table[2:3],
                seq_lens=torch.tensor([80], dtype=torch.int32),
                tot_seqlen=80,
                max_seq_len=80,
            ),
        ],
    )
    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1, 1, 1, 1, 2, 2, 2], dtype=torch.int32),
        block_table=block_table,
        block_size=BLOCK_SIZE,
        num_decode_tokens=1,
        fp8_prefill=fp8_prefill,
    )
    out, _ = impl.forward_mqa(
        (q_nope, q_nope.new_empty(num_tokens, impl.num_heads, 0)),
        cache,
        attn_metadata,
        SimpleNamespace(_k_scale_float=1.0),
    )

    # Workspace contents, snapshotted per chunk (chunks reuse the buffer):
    # each chunk's segment equals its request's dequantized [0, seq_len) rows,
    # in position order across blocks.
    rows = dequant(cache)
    assert [snap.shape[0] for snap in upconvert_snapshots] == [100, 80]
    torch.testing.assert_close(
        upconvert_snapshots[0], rows[slots_for(range(100), [3, 4])]
    )
    torch.testing.assert_close(
        upconvert_snapshots[1], rows[slots_for(range(80), [6, 7])]
    )

    expected_slots = [slots_for(decode_topk, [1])]
    expected_slots += [slots_for(row, [3, 4]) for row in prefill_topk[:4]]
    expected_slots += [slots_for(row, [6, 7]) for row in prefill_topk[4:]]
    ref = ref_sparse_attention(q_nope, expected_slots, rows)
    torch.testing.assert_close(out, ref)


def test_upconvert_chunk_context_ragged_requests():
    cache = make_cache(num_blocks=4, seed=17)
    block_table = make_block_table([[2], [3], [1, 2]])
    seq_lens = torch.tensor([5, 7, 70], dtype=torch.int32)

    dst = torch.full((5 + 7 + 70, HEAD), float("nan"), dtype=torch.bfloat16)
    _upconvert_chunk_context(
        cache, block_table, seq_lens, max_seq_len=70, dst=dst, k_scale=1.0
    )

    rows = dequant(cache)
    expected = torch.cat(
        [
            rows[slots_for(range(5), [2])],
            rows[slots_for(range(7), [3])],
            rows[slots_for(range(70), [1, 2])],
        ]
    )
    assert dst.shape[0] == expected.shape[0]
    torch.testing.assert_close(dst, expected)


def test_supported_kv_cache_dtypes():
    assert "fp8" in FlashAttnMLASparseBackend.supported_kv_cache_dtypes
    assert "fp8_e4m3" in FlashAttnMLASparseBackend.supported_kv_cache_dtypes


@pytest.mark.parametrize(
    "kv_dtype,ok",
    [
        ("auto", True),
        ("bfloat16", True),
        ("fp8", True),
        ("fp8_e4m3", True),
        ("fp8_e5m2", False),
        ("int8", False),
    ],
)
def test_supports_combination_fp8_gate(monkeypatch, kv_dtype, ok):
    monkeypatch.setattr(fa_sparse_mod, "flash_attn_supports_mla", lambda: True)
    reason = FlashAttnMLASparseBackend.supports_combination(
        head_size=HEAD,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        block_size=BLOCK_SIZE,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=SimpleNamespace(major=9),
    )
    assert (reason is None) == ok


def test_build_fp8_prefill_chunks_and_starts():
    builder = object.__new__(FlashAttnMLASparseMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.fp8_prefill_workspace_rows = 128

    seq_lens = torch.tensor([10, 100, 80, 30], dtype=torch.int32)
    qsl = torch.tensor([0, 1, 5, 8, 11], dtype=torch.int32)
    block_table = make_block_table([[0], [1], [2], [3]])
    common = SimpleNamespace(
        num_actual_tokens=11,
        seq_lens_cpu_upper_bound=seq_lens,
        query_start_loc_cpu=qsl,
        block_table_tensor=block_table,
    )
    metadata = SimpleNamespace(num_decodes=1, num_prefills=3)

    fp8_prefill = builder._build_fp8_prefill(common, metadata)

    assert fp8_prefill is not None
    assert fp8_prefill.request_ids.tolist() == [-1] + [0] * 4 + [1] * 3 + [2] * 3
    # 128 workspace rows: [100] and [80, 30] fit together -> two chunks.
    slices = [(c.tokens_slice.start, c.tokens_slice.stop) for c in fp8_prefill.chunks]
    assert slices == [(1, 5), (5, 11)]
    assert [c.tot_seqlen for c in fp8_prefill.chunks] == [100, 110]
    # Full starts rebased per chunk: [0, 100, 180] -> chunk 1 keeps its base,
    # chunk 2's entries shift by starts[1]=100.
    assert fp8_prefill.workspace_starts.tolist() == [0, 0, 80]
    assert fp8_prefill.chunks[1].seq_lens.tolist() == [80, 30]
    assert fp8_prefill.chunks[1].block_table.shape[0] == 2


# ---------------------------------------------------------------------------
# Fused Triton gather-dequant: bit-exactness vs the unfused torch chain.
# ---------------------------------------------------------------------------


def torch_gather_dequant(cache, slots, k_scale):
    """The unfused chain the kernel replaces: uint8 index_select -> fp8 view
    -> bf16 cast -> scalar mul (torch computes the mul in f32 opmath)."""
    head = cache.shape[-1]
    rows = (
        cache.view(torch.uint8).reshape(-1, head).index_select(0, slots.to(torch.int64))
    )
    rows = rows.view(torch.float8_e4m3fn).to(torch.bfloat16)
    return rows if k_scale == 1.0 else rows * k_scale


def assert_bit_equal(actual, expected):
    same = (actual == expected) | (actual.isnan() & expected.isnan())
    assert same.all(), (
        f"fused kernel diverges from torch chain at {int((~same).sum())} positions"
    )


def exhaustive_e4m3_cache():
    """One cache row per possible e4m3 byte, so gathering all 256 rows covers
    the entire fp8 input domain."""
    bytes_ = torch.arange(256, dtype=torch.uint8).unsqueeze(1).expand(256, HEAD)
    return bytes_.reshape(-1).view(torch.float8_e4m3fn).view(4, BLOCK_SIZE, HEAD)


@pytest.mark.parametrize("k_scale", [1.0, 0.25, 3.0, 1e-3])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_gather_dequant_bit_exact_all_e4m3_values(k_scale, idx_dtype):
    cache = exhaustive_e4m3_cache()
    slots = torch.arange(256, dtype=idx_dtype)

    out = _gather_dequant_rows(cache, slots, k_scale)
    ref = torch_gather_dequant(cache, slots, k_scale)

    assert out.dtype == torch.bfloat16
    assert_bit_equal(out, ref)
    # Semantic anchors (scale-aware, bf16 tolerance): +-0, max subnormal,
    # +-max finite, NaN.
    row = lambda b: out[int(b)]  # noqa: E731
    assert row(0x00).eq(0).all()
    assert row(0x80).eq(0).all()  # -0 compares equal to 0
    assert torch.isclose(
        row(0x07)[0].float(), torch.tensor(7 * 2**-9 * k_scale), rtol=1 / 64
    )
    assert torch.isclose(
        row(0x7E)[0].float(), torch.tensor(448.0 * k_scale), rtol=1 / 64
    )
    assert torch.isclose(
        row(0xFE)[0].float(), torch.tensor(-448.0 * k_scale), rtol=1 / 64
    )
    assert row(0x7F).isnan().all() and row(0xFF).isnan().all()


def test_gather_dequant_clamps_padding_and_duplicates():
    cache = make_cache(num_blocks=2, seed=23)
    slots = torch.tensor([5, -1, 5, 0, 257, -1], dtype=torch.int32)

    out = _gather_dequant_rows(cache, slots, 1.0)

    assert torch.equal(out[0], out[2])  # duplicate ids gather the same row
    clamped = slots.clamp(min=0).clamp(max=2 * BLOCK_SIZE - 1)
    ref = torch_gather_dequant(cache, clamped, 1.0)
    assert_bit_equal(out, ref)


def test_gather_dequant_empty_rows():
    cache = make_cache(num_blocks=1)
    out = _gather_dequant_rows(cache, torch.zeros(0, dtype=torch.int64), 1.0)
    assert out.shape == (0, HEAD)
