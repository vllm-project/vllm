# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 sparse prefill attention kernels."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _FP8_DTYPES,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseTritonImpl,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip(
        "MiniMax M3 attention kernels require CUDA or ROCm.",
        allow_module_level=True,
    )


@pytest.fixture
def kv_layout(request):
    """Set the global KV cache layout for one test and restore it after."""
    set_kv_cache_layout(request.param)
    try:
        yield request.param
    finally:
        set_kv_cache_layout(None)


def _stride_order_for(backend: type[MiniMaxM3SparseBackend], ndim: int) -> tuple:
    """Mirror the allocator's stride-order resolution (identity fallback)."""
    try:
        stride_order = backend.get_kv_cache_stride_order()
        assert len(stride_order) == ndim
    except (AttributeError, NotImplementedError):
        stride_order = tuple(range(ndim))
    return stride_order


def _allocate_main_kv_via_contract(
    num_pages: int, device: torch.device | str = "cuda"
) -> torch.Tensor:
    """Build the main KV cache exactly as the production allocator does for the
    currently active layout: allocate the physical (permuted) tensor, then
    expose the inverse-permuted logical-NHD view the backend sees."""
    logical_shape = MiniMaxM3SparseBackend.get_kv_cache_shape(
        num_pages, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    )
    stride_order = _stride_order_for(MiniMaxM3SparseBackend, len(logical_shape))
    physical_shape = tuple(logical_shape[i] for i in stride_order)
    inv_order = [stride_order.index(i) for i in range(len(stride_order))]
    raw = torch.randn(physical_shape, device=device, dtype=DTYPE)
    return raw.permute(*inv_order)


NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SM_SCALE = HEAD_DIM**-0.5
TOPK = 16


@pytest.mark.parametrize(
    ("kv_cache_dtype", "expected_dtype"),
    [
        ("fp8", current_platform.fp8_dtype()),
        ("fp8_e4m3", current_platform.fp8_dtype()),
        (
            "fp8_e5m2",
            torch.float8_e5m2fnuz
            if current_platform.is_fp8_fnuz()
            else torch.float8_e5m2,
        ),
    ],
)
def test_sparse_impl_uses_platform_fp8_dtype(
    kv_cache_dtype: str,
    expected_dtype: torch.dtype,
):
    impl = MiniMaxM3SparseTritonImpl(
        num_heads=NUM_Q_HEADS,
        head_size=HEAD_DIM,
        scale=SM_SCALE,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype=kv_cache_dtype,
        topk_blocks=TOPK,
        sparse_block_size=BLOCK_SIZE,
    )
    assert impl.kv_cache_fp8_dtype == expected_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ],
)
def test_sparse_kernels_recognize_fp8_dtypes(dtype: torch.dtype):
    assert dtype in _FP8_DTYPES


# Index top-k kernels.
def _reference_index_topk(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, topk), -1, device=idx_q.device, dtype=torch.int32
    )

    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q = idx_q[q_start:q_end]
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[req_id, :num_blocks]
        k = index_kv_cache[pages].reshape(num_blocks * BLOCK_SIZE, -1)
        score = sm_scale * torch.einsum("qhd,kd->hqk", q.float(), k.float())

        q_pos = prefix_len + torch.arange(q_len, device=idx_q.device)
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        score = score.reshape(num_idx_heads, q_len, num_blocks, BLOCK_SIZE)
        score_tensor = score.max(dim=3).values

        valid_blocks = (q_pos + BLOCK_SIZE) // BLOCK_SIZE
        for local_q, num_valid_blocks in enumerate(valid_blocks.tolist()):
            end = min(init_blocks, num_valid_blocks)
            score_tensor[:, local_q, :end] = 1e30
            start = max(0, num_valid_blocks - local_blocks)
            score_tensor[:, local_q, start:num_valid_blocks] = 1e29

            k = min(topk, num_valid_blocks)
            topk_idx = score_tensor[:, local_q].topk(k, dim=1).indices
            out[:, q_start + local_q, :k] = topk_idx
        q_start = q_end

    return out


def _assert_topk_indices_equal_unordered(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare selected sparse blocks without requiring a deterministic order."""
    assert actual.shape == expected.shape
    actual_flat = actual.cpu().reshape(-1, actual.shape[-1]).tolist()
    expected_flat = expected.cpu().reshape(-1, expected.shape[-1]).tolist()
    for actual_row, expected_row in zip(actual_flat, expected_flat):
        assert set(actual_row) == set(expected_row)


def _reference_decode_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    score_block_stride: int,
) -> torch.Tensor:
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, score_block_stride),
        -float("inf"),
        device=idx_q.device,
        dtype=torch.float32,
    )
    for req_id, seq_len in enumerate(seq_lens.tolist()):
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        token_start = req_id * decode_query_len
        q = idx_q[token_start : token_start + decode_query_len].float()
        pages = block_table[req_id, :num_blocks]
        k = index_kv_cache[pages].reshape(num_blocks * BLOCK_SIZE, -1).float()
        score = torch.einsum("qhd,kd->hqk", q, k)
        q_pos = (
            seq_len
            - decode_query_len
            + torch.arange(decode_query_len, device=idx_q.device)
        )
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        out[:, token_start : token_start + decode_query_len, :num_blocks] = (
            score.reshape(num_idx_heads, decode_query_len, num_blocks, BLOCK_SIZE)
            .max(dim=3)
            .values
        )
    return out


def test_prefill_index_topk_correctness():
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    q_lens = torch.tensor((4, 3), device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor((0, 1024), device="cuda", dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_seq_len = seq_lens.max().item()
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.ones(q_lens.sum().item(), num_idx_heads, head_dim, device="cuda")
    index_kv_cache = torch.empty(num_pages, BLOCK_SIZE, head_dim, device="cuda")
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = block_table[req_id, block_id]
            index_kv_cache[page].fill_(block_id + 1)

    score = minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=q_lens.max().item(),
        max_seq_len=max_seq_len,
        num_kv_heads=num_idx_heads,
    )
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix_lens,
        max_query_len=q_lens.max().item(),
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
    )
    _assert_topk_indices_equal_unordered(actual, expected)


# MSA indexer (SM100): fmha_sm100 OnlyScore for the per-block scores, then the
# Triton minimax_m3_index_topk for selection (no sparse_topk_select). Uses a
# deterministic construction (idx_q == 1, distinct e4m3-exact per-block values)
# so scores are strictly monotonic in the block id -> exact top-k agreement.
def _fmha_indexer_topk(
    idx_q: torch.Tensor,  # [total_q, H, 128] bf16/e4m3
    index_cache: torch.Tensor,  # [num_pages, 128, 128] bf16/e4m3
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    sm_scale: float,
    topk: int,
) -> torch.Tensor:
    """Replicate MiniMaxM3IndexerMSAImpl's score path (single decode/prefill side)."""
    from vllm.third_party.fmha_sm100.api import _fmha_sm100, _fmha_sm100_plan

    num_idx_heads, head_dim = idx_q.shape[1], idx_q.shape[2]
    nvp = [(s + 127) // 128 for s in seq_lens.tolist()]
    kv_indices = torch.cat([block_table[r, : nvp[r]] for r in range(len(nvp))]).to(
        torch.int32
    )

    qo = q_lens.cpu().to(torch.int32)
    kv = seq_lens.cpu().to(torch.int32)
    plan = _fmha_sm100_plan(
        qo,
        kv,
        num_idx_heads,
        num_kv_heads=1,
        qo_offset=kv - qo,
        page_size=128,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
    )
    k_pages = index_cache.view(index_cache.shape[0], 1, 128, head_dim)
    _, max_score = _fmha_sm100(
        idx_q,
        k_pages,
        k_pages,
        plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=sm_scale,
    )

    batch = q_lens.numel()
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=idx_q.device)
    cu[1:] = q_lens.to(torch.int32).cumsum(0)
    # max_score [H, k_tiles, total_q] -> transpose to [H, total_q, k_tiles].
    return minimax_m3_index_topk(
        max_score.transpose(1, 2),
        cu,
        prefix_lens.to(torch.int32),
        int(q_lens.max()),
        topk,
        0,  # init_blocks
        0,  # local_blocks
    )


# e4m3-exact, strictly-increasing per-block values: with idx_q == 1 (also exact)
# the per-block scores are exact and distinct in BOTH bf16 and e4m3, so the fp8
# score path selects the same top-k as the reference (no quantization ties).
_E4M3_EXACT_VALUES = [
    *range(1, 17),  # 1..16  (step 1)
    *range(18, 33, 2),  # 18..32 (step 2)
    *range(36, 65, 4),  # 36..64 (step 4)
    *range(72, 129, 8),  # 72..128 (step 8)
]


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fmha_sm100 indexer requires SM100 (Blackwell).",
)
@pytest.mark.parametrize("index_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("q_lens", "prefix_lens"),
    [
        ((4, 3), (2048, 2560)),  # prefill: every token sees >= 16 causal blocks
        ((1, 1, 1), (2048, 3000, 4096)),  # decode: one query token per request
    ],
)
def test_fmha_sm100_indexer_matches_reference(q_lens, prefix_lens, index_dtype):
    torch.manual_seed(0)
    num_idx_heads, head_dim = 4, HEAD_DIM
    device = "cuda"

    q_lens_t = torch.tensor(q_lens, device=device, dtype=torch.int32)
    prefix_lens_t = torch.tensor(prefix_lens, device=device, dtype=torch.int32)
    seq_lens = prefix_lens_t + q_lens_t
    batch = len(q_lens)
    max_blocks = (int(seq_lens.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert max_blocks <= len(_E4M3_EXACT_VALUES)
    num_pages = batch * max_blocks
    block_table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, max_blocks
    )

    idx_q = torch.ones(
        int(q_lens_t.sum()), num_idx_heads, head_dim, device=device, dtype=index_dtype
    )
    index_cache = torch.empty(
        num_pages, BLOCK_SIZE, head_dim, device=device, dtype=index_dtype
    )
    for r in range(batch):
        for b in range(max_blocks):
            index_cache[block_table[r, b]] = float(_E4M3_EXACT_VALUES[b])

    sm_scale = head_dim**-0.5
    actual = _fmha_indexer_topk(
        idx_q,
        index_cache,
        block_table,
        q_lens_t,
        seq_lens,
        prefix_lens_t,
        sm_scale,
        TOPK,
    )
    expected = _reference_index_topk(
        idx_q,
        index_cache,
        block_table,
        q_lens_t,
        seq_lens,
        prefix_lens_t,
        TOPK,
        init_blocks=0,
        local_blocks=0,
        sm_scale=sm_scale,
    )
    _assert_topk_indices_equal_unordered(actual, expected)


# Full impl-level parity: drive both MiniMaxM3IndexerMSAImpl (fmha/CuteDSL score
# + unified top-k) and MiniMaxM3IndexerTritonImpl through their real metadata
# builders on the SAME CommonAttentionMetadata + index cache, and assert the
# selected blocks agree. This exercises all the metadata the impl/kernels consume
# (decode/prefill split, cu_seqlens_q rebasing, prefix_lens, kv_indices gather,
# decode_pages split) -- a metadata bug on either side shifts the causal window
# or the block->page mapping and breaks the comparison.
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fmha_sm100 indexer requires SM100 (Blackwell).",
)
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize("index_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_msa_indexer_impl_matches_triton(topk, index_dtype, monkeypatch):
    import vllm.models.minimax_m3.common.indexer as indexer_mod
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.common.indexer import (
        MiniMaxM3IndexerTritonImpl,
        MiniMaxM3IndexerTritonMetadataBuilder,
    )
    from vllm.models.minimax_m3.nvidia.indexer_msa import (
        MiniMaxM3IndexerMSAImpl,
        MiniMaxM3IndexerMSAMetadataBuilder,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_idx_heads, head_dim = 4, HEAD_DIM
    # TP=1: avoid requiring an initialized distributed group in a unit test.
    monkeypatch.setattr(indexer_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE, max_model_len=8192, max_num_batched_tokens=8192
    )
    vllm_config.model_config.hf_config.sparse_attention_config = {
        "sparse_num_index_heads": num_idx_heads
    }

    # Decode-first mixed batch: 2 decode reqs (q_len 1) then 2 prefill reqs. Long
    # prefixes so every token sees > TOPK causal blocks (non-trivial selection).
    batch = BatchSpec(seq_lens=[2305, 2561, 2624, 2720], query_lens=[1, 1, 64, 96])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    num_tokens = batch.compute_num_tokens()

    # Deterministic index cache: distinct, monotonic per-logical-block values so
    # the top-k is unambiguous (both kernels pick the same blocks, no fp ties).
    block_table = common.block_table_tensor
    num_pages = int(block_table.max().item()) + 1
    index_cache = torch.zeros(
        num_pages, BLOCK_SIZE, head_dim, device=device, dtype=index_dtype
    )
    for r, seq_len in enumerate(batch.seq_lens):
        for b in range((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE):
            index_cache[block_table[r, b]] = float(b + 1)
    index_q = torch.ones(
        num_tokens, num_idx_heads * head_dim, device=device, dtype=index_dtype
    )

    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=head_dim, dtype=DTYPE
    )
    impl_kwargs = dict(
        num_kv_heads=num_idx_heads,
        scale=head_dim**-0.5,
        topk_blocks=topk,
        sparse_block_size=BLOCK_SIZE,
        num_index_heads=num_idx_heads,
        index_head_dim=head_dim,
        init_blocks=0,
        local_blocks=0,
    )

    with set_current_vllm_config(vllm_config):
        msa_impl = MiniMaxM3IndexerMSAImpl(prefix="idx_msa", **impl_kwargs)
        triton_impl = MiniMaxM3IndexerTritonImpl(prefix="idx_triton", **impl_kwargs)
        msa_builder = MiniMaxM3IndexerMSAMetadataBuilder(
            spec, [msa_impl.index_cache.prefix], vllm_config, device
        )
        triton_builder = MiniMaxM3IndexerTritonMetadataBuilder(
            spec, [triton_impl.index_cache.prefix], vllm_config, device
        )

    # Both impls score against the same index keys.
    msa_impl.index_cache.kv_cache = index_cache
    triton_impl.index_cache.kv_cache = index_cache

    # Exercise the shared persistent top-k buffer for BOTH impls: each must write
    # decode ([:, :nd]) and prefill ([:, nd:]) into its buffer and return views.
    # Separate buffers so the two forwards don't clobber each other.
    nd = sum(q for q in batch.query_lens if q <= 1)
    msa_impl.topk_indices_buffer = torch.full(
        (num_idx_heads, num_tokens, topk), -2, dtype=torch.int32, device=device
    )
    triton_impl.topk_indices_buffer = torch.full(
        (num_idx_heads, num_tokens, topk), -2, dtype=torch.int32, device=device
    )

    attn_metadata = {
        msa_impl.index_cache.prefix: msa_builder.build(0, common),
        triton_impl.index_cache.prefix: triton_builder.build(0, common),
    }
    with set_forward_context(attn_metadata, vllm_config):
        msa_decode, msa_prefill = msa_impl(index_q)
        tri_decode, tri_prefill = triton_impl(index_q)

    assert msa_decode is not None and tri_decode is not None
    assert msa_prefill is not None and tri_prefill is not None
    _assert_topk_indices_equal_unordered(msa_decode, tri_decode)
    _assert_topk_indices_equal_unordered(msa_prefill, tri_prefill)
    # decode/prefill outputs are views into each impl's persistent buffer.
    for impl, dec, pre in (
        (msa_impl, msa_decode, msa_prefill),
        (triton_impl, tri_decode, tri_prefill),
    ):
        buf = impl.topk_indices_buffer
        assert dec.data_ptr() == buf[:, :nd, :].data_ptr()
        assert pre.data_ptr() == buf[:, nd:, :].data_ptr()


@pytest.mark.parametrize(
    ("decode_query_len", "max_decode_query_len"),
    [
        (1, 1),
        (1, 4),
        (4, 4),
    ],
)
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
def test_decode_index_topk_correctness(
    decode_query_len: int,
    max_decode_query_len: int,
    num_padded_reqs: int,
):
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    active_seq_lens = torch.tensor((7, 129, 1025), device="cuda", dtype=torch.int32)
    q_lens = torch.full_like(active_seq_lens, decode_query_len)
    prefix_lens = active_seq_lens - decode_query_len
    active_batch = active_seq_lens.numel()
    batch = active_batch + num_padded_reqs
    seq_lens = torch.cat(
        [
            active_seq_lens,
            torch.zeros(num_padded_reqs, device="cuda", dtype=torch.int32),
        ]
    )
    max_seq_len = active_seq_lens.max().item()
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = active_batch * max_blocks

    active_block_table = torch.randperm(
        num_pages, device="cuda", dtype=torch.int32
    ).reshape(active_batch, max_blocks)
    block_table = torch.zeros(batch, max_blocks, device="cuda", dtype=torch.int32)
    block_table[:active_batch] = active_block_table
    idx_q = torch.randn(
        batch * decode_query_len, num_idx_heads, head_dim, device="cuda"
    )
    index_kv_cache = torch.randn(num_pages, BLOCK_SIZE, head_dim, device="cuda")

    actual = minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        decode_query_len=decode_query_len,
        max_decode_query_len=max_decode_query_len,
    )
    expected = torch.full_like(actual, -1)
    active_tokens = active_batch * decode_query_len
    expected[:, :active_tokens] = _reference_index_topk(
        idx_q[:active_tokens],
        index_kv_cache,
        block_table[:active_batch],
        q_lens,
        active_seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
    )
    _assert_topk_indices_equal_unordered(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fp8 e4m3 indexer cache is the SM100 (MSA) path.",
)
@pytest.mark.parametrize("num_idx_heads", [1, 4])
def test_decode_index_topk_fp8(num_idx_heads: int):
    """The standalone Triton path must score FP8 inputs in FP32 so its top-k
    matches a reference computed from the dequantized FP8 values."""
    torch.manual_seed(0)
    topk, init_blocks, local_blocks, head_dim = 8, 0, 1, 128
    decode_query_len = 1
    active_seq_lens = torch.tensor((129, 1025, 4097), device="cuda", dtype=torch.int32)
    q_lens = torch.full_like(active_seq_lens, decode_query_len)
    prefix_lens = active_seq_lens - decode_query_len
    batch = active_seq_lens.numel()
    max_seq_len = int(active_seq_lens.max())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks
    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.randn(
        batch * decode_query_len, num_idx_heads, head_dim, device="cuda"
    ).to(torch.float8_e4m3fn)
    index_kv_cache = torch.randn(num_pages, BLOCK_SIZE, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )

    actual = minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        active_seq_lens,
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        decode_query_len=decode_query_len,
        max_decode_query_len=decode_query_len,
    )
    # Reference from the DEQUANTIZED fp8 values (the kernel computes the fp8 QK
    # in fp32 with no scaling, so it must match an unscaled fp32 matmul of the
    # same e4m3 values).
    expected = _reference_index_topk(
        idx_q.float(),
        index_kv_cache.float(),
        block_table,
        q_lens,
        active_seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
    )
    _assert_topk_indices_equal_unordered(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="CuteDSL index decode score requires Blackwell.",
)
@pytest.mark.parametrize(
    ("dtype", "decode_query_len", "max_decode_query_len"),
    [
        (torch.bfloat16, 1, 1),
        (torch.bfloat16, 3, 8),
        (torch.float8_e4m3fn, 1, 1),
        (torch.float8_e4m3fn, 3, 8),
        (torch.float8_e4m3fn, 8, 8),
    ],
)
def test_decode_index_score_cutedsl_correctness(
    dtype: torch.dtype,
    decode_query_len: int,
    max_decode_query_len: int,
):
    pytest.importorskip("cutlass")
    from vllm.models.minimax_m3.nvidia.ops import (
        minimax_m3_index_decode_score_cutedsl,
    )

    torch.manual_seed(0)
    init_blocks, local_blocks = 0, 0
    num_idx_heads, head_dim = 4, 128
    active_seq_lens = torch.tensor((1025, 4097), device="cuda", dtype=torch.int32)
    batch = active_seq_lens.numel()
    total_q = batch * decode_query_len
    max_seq_len = int(active_seq_lens.max())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    score_block_stride = ((max_blocks + 15) // 16) * 16
    num_pages = batch * max_blocks
    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.randn(total_q, num_idx_heads, head_dim, device="cuda").to(dtype)
    index_kv_cache = torch.randn(num_pages, BLOCK_SIZE, head_dim, device="cuda").to(
        dtype
    )
    unified_score = torch.full(
        (total_q, num_idx_heads, score_block_stride),
        -float("inf"),
        device="cuda",
        dtype=torch.float32,
    )
    score = unified_score.transpose(0, 1)

    minimax_m3_index_decode_score_cutedsl(
        idx_q,
        index_kv_cache,
        block_table,
        active_seq_lens,
        max_seq_len=max_seq_len,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        decode_query_len=decode_query_len,
        max_decode_query_len=max_decode_query_len,
        score_out=score,
    )
    expected = _reference_decode_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        active_seq_lens,
        decode_query_len,
        score_block_stride,
    )
    torch.testing.assert_close(score, expected)


# Sparse attention kernels.
def _reference_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float32)
    gqa_group_size = NUM_Q_HEADS // NUM_KV_HEADS
    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q_req = q[q_start:q_end]
        positions = torch.arange(seq_len, device="cuda")
        pages = block_table[req_id, positions // BLOCK_SIZE]
        rows = positions % BLOCK_SIZE
        kv_req = kv_cache[pages, :, rows]
        k_req = kv_req[..., :HEAD_DIM]
        v_req = kv_req[..., HEAD_DIM:].float()

        q_pos = prefix_len + torch.arange(q_len, device="cuda")
        key_blocks = positions // BLOCK_SIZE
        causal_mask = positions.unsqueeze(0) <= q_pos.unsqueeze(1)

        for kv_head in range(NUM_KV_HEADS):
            selected = topk_idx[kv_head, q_start:q_end]
            selected_mask = (key_blocks[None, :, None] == selected[:, None, :]).any(-1)
            mask = causal_mask & selected_mask
            head_start = kv_head * gqa_group_size
            head_end = head_start + gqa_group_size

            q_heads = q_req[:, head_start:head_end].transpose(0, 1)
            k_head = k_req[:, kv_head].T.expand(gqa_group_size, -1, -1)
            scores = torch.bmm(q_heads, k_head, out_dtype=torch.float32)
            scores = scores.transpose(0, 1) * SM_SCALE
            probs = torch.softmax(
                scores.masked_fill(~mask[:, None, :], -float("inf")), -1
            )
            out[q_start:q_end, head_start:head_end] = torch.einsum(
                "qhk,kd->qhd", probs, v_req[:, kv_head]
            )
        q_start += q_len
    return out.to(q.dtype)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"], indirect=True)
@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ((129, 257), (129, 257)),
        ((65, 129, 257), (129, 257, 385)),
    ],
)
def test_prefill_sparse_attention_correctness(
    kv_layout: str,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
):
    assert len(q_lens) == len(kv_lens)
    assert all(kv_len >= q_len for q_len, kv_len in zip(q_lens, kv_lens))

    # Build paged-KV metadata, including a non-identity page order.
    batch = len(q_lens)
    pages_per_req = [(kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE for kv_len in kv_lens]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device="cuda", dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[
            base_page : base_page + num_req_pages
        ]
        base_page += num_req_pages

    q_lens_t = torch.tensor(q_lens, device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = q_lens_t.cumsum(0)
    total_q = sum(q_lens)
    max_seqlen_q = max(q_lens)

    q_shape = (total_q, NUM_Q_HEADS, HEAD_DIM)
    q = torch.randn(q_shape, device="cuda", dtype=DTYPE)
    # Allocate the main KV cache through the backend layout contract so the
    # physical storage matches the active layout (contiguous NHD or strided
    # HND), while the kernels and reference see the logical-NHD view.
    kv_cache = _allocate_main_kv_via_contract(num_pages)

    # Build sparse block indices with the same contract as the real M3 indexer:
    # one forced local block, then score-selected older causal blocks.
    topk_shape = (NUM_KV_HEADS, total_q, TOPK)
    topk_idx = torch.full(topk_shape, -1, device="cuda", dtype=torch.int32)
    q_start = 0
    for q_len, prefix_len in zip(q_lens_t.tolist(), prefix_lens.tolist()):
        for local_q in range(q_len):
            current_block = (prefix_len + local_q) // BLOCK_SIZE
            older_blocks = torch.randperm(
                current_block, device="cuda", dtype=torch.int32
            )
            selected = torch.cat(
                [
                    torch.tensor([current_block], device="cuda", dtype=torch.int32),
                    older_blocks[: TOPK - 1],
                ]
            )
            topk_idx[:, q_start + local_q, : selected.numel()] = selected
        q_start += q_len

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_seqlen_q,
        NUM_KV_HEADS,
        SM_SCALE,
        actual,
    )

    expected = _reference_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        q_lens_t,
        seq_lens,
        prefix_lens,
    )
    torch.accelerator.synchronize()

    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < 2.5e-4
    assert error.max().item() < 1.7e-2


def test_main_backend_layout_contract():
    """The main sparse backend exposes the logical-NHD shape and the
    flash_attn-style stride order for each layout."""
    nb, bs, h, d = 7, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    logical = MiniMaxM3SparseBackend.get_kv_cache_shape(nb, bs, h, d)
    assert logical == (nb, h, bs, 2 * d)
    # The old separate K/V-axis shape is no longer the logical shape.
    assert logical != (nb, 2, bs, h, d)

    try:
        set_kv_cache_layout("HND")
        assert MiniMaxM3SparseBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        set_kv_cache_layout("NHD")
        assert MiniMaxM3SparseBackend.get_kv_cache_stride_order() == (0, 2, 1, 3)
    finally:
        set_kv_cache_layout(None)

    for layout in ("NHD", "HND"):
        try:
            set_kv_cache_layout(layout)
            order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
        finally:
            set_kv_cache_layout(None)
        # Valid permutation: no duplicates, covers every axis.
        assert set(order) == set(range(len(order)))

    # M3 has no cross-layer KV blocks.
    with pytest.raises(NotImplementedError):
        MiniMaxM3SparseBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )


def test_aiter_sparse_pa_layout_contract(monkeypatch):
    """The shuffle-only AITER path retains separately contiguous K/V storage."""
    import vllm.models.minimax_m3.common.sparse_attention as sparse_attn_mod

    monkeypatch.setattr(sparse_attn_mod.rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_attn_mod.rocm_aiter_ops,
        "is_shuffle_kv_cache_enabled",
        lambda: True,
    )

    nb, bs, h, d = 7, BLOCK_SIZE, 1, HEAD_DIM
    logical = MiniMaxM3SparseBackend.get_kv_cache_shape(nb, bs, h, d)
    order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
    assert logical == (nb, 2, bs, h, d)
    assert order == (1, 0, 2, 3, 4)

    physical_shape = tuple(logical[i] for i in order)
    inv_order = [order.index(i) for i in range(len(order))]
    raw = torch.empty(physical_shape, device="cuda", dtype=DTYPE)
    logical_view = raw.permute(*inv_order)
    key_cache, value_cache = logical_view.unbind(1)
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()


def test_aiter_sparse_pa_rejects_multiple_kv_heads(monkeypatch):
    """Do not pair AITER's separated cache layout with the Triton fallback."""
    import vllm.models.minimax_m3.common.sparse_attention as sparse_attn_mod

    monkeypatch.setattr(sparse_attn_mod.rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_attn_mod.rocm_aiter_ops,
        "is_shuffle_kv_cache_enabled",
        lambda: True,
    )

    with pytest.raises(ValueError, match="num_kv_heads == 1"):
        MiniMaxM3SparseBackend.get_kv_cache_shape(7, BLOCK_SIZE, 2, HEAD_DIM)


def test_main_backend_unknown_layout_raises(monkeypatch):
    """An unrecognized layout (injected past env-var validation) is rejected."""
    import vllm.models.minimax_m3.common.sparse_attention as sparse_attn_mod

    monkeypatch.setattr(sparse_attn_mod, "get_kv_cache_layout", lambda: "BOGUS")
    with pytest.raises(ValueError, match="Unknown cache layout format"):
        MiniMaxM3SparseBackend.get_kv_cache_stride_order()


def test_indexer_backend_stride_order_is_identity():
    """The 3-dim indexer cache must not inherit the parent's 4-element stride
    order; it overrides to the 3-element identity so the allocator keeps the
    contiguous layout."""
    assert MiniMaxM3IndexerBackend.get_kv_cache_stride_order() == (0, 1, 2)

    # Cross-layer (per-layer-stacked) KV blocks are not supported.
    with pytest.raises(NotImplementedError):
        MiniMaxM3IndexerBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )

    # The stride order matches the 3-dim indexer shape rank.
    indexer_shape = MiniMaxM3IndexerBackend.get_kv_cache_shape(
        5, BLOCK_SIZE, 1, HEAD_DIM
    )
    assert len(indexer_shape) == 3
    assert _stride_order_for(MiniMaxM3IndexerBackend, len(indexer_shape)) == (0, 1, 2)


def test_hnd_allocation_is_packed_head_major():
    """Under HND the backend-visible logical view is the packed head-major
    physical allocation."""
    nb, bs, h, d = 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    logical = MiniMaxM3SparseBackend.get_kv_cache_shape(nb, bs, h, d)
    try:
        set_kv_cache_layout("HND")
        stride_order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
    finally:
        set_kv_cache_layout(None)

    physical_shape = tuple(logical[i] for i in stride_order)
    assert physical_shape == (nb, h, bs, 2 * d)

    inv_order = [stride_order.index(i) for i in range(len(stride_order))]
    raw = torch.empty(physical_shape, device="cuda", dtype=DTYPE)
    view = raw.permute(*inv_order)
    expected = raw.view((nb, h, bs, 2 * d))

    assert view.shape == expected.shape
    assert view.stride() == expected.stride()
    assert view.storage_offset() == expected.storage_offset()

    # Negative: the NHD stride order under HND does not reproduce the
    # head-major view.
    wrong_order = (0, 2, 1, 3)
    wrong_inv = [wrong_order.index(i) for i in range(len(wrong_order))]
    wrong_view = raw.view(tuple(logical[i] for i in wrong_order)).permute(*wrong_inv)
    assert wrong_view.stride() != expected.stride()


def test_main_cache_is_block_first_and_unpadded():
    """The allocator's contiguous-view branch (not the padded-strided branch)
    is used for the main GQA cache: its spec is unpadded and the physical
    layout keeps num_blocks as the first dimension under both layouts."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        head_size_v=HEAD_DIM,
        dtype=DTYPE,
    )
    # Unpadded -> allocator uses kv_tensor.view(...) rather than as_strided().
    assert spec.page_size_padded is None

    logical = MiniMaxM3SparseBackend.get_kv_cache_shape(
        4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    )
    for layout in ("NHD", "HND"):
        try:
            set_kv_cache_layout(layout)
            order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
        finally:
            set_kv_cache_layout(None)
        inv_order = [order.index(i) for i in range(len(order))]
        # Physical first dim is num_blocks (block-first); required by the
        # padded-strided branch's block-first assumption if it were ever taken.
        assert inv_order[0] == 0
        assert logical[order[0]] == logical[0]


def _build_decode_inputs(
    seq_lens_list: tuple[int, ...],
    decode_query_len: int = 1,
    num_padded_reqs: int = 0,
):
    """Shared decode setup: uniform query tokens per request, a non-identity
    block table, and topk indices selecting the current block plus older causal
    blocks for each query token."""
    active_batch = len(seq_lens_list)
    batch = active_batch + num_padded_reqs
    pages_per_req = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens_list]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device="cuda", dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[
            base_page : base_page + num_req_pages
        ]
        base_page += num_req_pages

    seq_lens = torch.tensor(
        (*seq_lens_list, *([0] * num_padded_reqs)),
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn(
        batch * decode_query_len, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE
    )

    topk_idx = torch.full(
        (NUM_KV_HEADS, batch * decode_query_len, TOPK),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    token_id = 0
    for req_id, seq_len in enumerate(seq_lens_list):
        for local_q in range(decode_query_len):
            query_pos = seq_len - decode_query_len + local_q
            current_block = query_pos // BLOCK_SIZE
            older_blocks = torch.randperm(
                current_block, device="cuda", dtype=torch.int32
            )
            selected = torch.cat(
                [
                    torch.tensor([current_block], device="cuda", dtype=torch.int32),
                    older_blocks[: TOPK - 1],
                ]
            )
            topk_idx[:, token_id, : selected.numel()] = selected
            token_id += 1

    return q, block_table, seq_lens, topk_idx, num_pages


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MiniMax-M3 AITER sparse PA is ROCm-only",
)
def test_aiter_sparse_block_table_handles_padded_decode_rows():
    """Zero-length padded decode rows must produce empty sparse attention."""
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        PAGES_PER_SPARSE_BLOCK,
        minimax_m3_build_sparse_block_table,
    )

    topk_idx = torch.tensor(
        [[[1, 0, -1], [0, 1, -1]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_table = torch.tensor(
        [[5, 7], [11, 13]],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([129, 0], device="cuda", dtype=torch.int32)

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table(
        topk_idx, block_table, seq_lens
    )
    torch.accelerator.synchronize()

    expected_bt = torch.zeros_like(sparse_bt)
    expected_ctx = torch.tensor([129, 0], device="cuda", dtype=torch.int32)

    for slot, block_id in enumerate((0, 1)):
        base_phys = int(block_table[0, block_id]) * PAGES_PER_SPARSE_BLOCK
        start = slot * PAGES_PER_SPARSE_BLOCK
        expected_bt[0, start : start + PAGES_PER_SPARSE_BLOCK] = torch.arange(
            base_phys,
            base_phys + PAGES_PER_SPARSE_BLOCK,
            device="cuda",
            dtype=torch.int32,
        )

    assert torch.equal(sparse_ctx, expected_ctx)
    assert torch.equal(sparse_bt, expected_bt)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MiniMax-M3 AITER sparse PA is ROCm-only",
)
def test_aiter_decode_sparse_block_table_supports_spec_decode():
    """AITER sparse PA needs one compact page-16 table per speculative query.

    The decode indexer flattens speculative rows as
    ``req_id * decode_query_len + local_q``. This verifies that the ROCm AITER
    block-table adapter uses the same mapping before handing rows to Gluon.
    """
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        PAGES_PER_SPARSE_BLOCK,
        minimax_m3_build_sparse_block_table_decode,
    )

    decode_query_len = 4
    seq_lens = torch.tensor([260, 132, 0], device="cuda", dtype=torch.int32)
    block_table = torch.tensor(
        [
            [5, 7, 11],
            [13, 17, 19],
            [0, 0, 0],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    topk_idx = torch.tensor(
        [
            [
                [2, 0, 3, -1],
                [0, 2, 3, -1],
                [2, 1, 0, -1],
                [1, 2, 0, -1],
                [1, 0, 2, -1],
                [0, 1, 2, -1],
                [1, -1, 0, -1],
                [0, 2, 1, -1],
                [0, 1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, -1, -1],
            ]
        ],
        device="cuda",
        dtype=torch.int32,
    )

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_decode(
        topk_idx, block_table, seq_lens, decode_query_len
    )
    torch.accelerator.synchronize()

    expected_bt = torch.zeros_like(sparse_bt)
    expected_ctx = torch.empty_like(sparse_ctx)
    topk_cpu = topk_idx.cpu()[0].tolist()
    block_table_cpu = block_table.cpu()
    seq_lens_cpu = seq_lens.cpu().tolist()

    for row, selected_blocks in enumerate(topk_cpu):
        req_id = row // decode_query_len
        local_q = row - req_id * decode_query_len
        query_pos = seq_lens_cpu[req_id] - decode_query_len + local_q
        causal_len = max(query_pos + 1, 0)
        if causal_len == 0:
            expected_ctx[row] = 0
            continue
        self_block = query_pos // BLOCK_SIZE

        valid = [blk for blk in selected_blocks if 0 <= blk <= self_block]
        full_blocks = [blk for blk in valid if blk < self_block]
        tail_blocks = [blk for blk in valid if blk == self_block]
        ordered_blocks = full_blocks + tail_blocks

        for slot, block_id in enumerate(ordered_blocks):
            base_phys = int(block_table_cpu[req_id, block_id]) * PAGES_PER_SPARSE_BLOCK
            start = slot * PAGES_PER_SPARSE_BLOCK
            expected_bt[row, start : start + PAGES_PER_SPARSE_BLOCK] = torch.arange(
                base_phys,
                base_phys + PAGES_PER_SPARSE_BLOCK,
                device="cuda",
                dtype=torch.int32,
            )

        if tail_blocks:
            expected_ctx[row] = (
                len(full_blocks) * BLOCK_SIZE + causal_len - self_block * BLOCK_SIZE
            )
        else:
            expected_ctx[row] = min(len(valid) * BLOCK_SIZE, causal_len)

    assert torch.equal(sparse_ctx, expected_ctx)
    assert torch.equal(sparse_bt, expected_bt)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"], indirect=True)
@pytest.mark.parametrize(
    "seq_lens_list",
    [(130, 257), (129, 200, 384)],
)
@pytest.mark.parametrize("decode_query_len", [1, 4])
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
def test_decode_sparse_attention_correctness(
    kv_layout: str,
    seq_lens_list: tuple[int, ...],
    decode_query_len: int,
    num_padded_reqs: int,
):
    """Decode (split-K) parity under both layouts: this is the only coverage of
    the decode-site cache feed, and the strided HND case fails if the kernel
    ignores the cache strides."""
    torch.manual_seed(0)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(
        seq_lens_list, decode_query_len, num_padded_reqs
    )
    kv_cache = _allocate_main_kv_via_contract(num_pages)

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        SM_SCALE,
        actual,
        decode_query_len,
    )

    # Reuse the prefill reference: decode is a uniform query chunk ending at
    # seq_len - 1 for each request.
    active_batch = len(seq_lens_list)
    active_tokens = active_batch * decode_query_len
    q_lens_t = torch.full(
        (len(seq_lens_list),), decode_query_len, device="cuda", dtype=torch.int32
    )
    active_seq_lens = seq_lens[:active_batch]
    prefix_lens = active_seq_lens - q_lens_t
    expected = _reference_sparse_attn(
        q[:active_tokens],
        kv_cache,
        topk_idx[:, :active_tokens],
        block_table[:active_batch],
        q_lens_t,
        active_seq_lens,
        prefix_lens,
    )
    torch.accelerator.synchronize()

    error = (actual[:active_tokens].float() - expected.float()).abs()
    assert error.mean().item() < 2.5e-4
    assert error.max().item() < 1.7e-2


def test_decode_wrong_layout_breaks_parity():
    """Negative (AC-3/AC-5): consuming the physical HND buffer as if it were
    already contiguous-NHD (i.e. skipping the allocator's inverse permute)
    reorders the K/V content, so the decode output no longer matches the
    reference computed on the correct logical view. The mislabeled tensor keeps
    the same shape as the correct view, so the kernel stays in bounds."""
    torch.manual_seed(0)
    seq_lens_list = (130, 257)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(seq_lens_list)

    # Physical HND storage [blocks, heads, block, packed_kv_dim].
    phys = torch.randn(
        (num_pages, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM),
        device="cuda",
        dtype=DTYPE,
    )
    # Correct logical packed-HND view vs. the same bytes mislabeled as NHD
    # physical storage and then exposed as a logical cache.
    correct = phys
    wrong = phys.reshape(num_pages, BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD_DIM).permute(
        0, 2, 1, 3
    )

    q_lens_t = torch.ones(len(seq_lens_list), device="cuda", dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    expected = _reference_sparse_attn(
        q, correct, topk_idx, block_table, q_lens_t, seq_lens, prefix_lens
    )

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q, wrong, topk_idx, block_table, seq_lens, NUM_KV_HEADS, SM_SCALE, actual, 1
    )
    torch.accelerator.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() > 1.7e-2


def _make_attn_group(backend, spec):
    return AttentionGroup(
        backend=backend,
        layer_names=["main"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )


def test_main_cache_byte_identical_through_production_allocator():
    """AC-2: drive the real allocator (`_reshape_kv_cache`) for the M3 main
    `FullAttentionSpec` under HND and assert the backend-visible view has the
    same shape, stride, and storage offset as the packed-HND allocation; the
    indexer `MLAAttentionSpec` allocates through the same path to its 3-dim
    shape."""
    nb = 4
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        head_size_v=HEAD_DIM,
        dtype=DTYPE,
    )
    raw = torch.zeros(nb * spec.page_size_bytes, dtype=torch.int8)
    group = _make_attn_group(MiniMaxM3SparseBackend, spec)
    try:
        set_kv_cache_layout("HND")
        kv_caches = _reshape_kv_cache([group], {"main": raw}, "auto", [BLOCK_SIZE], {})
    finally:
        set_kv_cache_layout(None)
    view = kv_caches["main"]

    oracle = raw.view(DTYPE).view((nb, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM))
    assert tuple(view.shape) == tuple(oracle.shape)
    assert view.stride() == oracle.stride()
    assert view.storage_offset() == oracle.storage_offset()

    # Indexer cache allocates through the same path under both layouts.
    ispec = MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=HEAD_DIM, dtype=DTYPE
    )
    for layout in ("NHD", "HND"):
        iraw = torch.zeros(nb * ispec.page_size_bytes, dtype=torch.int8)
        igroup = AttentionGroup(
            backend=MiniMaxM3IndexerBackend,
            layer_names=["idx"],
            kv_cache_spec=ispec,
            kv_cache_group_id=0,
        )
        try:
            set_kv_cache_layout(layout)
            iout = _reshape_kv_cache([igroup], {"idx": iraw}, "auto", [BLOCK_SIZE], {})
        finally:
            set_kv_cache_layout(None)
        assert tuple(iout["idx"].shape) == (nb, BLOCK_SIZE, HEAD_DIM)


def test_indexer_inherited_stride_order_trips_allocator_assert():
    """AC-4 negative: without the indexer override, the inherited 4-element
    stride order trips the allocator's `len(stride_order) == len(shape)` assert
    for the 3-dim indexer shape; the `AssertionError` is NOT swallowed by the
    allocator's `(AttributeError, NotImplementedError)` fallback."""

    class _BrokenIndexerBackend(MiniMaxM3IndexerBackend):
        # Simulate inheriting the parent's 4-element stride order.
        get_kv_cache_stride_order = staticmethod(
            MiniMaxM3SparseBackend.get_kv_cache_stride_order
        )

    nb = 4
    ispec = MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=HEAD_DIM, dtype=DTYPE
    )
    iraw = torch.zeros(nb * ispec.page_size_bytes, dtype=torch.int8)
    igroup = AttentionGroup(
        backend=_BrokenIndexerBackend,
        layer_names=["idx"],
        kv_cache_spec=ispec,
        kv_cache_group_id=0,
    )
    try:
        set_kv_cache_layout("HND")
        with pytest.raises(AssertionError):
            _reshape_kv_cache([igroup], {"idx": iraw}, "auto", [BLOCK_SIZE], {})
    finally:
        set_kv_cache_layout(None)


def test_padded_main_cache_is_flagged():
    """AC-2.1 negative: the M3 main cache relies on the allocator's
    contiguous-view branch (`page_size_padded is None`). A spec that sets
    `page_size_padded` is explicitly flagged rather than silently wrong-strided."""

    def _require_unpadded_block_first(spec, stride_order):
        inv_order = [stride_order.index(i) for i in range(len(stride_order))]
        assert spec.page_size_padded is None, (
            "main GQA cache must be unpadded to use the contiguous-view "
            "allocator branch"
        )
        assert inv_order[0] == 0, "main GQA cache must remain block-first"

    try:
        set_kv_cache_layout("HND")
        stride_order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
    finally:
        set_kv_cache_layout(None)

    good = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        head_size_v=HEAD_DIM,
        dtype=DTYPE,
    )
    _require_unpadded_block_first(good, stride_order)  # passes

    padded = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        head_size_v=HEAD_DIM,
        dtype=DTYPE,
        page_size_padded=good.page_size_bytes + 128,
    )
    with pytest.raises(AssertionError):
        _require_unpadded_block_first(padded, stride_order)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"], indirect=True)
def test_reshape_and_cache_flash_write_persists(kv_layout: str):
    """AC-5 write path: the `reshape_and_cache_flash` write site now consumes
    packed-content K/V split views. Writing through those views must persist
    into the bound storage under both layouts."""
    torch.manual_seed(0)
    num_pages = 4
    kv_cache = _allocate_main_kv_via_contract(num_pages)
    with torch.no_grad():
        kv_cache.zero_()

    # Exactly the production write-site code under test.
    key_cache, value_cache = kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)

    num_tokens = 12
    slot_mapping = torch.randperm(num_pages * BLOCK_SIZE, device="cuda")[
        :num_tokens
    ].to(torch.int64)
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    value = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    scale = torch.ones((), device="cuda")
    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto", scale, scale
    )
    torch.accelerator.synchronize()

    # Read back through the independent logical view; proves the writes landed
    # in the engine-bound storage, not a detached copy.
    for t in range(num_tokens):
        slot = int(slot_mapping[t].item())
        blk, intra = divmod(slot, BLOCK_SIZE)
        torch.testing.assert_close(kv_cache[blk, :, intra, :HEAD_DIM], key[t])
        torch.testing.assert_close(kv_cache[blk, :, intra, HEAD_DIM:], value[t])
