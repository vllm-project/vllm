# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 sparse prefill attention kernels."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3SparseBackend,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

if not current_platform.is_cuda():
    pytest.skip("MiniMax M3 attention kernels require CUDA.", allow_module_level=True)


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
    sm_scale: float,
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
        score = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * sm_scale

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

    actual = minimax_m3_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=q_lens.max().item(),
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
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
        head_dim**-0.5,
    )
    assert torch.equal(actual, expected)


def test_decode_index_topk_correctness():
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    seq_lens = torch.tensor((7, 129, 1025), device="cuda", dtype=torch.int32)
    q_lens = torch.ones_like(seq_lens)
    prefix_lens = seq_lens - 1
    batch = seq_lens.numel()
    max_seq_len = seq_lens.max().item()
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.ones(batch, num_idx_heads, head_dim, device="cuda")
    index_kv_cache = torch.empty(num_pages, BLOCK_SIZE, head_dim, device="cuda")
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = block_table[req_id, block_id]
            index_kv_cache[page].fill_(block_id + 1)

    actual = minimax_m3_index_topk_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
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
        head_dim**-0.5,
    )
    assert torch.equal(actual, expected)


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
        k_req = kv_cache[pages, 0, rows]
        v_req = kv_cache[pages, 1, rows].float()

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
@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ((129, 257), (129, 257)),
        ((65, 129, 257), (129, 257, 385)),
    ],
)
def test_prefill_sparse_attention_correctness(
    kv_layout: str,
    backend: str,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
):
    if backend == "cutedsl":
        if not current_platform.is_device_capability_family(100):
            pytest.skip("MiniMax M3 CuteDSL prefill requires CUDA SM10x.")
        if not has_cutedsl():
            pytest.skip("cutedsl (cutlass) is not installed")

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
    cu_seqlens_k = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens_k[1:] = seq_lens.cumsum(0)
    total_q = sum(q_lens)
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(kv_lens)

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
    if backend == "triton":
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
    else:
        from vllm.models.minimax_m3.nvidia.ops.prefill_gqa_sparse import (
            minimax_m3_sparse_attn_cutedsl,
        )

        minimax_m3_sparse_attn_cutedsl(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens,
            cu_seqlens_k,
            seq_lens,
            max_seqlen_q,
            max_seqlen_k,
            NUM_KV_HEADS,
            SM_SCALE,
            actual,
            total_kv_blocks=num_pages,
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
    assert logical == (nb, 2, bs, h, d)
    # The old HND-ordered shape is no longer the logical shape.
    assert logical != (nb, 2, h, bs, d)

    try:
        set_kv_cache_layout("HND")
        assert MiniMaxM3SparseBackend.get_kv_cache_stride_order() == (0, 1, 3, 2, 4)
        set_kv_cache_layout("NHD")
        assert MiniMaxM3SparseBackend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
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


def test_main_backend_unknown_layout_raises(monkeypatch):
    """An unrecognized layout (injected past env-var validation) is rejected."""
    import vllm.models.minimax_m3.common.sparse_attention as sparse_attn_mod

    monkeypatch.setattr(sparse_attn_mod, "get_kv_cache_layout", lambda: "BOGUS")
    with pytest.raises(ValueError, match="Unknown cache layout format"):
        MiniMaxM3SparseBackend.get_kv_cache_stride_order()


def test_indexer_backend_stride_order_is_identity():
    """The 3-dim indexer cache must not inherit the parent's 5-element stride
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


def test_hnd_allocation_is_byte_identical_to_transpose():
    """Under HND the backend-visible logical view is byte-identical to the
    pre-change allocate-HND-then-transpose(2, 3) workaround."""
    nb, bs, h, d = 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    logical = MiniMaxM3SparseBackend.get_kv_cache_shape(nb, bs, h, d)
    try:
        set_kv_cache_layout("HND")
        stride_order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
    finally:
        set_kv_cache_layout(None)

    physical_shape = tuple(logical[i] for i in stride_order)
    # The physical (permuted) shape equals the old hardcoded HND shape.
    assert physical_shape == (nb, 2, h, bs, d)

    inv_order = [stride_order.index(i) for i in range(len(stride_order))]
    raw = torch.empty(physical_shape, device="cuda", dtype=DTYPE)
    view = raw.permute(*inv_order)
    expected = raw.view((nb, 2, h, bs, d)).transpose(2, 3)

    assert view.shape == expected.shape
    assert view.stride() == expected.stride()
    assert view.storage_offset() == expected.storage_offset()

    # Negative: the identity (wrong) stride order under HND does not reproduce
    # the transpose view.
    wrong_view = raw.view(logical)
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


def _build_decode_inputs(seq_lens_list: tuple[int, ...]):
    """Shared decode setup: one query token per request at position seq_len-1,
    a non-identity block table, and topk indices selecting the current block
    plus older causal blocks."""
    batch = len(seq_lens_list)
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

    seq_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)

    topk_idx = torch.full(
        (NUM_KV_HEADS, batch, TOPK), -1, device="cuda", dtype=torch.int32
    )
    for req_id, seq_len in enumerate(seq_lens_list):
        current_block = (seq_len - 1) // BLOCK_SIZE
        older_blocks = torch.randperm(current_block, device="cuda", dtype=torch.int32)
        selected = torch.cat(
            [
                torch.tensor([current_block], device="cuda", dtype=torch.int32),
                older_blocks[: TOPK - 1],
            ]
        )
        topk_idx[:, req_id, : selected.numel()] = selected

    return q, block_table, seq_lens, topk_idx, num_pages


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"], indirect=True)
@pytest.mark.parametrize(
    "seq_lens_list",
    [(130, 257), (129, 200, 384)],
)
def test_decode_sparse_attention_correctness(
    kv_layout: str,
    seq_lens_list: tuple[int, ...],
):
    """Decode (split-K) parity under both layouts: this is the only coverage of
    the decode-site cache feed, and the strided HND case fails if the kernel
    ignores the cache strides."""
    torch.manual_seed(0)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(seq_lens_list)
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
    )

    # Reuse the prefill reference: each request is a single query token at
    # position seq_len-1 (q_len == 1, prefix_len == seq_len-1).
    q_lens_t = torch.ones(len(seq_lens_list), device="cuda", dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    expected = _reference_sparse_attn(
        q, kv_cache, topk_idx, block_table, q_lens_t, seq_lens, prefix_lens
    )
    torch.accelerator.synchronize()

    error = (actual.float() - expected.float()).abs()
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

    # Physical HND storage [blocks, 2, heads, block, dim].
    phys = torch.randn(
        (num_pages, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM), device="cuda", dtype=DTYPE
    )
    # Correct logical-NHD view (strided) vs. the same bytes mislabeled as a
    # contiguous-NHD cache — same shape, different content mapping.
    correct = phys.permute(0, 1, 3, 2, 4)
    wrong = phys.reshape(num_pages, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

    q_lens_t = torch.ones(len(seq_lens_list), device="cuda", dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    expected = _reference_sparse_attn(
        q, correct, topk_idx, block_table, q_lens_t, seq_lens, prefix_lens
    )

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q, wrong, topk_idx, block_table, seq_lens, NUM_KV_HEADS, SM_SCALE, actual
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
    same shape, stride, and storage offset as the pre-change
    allocate-HND-then-transpose path; the indexer `MLAAttentionSpec` allocates
    through the same path to its 3-dim shape."""
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

    oracle = raw.view(DTYPE).view((nb, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM))
    oracle = oracle.transpose(2, 3)
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
    """AC-4 negative: without the indexer override, the inherited 5-element
    stride order trips the allocator's `len(stride_order) == len(shape)` assert
    for the 3-dim indexer shape; the `AssertionError` is NOT swallowed by the
    allocator's `(AttributeError, NotImplementedError)` fallback."""

    class _BrokenIndexerBackend(MiniMaxM3IndexerBackend):
        # Simulate inheriting the parent's 5-element stride order.
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
    `self.kv_cache.unbind(1)` directly. Writing through those views must persist
    into the bound storage (read back through an independent logical view) under
    both layouts — a `.contiguous()` copy of the unbind slice would leave the
    bound storage unchanged."""
    torch.manual_seed(0)
    num_pages = 4
    kv_cache = _allocate_main_kv_via_contract(num_pages)
    with torch.no_grad():
        kv_cache.zero_()

    # Exactly the production write-site code under test.
    key_cache, value_cache = kv_cache.unbind(1)

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
        torch.testing.assert_close(kv_cache[blk, 0, intra], key[t])
        torch.testing.assert_close(kv_cache[blk, 1, intra], value[t])
