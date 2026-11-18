# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse MLA index remap test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse MLA index remap test requires a supported AITER "
        "installation.",
        allow_module_level=True,
    )

from vllm.model_executor.layers.attention import sparse_mla_attention as sparse_mla_mod
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod

NUM_HEADS = 16
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = 576
NUM_TOKENS = 2
TOPK_TOKENS = 4


@pytest.mark.parametrize("rows,topk", [(0, 512), (1, 512), (96, 2048), (300, 2048)])
def test_dcp_csr_preserves_valid_slots_and_empty_rows(rows, topk):
    generator = torch.Generator(device="cuda").manual_seed(7)
    # A row stride larger than top-k also occurs with sliced index buffers.
    indices = torch.randint(
        -1024,
        4096,
        (rows, topk + 16),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )[:, :topk]
    if rows:
        indices[0].fill_(-1)
    indptr, packed = sparse_mod._compact_dcp_indices(indices)
    expected = indices[indices >= 0]
    counts = (indices >= 0).sum(dim=1, dtype=torch.int32)
    torch.testing.assert_close(
        indptr[1:], counts.cumsum(0), atol=0, rtol=0, check_dtype=False
    )
    assert indptr[0].item() == 0
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def _poison_empty(monkeypatch, value):
    """Fill every `torch.empty` result, so an unwritten entry is visible."""
    real_empty = torch.empty

    def empty(*args, **kwargs):
        return real_empty(*args, **kwargs).fill_(value)

    monkeypatch.setattr(torch, "empty", empty)


@pytest.mark.parametrize("rows,topk", [(0, 512), (1, 512), (96, 2048), (300, 2048)])
def test_dcp_csr_with_counts_packs_row_prefixes(monkeypatch, rows, topk):
    """The counts arm is the DCP decode path: every pure-decode step uses it.

    With counts the valid entries are a row prefix rather than the entries
    above the `-1` sentinel, and `indptr` is left uninitialised for the
    kernel to fill. Poisoning the allocation fails this test if
    `_pack_valid_rows` ever stops writing all `rows + 1` entries.
    """
    generator = torch.Generator(device="cuda").manual_seed(13)
    # Positive garbage past the prefix: a sentinel-based pack would keep it.
    indices = torch.randint(
        0,
        4096,
        (rows, topk + 16),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )[:, :topk]
    counts = torch.randint(
        0, topk + 1, (rows,), generator=generator, dtype=torch.int32, device="cuda"
    )
    if rows:
        counts[0] = 0
        counts[-1] = topk

    _poison_empty(monkeypatch, -7)
    indptr, packed = sparse_mod._compact_dcp_indices(indices, counts)

    assert indptr.numel() == rows + 1
    assert indptr[0].item() == 0
    torch.testing.assert_close(
        indptr[1:], counts.cumsum(0), atol=0, rtol=0, check_dtype=False
    )
    expected = torch.cat(
        [indices[row, : counts[row]] for row in range(rows)] + [indices.new_empty(0)]
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


@triton.jit
def _score_key_probe(src, dst, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(dst + i, sparse_mod._score_key(tl.load(src + i)))


def test_score_key_orders_floats_and_sinks_nan():
    """merge_dcp_scores compares scores as integers, so the map must order.

    NaN has to land below -inf: a NaN logit must never win a top-k slot from
    a real one.
    """
    values = torch.tensor(
        [float("nan"), float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf")],
        dtype=torch.float32,
        device="cuda",
    )
    keys = torch.empty(8, dtype=torch.int32, device="cuda")
    _score_key_probe[(1,)](torch.cat([values, values[:1]]), keys, 8)

    assert keys[0].item() == -2147483648
    ordered = keys[:7].tolist()
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == 7


def _score_keys(scores):
    bits = scores.view(torch.int32).to(torch.int64)
    keys = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return torch.where(
        (bits & 0x7FFFFFFF) > 0x7F800000, torch.tensor(-2147483648), keys
    )


def _reference_merge(scores, local_indices, token_block_table, rank, page, stride):
    """Selection of _select_owned_slots, written out row by row.

    Args:
        scores: Rank-major scores of shape `[rows, world * k]`.
        local_indices: Every rank's local KV positions, `[world, rows, k]`.
            A negative entry is a padded slot the row does not own.
        token_block_table: Per-row block table.
        rank: The rank whose selection is wanted.
        page: KV block size.
        stride: Cache rows per KV block.

    Returns:
        The CSR row pointer and the packed physical KV rows of `rank`.

    """
    world, rows, k = local_indices.shape
    keys = _score_keys(scores)
    # The k-th largest key: the threshold the aiter top-k hands the kernel.
    threshold = keys.sort(dim=1, descending=True).values[:, k - 1].tolist()
    keys = keys.tolist()
    live = (local_indices[rank] >= 0).tolist()
    locals_ = local_indices[rank].tolist()
    pages = token_block_table.tolist()
    counts = []
    packed = []
    for row in range(rows):
        limit = threshold[row]
        row_keys = keys[row]
        room = k - sum(key > limit for key in row_keys)
        mine = row_keys[rank * k : (rank + 1) * k]
        tied = [live[row][col] and mine[col] == limit for col in range(k)]
        if any(tied):
            # The kernel counts every earlier plane's entries at the threshold
            # with no liveness mask, so the reference does the same.
            room -= sum(key == limit for key in row_keys[: rank * k])
        cols = []
        ordinal = 0
        for col in range(k):
            if not live[row][col]:
                continue
            if mine[col] > limit:
                cols.append(col)
            elif tied[col]:
                if ordinal < room:
                    cols.append(col)
                ordinal += 1
        counts.append(len(cols))
        for col in cols:
            local = locals_[row][col]
            packed.append(pages[row][local // page] * stride + local % page)
    indptr = torch.tensor([0] + counts, dtype=torch.int32).cumsum(0)
    return indptr, torch.tensor(packed, dtype=torch.int32, device=scores.device)


def _merge_inputs(rows, k, world_size, num_pages, page, live_per_row, seed):
    """Build one merge case: tied scores, with each plane padding its tail.

    `live_per_row` gives each row's total live entries across the ranks; a
    row with fewer than `k` of them is a short row, whose k-th score is then
    `-inf` and whose padding ties with it.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # A tiny score range makes ties at the k-th score the common case.
    scores = torch.randint(
        1, 4, (rows, world_size * k), generator=generator, device="cuda"
    ).to(torch.float32)
    local_indices = torch.randint(
        0,
        num_pages * page,
        (world_size, rows, k),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    planes = scores.view(rows, world_size, k).permute(1, 0, 2)
    for row, total in enumerate(live_per_row):
        for plane in range(world_size):
            # Spread the row's context over the ranks, as DCP interleaving
            # does, and let each plane pad its own tail.
            owned = total // world_size + (plane < total % world_size)
            local_indices[plane, row, owned:] = -1
            planes[plane, row, owned:] = float("-inf")
    token_block_table = torch.randint(
        0,
        1000,
        (rows, num_pages),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    return scores, local_indices, token_block_table


def _check_merge(scores, local_indices, token_block_table, page, stride):
    """Run every rank through merge_dcp_scores and return the slots taken."""
    world_size, rows, k = local_indices.shape
    total = 0
    for rank in range(world_size):
        slots, indptr = sparse_mod.merge_dcp_scores(
            scores,
            local_indices[rank],
            token_block_table,
            rank,
            world_size,
            page,
            stride,
        )
        want_indptr, want_slots = _reference_merge(
            scores, local_indices, token_block_table, rank, page, stride
        )
        torch.testing.assert_close(
            indptr.cpu(), want_indptr, atol=0, rtol=0, check_dtype=False
        )
        torch.testing.assert_close(
            slots[: want_slots.numel()], want_slots, atol=0, rtol=0
        )
        total += int(indptr[-1])
    return total


@pytest.mark.parametrize("rows,k,world_size", [(8, 16, 4), (3, 512, 2)])
def test_merge_dcp_scores_matches_rank_major_tie_break(rows, k, world_size):
    """Ranks must together take the global top-k, each slot exactly once.

    The tie-break at the k-th score walks the lower-numbered ranks to work out
    how much room is left. An off-by-one there drops or duplicates KV slots
    without crashing, which only shows up as accuracy loss.

    Half the rows are short, so their k-th score is `-inf` and their padding
    ties with it. Padding must never be taken: it would read the block table
    at index `-1` and emit a KV row of another request.
    """
    sparse_mod._score_widths.clear()
    page = 4
    stride = 37
    # Rows alternate between fewer live entries than k and more.
    live_per_row = [k // 2 + row if row % 2 else 2 * k - row for row in range(rows)]
    scores, local_indices, token_block_table = _merge_inputs(
        rows, k, world_size, 16, page, live_per_row, seed=5
    )

    total = _check_merge(scores, local_indices, token_block_table, page, stride)
    assert total == sum(min(live, k) for live in live_per_row)


def test_merge_dcp_scores_grows_the_cached_row_widths():
    """The cached `ends` vector is reused, so a wider batch must grow it.

    A too-short `ends` would leave the extra rows reading whatever the
    neighbouring memory holds.
    """
    sparse_mod._score_widths.clear()
    k = 16
    world_size = 2
    page = 4
    stride = 37
    for rows in (1100, 1300, 8):
        scores, local_indices, token_block_table = _merge_inputs(
            rows, k, world_size, 16, page, [2 * k] * rows, seed=rows
        )
        _check_merge(scores, local_indices, token_block_table, page, stride)
    ends = sparse_mod._score_widths[(scores.device, world_size * k)]
    assert ends.numel() >= 1300


def test_merge_dcp_scores_keeps_padding_out_of_a_short_row():
    """A context shorter than top-k leaves the k-th score at -inf.

    The indexer pads such a row with -1 local indices and -inf scores, so
    every padded entry ties with the k-th score. Only the real entries may be
    taken: a padded one would read the block table at index -1 and emit a KV
    row that belongs to another request. Rows shorter than top-k happen on
    every request until it has produced topk_tokens tokens of context.
    """
    world_size, page, stride = 2, 4, 37
    neg_inf = float("-inf")
    scores = torch.tensor(
        [[9.0, 8.0, neg_inf, neg_inf, 7.0, neg_inf, neg_inf, neg_inf]],
        dtype=torch.float32,
        device="cuda",
    )
    local_indices = torch.tensor(
        [[0, 1, -1, -1], [2, -1, -1, -1]], dtype=torch.int32, device="cuda"
    )
    token_block_table = torch.tensor([[5]], dtype=torch.int32, device="cuda")

    taken = []
    for rank in range(world_size):
        slots, indptr = sparse_mod.merge_dcp_scores(
            scores,
            local_indices[rank : rank + 1],
            token_block_table,
            rank,
            world_size,
            page,
            stride,
        )
        assert indptr.tolist() == [0, 2 - rank]
        taken.extend(slots[: int(indptr[-1])].tolist())

    assert taken == [5 * stride + 0, 5 * stride + 1, 5 * stride + 2]


@pytest.fixture(autouse=True)
def _reset_index_epochs():
    # The counter is keyed by id(), and a freed test buffer can hand its id to
    # the next one, so a leftover count would leak across tests.
    sparse_mod._INDEX_EPOCHS.clear()
    yield
    sparse_mod._INDEX_EPOCHS.clear()


def _make_metadata():
    return SimpleNamespace(
        num_actual_tokens=NUM_TOKENS,
        req_id_per_token=torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cpu"),
        block_table=torch.arange(16, dtype=torch.int32, device="cpu").view(2, 8),
        paged_kv_indptr=torch.zeros(NUM_TOKENS + 1, dtype=torch.int32, device="cpu"),
        paged_kv_indices=torch.zeros(
            NUM_TOKENS * TOPK_TOKENS, dtype=torch.int32, device="cpu"
        ),
        block_size=1,
        topk_tokens=TOPK_TOKENS,
        remapped_buf=None,
        remapped_epoch=-1,
    )


def _patch_init_deps(monkeypatch, *, speculative: bool = False):
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper,
        "check_num_heads_validity",
        staticmethod(lambda num_heads: None),
    )
    vllm_config = SimpleNamespace(
        # One decode row per token. get_sparse_mla_index_group_max_rows() takes
        # min(max_num_batched_tokens, max_num_seqs * max_decode_query_len), so
        # a smaller max_num_seqs would clamp the index workspace below the
        # NUM_TOKENS rows these tests drive through forward_mqa.
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=NUM_TOKENS, max_num_seqs=NUM_TOKENS
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        speculative_config=SimpleNamespace() if speculative else None,
        attention_config=SimpleNamespace(hisparse_config=None, sparse_mla_shard_q=True),
        # One rank, so the layer takes neither the ShardQ nor the DCP path.
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        kv_transfer_config=None,
    )
    monkeypatch.setattr(sparse_mod, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(sparse_mla_mod, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        sparse_mla_mod, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        sparse_mod,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: tuple(
                torch.zeros(shape, dtype=dtype, device="cpu") for shape, dtype in specs
            )
        ),
    )


def _build_impl(indexer, topk_indices_buffer):
    return sparse_mod.ROCMAiterMLASparseImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_indices_buffer,
        indexer=indexer,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        v_head_dim=QK_NOPE_HEAD_DIM,
        kv_b_proj=None,
    )


def _new_buffer():
    return torch.zeros(NUM_TOKENS, TOPK_TOKENS, dtype=torch.int32, device="cpu")


def _patch_forward_deps(monkeypatch, impl, remap_calls):
    monkeypatch.setattr(
        sparse_mod,
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: remap_calls.append(args),
    )
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper,
        "get_mla_padded_q",
        staticmethod(lambda num_heads, q: q),
    )
    attn_out = torch.zeros(NUM_TOKENS, NUM_HEADS, KV_LORA_RANK, device="cpu")
    monkeypatch.setattr(
        type(impl),
        "_forward_mla",
        lambda self, layer, q, kv_cache, attn_metadata: (attn_out, None),
    )
    return attn_out


def _run_forward(impl, metadata):
    return impl.forward_mqa(
        torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, device="cpu"),
        torch.zeros(4, 1, HEAD_SIZE, device="cpu"),
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )


@pytest.mark.parametrize("has_indexer", [True, False])
def test_owns_indexer_is_structural(monkeypatch, has_indexer):
    """owns_indexer reflects the layer, not the serving config."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    indexer = SimpleNamespace(topk_indices_buffer=buffer) if has_indexer else None

    impl = _build_impl(indexer, buffer)

    assert impl.owns_indexer is has_indexer
    assert impl.topk_indices_buffer is buffer


def test_forward_mqa_remaps_once_per_metadata(monkeypatch):
    """Skip layers sharing a metadata share its remap; the first one pays.

    All layers read one top-k buffer, so the reuse has to key off the metadata
    that owns `paged_kv_indices`, not off the buffer being distinct.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    first = _build_impl(None, buffer)
    second = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    attn_out = _patch_forward_deps(monkeypatch, first, remap_calls)
    metadata = _make_metadata()

    out, lse = _run_forward(first, metadata)
    assert out is attn_out
    assert lse is None
    assert len(remap_calls) == 1
    assert metadata.remapped_buf is buffer

    _run_forward(second, metadata)
    assert len(remap_calls) == 1


def test_forward_mqa_remaps_for_each_metadata(monkeypatch):
    """Skip layers on a second metadata must remap its own paged_kv_indices.

    Regression: gating on `indexer is not None` skipped every layer in the
    non-indexer metadata group, leaving its `paged_kv_indices` unwritten.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    indexer_layer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_layer = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, indexer_layer, remap_calls)

    indexer_metadata = _make_metadata()
    _run_forward(indexer_layer, indexer_metadata)
    assert len(remap_calls) == 1
    assert indexer_metadata.remapped_buf is buffer

    # A distinct metadata instance carries its own, still-unwritten indices.
    skip_metadata = _make_metadata()
    _run_forward(skip_layer, skip_metadata)
    assert len(remap_calls) == 2
    assert skip_metadata.remapped_buf is buffer


def test_forward_mqa_remaps_after_a_later_indexer_write(monkeypatch):
    """A skip metadata must be refreshed once per indexer write, not once ever.

    Regression for cross-layer index sharing (`index_topk_freq`). With
    freq=4 on a 78-layer model the 57 skip layers share one metadata while the
    21 indexer layers share another, so the skip metadata has to pick up every
    new selection. Recording only the buffer left layers 4 onwards attending
    to the first selection of the pass, which collapsed RULER niah_single_2 to
    0.026 on GLM-5.3 and took GPQA from 0.884 to 0.672.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    # Construction order mirrors the layer stack: indexer, three skips, indexer.
    _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    first_skip = _build_impl(None, buffer)
    reuse_skip = _build_impl(None, buffer)
    _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    next_skip = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, first_skip, remap_calls)
    skip_metadata = _make_metadata()

    _run_forward(first_skip, skip_metadata)
    assert len(remap_calls) == 1

    # Same selection, same metadata: reuse.
    _run_forward(reuse_skip, skip_metadata)
    assert len(remap_calls) == 1

    # A later indexer layer has since rewritten the buffer in place. The object
    # is unchanged, so only the epoch can reveal it.
    _run_forward(next_skip, skip_metadata)
    assert len(remap_calls) == 2
    assert skip_metadata.remapped_epoch == next_skip.index_epoch


def test_index_epoch_tracks_the_preceding_indexer_write(monkeypatch):
    """Layers between two indexer layers share the earlier one's epoch."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    first_indexer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_a = _build_impl(None, buffer)
    skip_b = _build_impl(None, buffer)
    second_indexer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_c = _build_impl(None, buffer)

    assert skip_a.index_epoch == first_indexer.index_epoch
    assert skip_b.index_epoch == first_indexer.index_epoch
    assert second_indexer.index_epoch == first_indexer.index_epoch + 1
    assert skip_c.index_epoch == second_indexer.index_epoch


def test_forward_mqa_remaps_when_buffer_differs(monkeypatch):
    """A layer reading a different buffer cannot reuse the metadata's remap."""
    _patch_init_deps(monkeypatch)
    remapped_buffer = _new_buffer()
    other_buffer = _new_buffer()
    layer = _build_impl(None, other_buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, layer, remap_calls)
    metadata = _make_metadata()
    metadata.remapped_buf = remapped_buffer

    _run_forward(layer, metadata)
    assert len(remap_calls) == 1
    assert metadata.remapped_buf is other_buffer


def test_forward_mqa_indexer_layer_always_remaps(monkeypatch):
    """The indexer rewrites its buffer in place, so its remap is never stale."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    impl = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, impl, remap_calls)
    metadata = _make_metadata()
    metadata.remapped_buf = buffer
    metadata.remapped_epoch = impl.index_epoch

    _run_forward(impl, metadata)
    assert len(remap_calls) == 1
