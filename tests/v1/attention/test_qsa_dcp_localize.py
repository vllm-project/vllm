# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP selection localization and empty-owner behavior."""

from importlib import import_module

import pytest
import torch

PAD = -1
# The cache dtypes the QSA reader takes: e4m3 bytes as uint8 everywhere, the
# e4m3 view only on sm_89+ (Triton has no fp8e4nv pointer type below).
READER_CACHE_DTYPES = ["bf16", "uint8"] + (
    ["e4m3"]
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    else []
)


def _owner(pos, world, interleave):
    return (pos // interleave) % world


def _local_id(pos, world, interleave):
    return (pos // (world * interleave)) * interleave + (pos % interleave)


def _reference_localize(row, world, rank, interleave):
    """Plain enumeration. Returns (localized prefix, kept count)."""
    kept = []
    for g in row:
        if g < 0:
            continue
        if _owner(g, world, interleave) != rank:
            continue
        kept.append(_local_id(g, world, interleave))
    return kept, len(kept)


def _pack(rows, width, device="cpu"):
    """Build the packed buffer: selection columns plus a trailing count."""
    out = torch.full((len(rows), width + 1), PAD, dtype=torch.int32, device=device)
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            out[i, j] = g
        out[i, width] = len(row[:width])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 4), (4, 1), (4, 16)])
@pytest.mark.parametrize("width", [32, 2051])
def test_kernel_matches_the_reference(world, interleave, width):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    rows = (
        [list(range(0, 64, 2)), list(range(1, 40)), [], [7]]
        if width == 32
        else [list(range(width)), [0], []]
    )
    packed = _pack(rows, width).cuda()
    source = packed.clone()
    local = torch.empty_like(packed)
    for rank in range(world):
        expected = [
            _reference_localize(row[:width], world, rank, interleave) for row in rows
        ]
        for _ in range(5 if width == 2051 else 1):
            local.fill_(123)
            qsa_localize_dcp_indices(packed, local, world, rank, interleave)
            # The layer's selection buffer must survive: MTP steps read it again.
            assert torch.equal(packed, source)
            for row, (ids, count) in enumerate(expected):
                assert int(local[row, width]) == count
                assert local[row, :count].tolist() == ids
                assert bool((local[row, count:width] == PAD).all())


# --- MTP selection reuse ----------------------------------------------------


def _qsa_module():
    """The QSA module, imported after the model module (they import each other)."""
    import_module("vllm.models.qwen4_exp.nvidia.model")
    return import_module("vllm.models.qwen4_exp.nvidia.qsa")


def _localize_in_place(selection, world, rank, interleave):
    """The indexer's call: the impl localizes the layer buffer in place."""
    from types import SimpleNamespace

    impl = SimpleNamespace(
        dcp_world_size=world, dcp_rank=rank, cp_kv_cache_interleave_size=interleave
    )
    _qsa_module().Qwen4ExpQSAFlashAttentionImpl.localize_dcp_selection_(impl, selection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("world,interleave", [(2, 1), (4, 1), (4, 16)])
@pytest.mark.parametrize("width", [32, 2051])
def test_in_place_localize_equals_a_separate_output(world, interleave, width):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    rows = [list(range(0, 2 * width, 2)), list(range(1, 40)), [], [7]]
    packed = _pack(rows, width).cuda()
    for rank in range(world):
        expected = qsa_localize_dcp_indices(
            packed, torch.empty_like(packed), world, rank, interleave
        )
        buffer = packed.clone()
        _localize_in_place(buffer, world, rank, interleave)
        assert torch.equal(buffer, expected)


@pytest.mark.cpu_test
def test_without_dcp_the_selection_stays_global():
    selection = _pack([[5, 6], [7]], 8)
    source = selection.clone()
    _localize_in_place(selection, 1, 0, 1)
    assert torch.equal(selection, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA kernels need CUDA")
@pytest.mark.parametrize("world,interleave", [(2, 1), (4, 1), (4, 16)])
def test_reused_local_rows_match_per_pass_localization(world, interleave):
    """MTP step 0 localizes once; compaction and skip_topk steps reuse it.

    The old path localized the global rows on every pass. Localization is a
    per-row function, so it commutes with the row compaction of
    compact_topk_indices, and the reused rows give bitwise the same partial
    output and LSE on every pass.
    """
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    torch.manual_seed(7)
    page, heads, kv_heads, head_dim, seq_len = 16, 8, 2, 64, 256
    local_blocks = seq_len // world // page + 1
    shape = (local_blocks, page, kv_heads, head_dim)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    table = torch.arange(local_blocks, dtype=torch.int32, device="cuda")[None]
    # Step 0 selects for 6 query rows; compaction keeps rows 5, 1 and 3.
    rows = [list(range(r, seq_len, r + 2)) for r in range(6)]
    rows[1] = [0]  # an empty owner on every rank but 0
    width = max(len(r) for r in rows)
    selection = _pack(rows, width).cuda()
    kept = torch.tensor([5, 1, 3], device="cuda")
    query = torch.randn(3, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    token_to_req = torch.zeros(3, dtype=torch.int32, device="cuda")

    def attend(local):
        return qsa_sparse_paged_attention(
            query, key, value, local, table, token_to_req, False, return_lse=True
        )

    for rank in range(world):
        # Old path: the buffer keeps global rows; every pass localizes them.
        old_buffer = selection.clone()
        old_buffer[:3].copy_(old_buffer.index_select(0, kept))
        expected = [
            attend(
                qsa_localize_dcp_indices(
                    old_buffer[:3],
                    torch.empty_like(old_buffer[:3]),
                    world,
                    rank,
                    interleave,
                )
            )
            for _ in range(3)
        ]
        # New path: localize once at selection, compact, then reuse.
        buffer = selection.clone()
        _localize_in_place(buffer, world, rank, interleave)
        buffer[:3].copy_(buffer.index_select(0, kept))
        for (out, lse), (expected_out, expected_lse) in zip(
            (attend(buffer[:3]) for _ in range(3)), expected
        ):
            assert torch.equal(out, expected_out)
            assert torch.equal(lse, expected_lse)


# --- indexer wiring ---------------------------------------------------------


def _select_with_indexer(
    monkeypatch, *, skip_topk, dcp_world_size, selector_world_size=1
):
    """Run QSAIndexer.forward with fake caches; record the DCP localization.

    Returns (selection buffer, impl calls, kernel calls, select calls).
    """
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.qsa_cache import QSADCPShard
    from vllm.models.qwen4_exp.nvidia import indexer_qsa
    from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp as qsa_dcp_ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp_selector
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops

    Qwen4ExpQSAFlashAttentionImpl = _qsa_module().Qwen4ExpQSAFlashAttentionImpl

    num_tokens, token_topk, compress_ratio = 2, 4, 2
    impl_calls, kernel_calls, select_calls = [], [], []
    impl = SimpleNamespace(
        dcp_world_size=dcp_world_size,
        dcp_rank=0,
        cp_kv_cache_interleave_size=1,
        dcp_token_shard=QSADCPShard(0, dcp_world_size, 1),
    )

    def localize_dcp_selection_(selection):
        impl_calls.append(selection)
        Qwen4ExpQSAFlashAttentionImpl.localize_dcp_selection_(impl, selection)

    impl.localize_dcp_selection_ = localize_dcp_selection_
    monkeypatch.setattr(
        qsa_dcp_ops,
        "qsa_localize_dcp_indices",
        lambda *args: kernel_calls.append(args),
    )
    monkeypatch.setattr(indexer_qsa, "qsa_prepare", lambda *args, **kwargs: None)
    for name in (
        "qsa_select_paged_decode",
        "qsa_select_paged_prefill",
        "expand_qsa_block_indices",
    ):
        monkeypatch.setattr(
            qsa_indexer_ops,
            name,
            lambda *args, _name=name: select_calls.append(_name),
        )
    monkeypatch.setattr(
        qsa_dcp_selector,
        "expand_localize_qsa_block_indices",
        lambda *args: select_calls.append("expand_localize_qsa_block_indices"),
    )

    raw_metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        slot_mapping=torch.arange(num_tokens),
        block_table=torch.empty(0),
        query_start_loc=torch.arange(num_tokens + 1),
        logical_positions=torch.arange(num_tokens),
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        slot_mapping=torch.arange(num_tokens),
        k_work_metadata=torch.empty(0),
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        decode_query_len=1,
        visible_blocks=torch.ones(num_tokens, dtype=torch.int32),
        local_visible_blocks=torch.ones(num_tokens, dtype=torch.int32),
        selector_shard=(
            None if selector_world_size == 1 else QSADCPShard(0, selector_world_size, 1)
        ),
        block_table=torch.zeros(num_tokens, 1, dtype=torch.int32),
        query_start_loc=torch.arange(num_tokens + 1),
        logical_positions=torch.arange(num_tokens),
        max_query_len=1,
        max_seq_len=1,
    )
    indexer = SimpleNamespace(
        skip_topk=skip_topk,
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=1,
        indexer_dtype=torch.bfloat16,
        raw_key_cache=SimpleNamespace(
            kv_cache=torch.empty(0),
            rope_position_cache=None,
            rope_position_offset=0,
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=torch.empty(0)),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(0)),
        q_layernorm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.ones(1)),
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        packed_output_width=token_topk + 1,
    )
    attn = SimpleNamespace(
        use_fused_qsa_prepare=True,
        kv_cache=torch.empty(0, 1, 1, 2),
        kv_cache_dtype="auto",
        q_norm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(1)),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        impl=impl,
    )
    selection = _pack([[3, 1], [5, 2, 0]], token_topk)
    returned, _ = indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.zeros(num_tokens, 2),
        torch.arange(num_tokens),
        selection,
        attn=attn,
        qkv=torch.zeros(num_tokens, 4),
        slot_mapping=torch.arange(num_tokens),
    )
    assert returned is selection
    return selection, impl_calls, kernel_calls, select_calls


@pytest.mark.cpu_test
def test_indexer_localizes_each_new_selection_once(monkeypatch):
    selection, impl_calls, kernel_calls, select_calls = _select_with_indexer(
        monkeypatch, skip_topk=False, dcp_world_size=2
    )

    assert select_calls == ["qsa_select_paged_decode", "expand_qsa_block_indices"]
    assert len(impl_calls) == 1 and impl_calls[0] is selection
    assert len(kernel_calls) == 1
    source, out, world, rank, interleave = kernel_calls[0]
    assert source is selection and out is selection
    assert (world, rank, interleave) == (2, 0, 1)


@pytest.mark.cpu_test
def test_indexer_skip_topk_reuses_the_local_rows(monkeypatch):
    selection, impl_calls, kernel_calls, select_calls = _select_with_indexer(
        monkeypatch, skip_topk=True, dcp_world_size=2
    )

    assert select_calls == [] and impl_calls == [] and kernel_calls == []
    assert torch.equal(selection, _pack([[3, 1], [5, 2, 0]], 4))


@pytest.mark.cpu_test
def test_indexer_sharded_selector_localizes_in_the_fused_expand(monkeypatch):
    _, impl_calls, kernel_calls, select_calls = _select_with_indexer(
        monkeypatch, skip_topk=False, dcp_world_size=2, selector_world_size=2
    )

    assert select_calls == [
        "qsa_select_paged_decode",
        "expand_localize_qsa_block_indices",
    ]
    assert impl_calls == [] and kernel_calls == []


@pytest.mark.cpu_test
def test_indexer_does_not_localize_without_dcp(monkeypatch):
    selection, _, kernel_calls, select_calls = _select_with_indexer(
        monkeypatch, skip_topk=False, dcp_world_size=1
    )

    assert select_calls == ["qsa_select_paged_decode", "expand_qsa_block_indices"]
    assert kernel_calls == []
    assert torch.equal(selection, _pack([[3, 1], [5, 2, 0]], 4))


# --- one split with return_lse ----------------------------------------------


def _force_num_splits(monkeypatch, num_splits):
    """Make the reader use num_splits (at most its tile count)."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    select_config = qsa_ops._select_config

    def forced(*args, **kwargs):
        block_n, num_warps, num_tiles, _ = select_config(*args, **kwargs)
        return block_n, num_warps, num_tiles, min(num_splits, num_tiles)

    monkeypatch.setattr(qsa_ops, "_select_config", forced)


def _dcp_cache(cache_dtype, shape, k_scale, v_scale):
    """Random K/V of the requested storage: BF16, uint8 e4m3 or e4m3."""
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    if cache_dtype == "bf16":
        return key, value, None, None
    key = (key / k_scale).to(torch.float8_e4m3fn)
    value = (value / v_scale).to(torch.float8_e4m3fn)
    if cache_dtype == "uint8":
        key, value = key.view(torch.uint8), value.view(torch.uint8)
    return key, value, k_scale, v_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA kernels need CUDA")
@pytest.mark.parametrize("cache_dtype", READER_CACHE_DTYPES)
@pytest.mark.parametrize("with_empty_rows", [False, True])
def test_one_split_stores_the_bits_of_the_merge(
    monkeypatch, cache_dtype, with_empty_rows
):
    """With one split and return_lse the split kernel stores the output and
    the LSE itself. Both equal, bit for bit, the old path: fp32 partials of
    one split and the merge kernel."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    _force_num_splits(monkeypatch, 1)
    torch.manual_seed(5 + with_empty_rows)
    page, seq_len, heads, kv_heads, head_dim = 16, 512, 8, 2, 64
    blocks = seq_len // page
    key, value, k_scale, v_scale = _dcp_cache(
        cache_dtype, (blocks, page, kv_heads, head_dim), 0.75, 1.5
    )
    table = torch.arange(blocks, dtype=torch.int32, device="cuda")[None]
    rows = [list(range(r, seq_len, r + 1)) for r in range(7)]
    if with_empty_rows:
        rows[2] = []
        rows[5] = []
    selection = _pack(rows, max(len(r) for r in rows)).cuda()
    if with_empty_rows:
        selection[5, :8] = torch.arange(8, dtype=torch.int32)  # stale, count 0
    query = torch.randn(len(rows), heads, head_dim, dtype=torch.bfloat16).cuda()
    token_to_req = torch.zeros(len(rows), dtype=torch.int32, device="cuda")

    def attend():
        return qsa_ops.qsa_sparse_paged_attention(
            query,
            key,
            value,
            selection,
            table,
            token_to_req,
            False,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )

    direct_out, direct_lse = attend()
    monkeypatch.setattr(qsa_ops, "_merges_splits", lambda num_splits: True)
    merged_out, merged_lse = attend()

    assert torch.equal(direct_out.view(torch.int16), merged_out.view(torch.int16))
    assert torch.equal(direct_lse.view(torch.int32), merged_lse.view(torch.int32))
    if with_empty_rows:
        assert bool(torch.isneginf(direct_lse[[2, 5]]).all())
        assert bool((direct_out[[2, 5]] == 0).all())
    assert bool(torch.isfinite(direct_lse[[0, 1, 3]]).all())


# --- contracts that keep the gate from being applied twice -------------------


@pytest.mark.cpu_test
def test_localize_leaves_the_source_alone_without_dcp():
    """World size 1 still copies, so one call site covers both paths."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    packed = _pack([[5, 6], [7]], 8)
    source = packed.clone()
    local = torch.empty_like(packed)
    qsa_localize_dcp_indices(packed, local, 1, 0, 1)
    assert torch.equal(packed, source)
    assert torch.equal(local, source)


@pytest.mark.cpu_test
def test_the_kernel_wrapper_rejects_a_gate_with_return_lse():
    """The output gate must be applied after DCP merging, not by the kernel."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    query = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 2, 64, dtype=torch.bfloat16)
    indices = _pack([[0], [0]], 8)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    token_to_req = torch.zeros(2, dtype=torch.int32)
    args = (query, cache, cache, indices, block_table, token_to_req, False)

    with pytest.raises(ValueError, match="do not pass one"):
        qsa_ops.qsa_sparse_paged_attention(
            *args, output_gate=torch.zeros_like(query), return_lse=True
        )
    with pytest.raises(ValueError, match="requires an output gate"):
        qsa_ops.qsa_sparse_paged_attention(*args)


@pytest.mark.cpu_test
def test_empty_qsa_batch_returns_the_dcp_output_and_lse_pair():
    """A rank with no query tokens still supplies both DCP merge inputs."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    query = torch.empty((0, 4, 64), dtype=torch.bfloat16)
    cache = torch.empty((1, 16, 2, 64), dtype=torch.bfloat16)
    indices = torch.empty((0, 9), dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    token_to_req = torch.empty((0,), dtype=torch.int32)

    output, lse = qsa_sparse_paged_attention(
        query,
        cache,
        cache,
        indices,
        block_table,
        token_to_req,
        False,
        return_lse=True,
    )

    assert output.shape == query.shape
    assert output.dtype == torch.float32
    assert lse.shape == query.shape[:2]
    assert lse.dtype == torch.float32
    assert lse.device == query.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA kernels need CUDA")
@pytest.mark.parametrize(
    "world,interleave,seq_len,cache_dtype,k_scale,v_scale",
    [
        (2, 1, 64, torch.bfloat16, None, None),
        (4, 16, 200, torch.bfloat16, None, None),
        (4, 1, 200, torch.bfloat16, None, None),
        (2, 16, 64, torch.float8_e4m3fn, 0.75, 1.5),
        (4, 1, 64, torch.float8_e4m3fn, 0.75, 1.5),
        (4, 4, 64, torch.float8_e4m3fn, 0.75, 1.5),
        (2, 16, 64, torch.uint8, 0.75, 1.5),
    ],
)
def test_sharded_ranks_reproduce_the_single_rank_result(
    world, interleave, seq_len, cache_dtype, k_scale, v_scale
):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.dcp import _lse_weighted_combine

    if (
        cache_dtype == torch.float8_e4m3fn
        and not current_platform.has_device_capability(89)
    ):
        # Below sm_89 the reader takes the e4m3 bytes as uint8.
        cache_dtype = torch.uint8
    import_module("vllm.models.qwen4_exp.nvidia.model")
    qsa_backend = import_module(
        "vllm.models.qwen4_exp.nvidia.qsa"
    ).Qwen4ExpQSAFlashAttentionBackend
    assert qsa_backend.get_impl_cls().lse_base_on_e is False

    torch.manual_seed(1234 + seq_len)
    device = "cuda"
    page = 16
    num_kv_heads = 2
    head_dim = 64
    num_q_heads = 8

    full_blocks = (seq_len + page - 1) // page
    shape = (full_blocks, page, num_kv_heads, head_dim)
    k_source = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_source = torch.randn(shape, dtype=torch.bfloat16, device=device)
    if cache_dtype in (torch.float8_e4m3fn, torch.uint8):
        k_full = (k_source / k_scale).to(torch.float8_e4m3fn)
        v_full = (v_source / v_scale).to(torch.float8_e4m3fn)
        if cache_dtype == torch.uint8:
            k_full = k_full.view(torch.uint8)
            v_full = v_full.view(torch.uint8)
    else:
        k_full, v_full = k_source, v_source

    local_len = seq_len // world + page
    local_blocks = (local_len + page - 1) // page
    local_shape = (local_blocks, page, num_kv_heads, head_dim)
    k_rank = [
        torch.zeros(local_shape, dtype=cache_dtype, device=device) for _ in range(world)
    ]
    v_rank = [
        torch.zeros(local_shape, dtype=cache_dtype, device=device) for _ in range(world)
    ]
    for pos in range(seq_len):
        rank = _owner(pos, world, interleave)
        local_id = _local_id(pos, world, interleave)
        k_rank[rank][local_id // page, local_id % page] = k_full[
            pos // page, pos % page
        ]
        v_rank[rank][local_id // page, local_id % page] = v_full[
            pos // page, pos % page
        ]

    num_rows = 6
    selections = [
        list(range(row, seq_len, row + 2))[: seq_len // 2] for row in range(num_rows)
    ]
    selections[0] = [0]
    width = max(len(s) for s in selections)
    global_sel = _pack(selections, width, device)

    q = torch.randn(
        (num_rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)
    full_table = torch.arange(full_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    local_table = torch.arange(
        local_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    reference, reference_lse = qsa_sparse_paged_attention(
        q,
        k_full,
        v_full,
        global_sel,
        full_table,
        token_to_req,
        False,
        k_scale=k_scale,
        v_scale=v_scale,
        return_lse=True,
    )
    assert reference.dtype == torch.float32
    if cache_dtype == torch.uint8:
        native_out, native_lse = qsa_sparse_paged_attention(
            q,
            k_full.view(torch.float8_e4m3fn),
            v_full.view(torch.float8_e4m3fn),
            global_sel,
            full_table,
            token_to_req,
            False,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )
        torch.testing.assert_close(reference, native_out, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(reference_lse, native_lse, rtol=1e-3, atol=1e-3)

    outs, lses = [], []
    for rank in range(world):
        local_sel = qsa_localize_dcp_indices(
            global_sel, torch.empty_like(global_sel), world, rank, interleave
        )
        empty = local_sel[:, -1] <= 0
        if rank:
            assert bool(empty[0])
        out, lse = qsa_sparse_paged_attention(
            q,
            k_rank[rank],
            v_rank[rank],
            local_sel,
            local_table,
            token_to_req,
            False,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )
        if rank:
            assert torch.isneginf(lse[0]).all()
        outs.append(out.float())
        lses.append(lse.float())

    merged, merged_lse = _lse_weighted_combine(
        torch.stack(outs),
        torch.stack(lses),
        return_lse=True,
        is_lse_base_on_e=False,
    )

    torch.testing.assert_close(
        merged.to(reference.dtype), reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(merged_lse, reference_lse.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.cpu_test
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_qsa_impl_passes_the_mtp_interleave_check(monkeypatch, interleave):
    """The replicated QSA path accepts MTP at any interleave, not only the
    sharded selector."""
    from types import SimpleNamespace

    from vllm.v1.worker import cp_utils

    Qwen4ExpQSAFlashAttentionImpl = _qsa_module().Qwen4ExpQSAFlashAttentionImpl

    impl = SimpleNamespace(
        dcp_world_size=4,
        need_to_return_lse_for_decode=True,
        supports_mtp_with_cp_non_trivial_interleave_size=(
            Qwen4ExpQSAFlashAttentionImpl.supports_mtp_with_cp_non_trivial_interleave_size
        ),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=4,
            cp_kv_cache_interleave_size=interleave,
        ),
        speculative_config=object(),
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *args: {"qsa": SimpleNamespace(impl=impl)},
    )
    cp_utils.check_attention_cp_compatibility(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA kernels need CUDA")
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("world,interleave", [(4, 1), (4, 4), (4, 16), (2, 8)])
def test_mtp_verify_rows_match_the_single_rank_result(world, interleave, cache_dtype):
    """MTP verify rows (one request, consecutive positions, each row
    selecting causally up to its own position) give the single-rank result
    at every interleave that divides the block size."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.dcp import _lse_weighted_combine

    page, head_dim, num_kv_heads, num_q_heads = 16, 64, 2, 8
    if cache_dtype == torch.float8_e4m3fn:
        if current_platform.has_device_capability(89):
            storage, k_scale, v_scale = torch.float8_e4m3fn, 0.75, 1.5
        else:
            storage, k_scale, v_scale = torch.uint8, 0.75, 1.5
    else:
        storage, k_scale, v_scale = torch.bfloat16, None, None
    torch.manual_seed(world * 100 + interleave)
    device, seq_len, verify_len = "cuda", 300, 4
    full_blocks = (seq_len + page - 1) // page
    shape = (full_blocks, page, num_kv_heads, head_dim)
    k_full = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_full = torch.randn(shape, dtype=torch.bfloat16, device=device)
    if storage != torch.bfloat16:
        k_full = (k_full / k_scale).to(torch.float8_e4m3fn).view(storage)
        v_full = (v_full / v_scale).to(torch.float8_e4m3fn).view(storage)
    local_blocks = (seq_len // world + page + page - 1) // page
    local_shape = (local_blocks, page, num_kv_heads, head_dim)
    k_rank = [
        torch.zeros(local_shape, dtype=storage, device=device) for _ in range(world)
    ]
    v_rank = [
        torch.zeros(local_shape, dtype=storage, device=device) for _ in range(world)
    ]
    for pos in range(seq_len):
        r, lid = _owner(pos, world, interleave), _local_id(pos, world, interleave)
        k_rank[r][lid // page, lid % page] = k_full[pos // page, pos % page]
        v_rank[r][lid // page, lid % page] = v_full[pos // page, pos % page]

    # The verify span: the last verify_len positions; row i sees up to its own
    # position (the drafted tokens of the earlier rows included).
    positions = range(seq_len - verify_len, seq_len)
    selections = [list(range(0, p + 1, 3)) + [p] for p in positions]
    width = max(len(s) for s in selections)
    global_sel = _pack(selections, width, device)
    q = torch.randn(
        (verify_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    token_to_req = torch.zeros(verify_len, dtype=torch.int32, device=device)
    table = torch.arange(full_blocks, dtype=torch.int32, device=device)[None]
    local_table = torch.arange(local_blocks, dtype=torch.int32, device=device)[None]
    scales = {"k_scale": k_scale, "v_scale": v_scale}

    reference, reference_lse = qsa_sparse_paged_attention(
        q,
        k_full,
        v_full,
        global_sel,
        table,
        token_to_req,
        False,
        return_lse=True,
        **scales,
    )
    outs, lses = [], []
    for rank in range(world):
        local_sel = qsa_localize_dcp_indices(
            global_sel, torch.empty_like(global_sel), world, rank, interleave
        )
        out, lse = qsa_sparse_paged_attention(
            q,
            k_rank[rank],
            v_rank[rank],
            local_sel,
            local_table,
            token_to_req,
            False,
            return_lse=True,
            **scales,
        )
        outs.append(out.float())
        lses.append(lse)
    merged, merged_lse = _lse_weighted_combine(
        torch.stack(outs), torch.stack(lses), return_lse=True, is_lse_base_on_e=False
    )
    torch.testing.assert_close(
        merged.to(reference.dtype), reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(merged_lse, reference_lse.float(), rtol=1e-3, atol=1e-3)
