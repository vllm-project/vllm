# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DCP-aware sparse indexer.

Validates the global<->local position remap math and the global top-k
reconstruction: for interleaved DCP sharding, the *union* over ranks of the
per-rank `_dcp_global_topk_remap` output (mapped back to global positions)
must equal the true global top-k of the concatenated full logits.

These tests exercise the pure tensor logic in
`vllm/model_executor/layers/sparse_attn_indexer.py` (no GPU, no real
distributed communicator): `get_dcp_group` is monkeypatched to a fake that
emulates all_gather by concatenating all ranks' pre-computed contributions.
"""

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as idx_mod
import vllm.v1.attention.backends.mla.flashmla_sparse as sparse_mla_mod
from vllm.model_executor.layers.sparse_attn_indexer import (
    _dcp_finalize_topk_remap,
    _dcp_pack_topk_candidates,
    _global_to_local_position,
    _local_to_global_position,
    _use_persistent_topk_decode,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    _dcp_local_indexer_seq_lens,
    _decode_topk_max_seq_len,
    _localize_decode_seq_lens_inplace,
    split_indexer_prefill_chunks,
)


class _FakeDCPGroup:
    """Single-process stand-in for the DCP group coordinator.

    `set_all` pre-stashes every rank's packed (global_pos, score_bits)
    candidate tensor. `all_gather` then concatenates them in rank order,
    emulating the single packed all-gather used by the DCP remap.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self._packed: list[torch.Tensor] | None = None
        self.all_gather_calls = 0

    def set_all(self, pos_list, scores_list):
        self._packed = []
        for pos, scores in zip(pos_list, scores_list):
            packed = torch.empty(
                (pos.shape[0], 2, pos.shape[1]),
                dtype=torch.int32,
                device=pos.device,
            )
            packed[:, 0, :].copy_(pos)
            packed[:, 1, :].copy_(
                scores.to(torch.float32).contiguous().view(torch.int32)
            )
            self._packed.append(packed)

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.all_gather_calls += 1
        assert self._packed is not None
        assert dim == 2
        assert tensor.dtype == torch.int32
        assert tensor.ndim == 3
        assert tensor.shape[1] == 2
        assert torch.equal(tensor, self._packed[self.rank_in_group])
        return torch.cat(self._packed, dim=dim)


def _local_to_global(local_idx, rank, world_size, interleave):
    return _local_to_global_position(local_idx, rank, world_size, interleave)


def _make_common_metadata(
    seq_lens_cpu: torch.Tensor,
    max_seq_len: int,
    *,
    dcp_local_seq_lens_cpu: torch.Tensor | None = None,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    exact_seq_lens_cpu: torch.Tensor | None = None,
) -> CommonAttentionMetadata:
    num_reqs = seq_lens_cpu.shape[0]
    query_start_loc_cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=max_seq_len,
        block_table_tensor=torch.empty((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.empty(num_reqs, dtype=torch.int64),
        dcp_local_seq_lens_cpu=dcp_local_seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=exact_seq_lens_cpu,
    )


def test_decode_persistent_topk_enabled_for_dcp(monkeypatch):
    monkeypatch.setattr(idx_mod.current_platform, "is_cuda", lambda: True)

    assert _use_persistent_topk_decode(512)
    assert _use_persistent_topk_decode(1024)
    assert _use_persistent_topk_decode(2048)
    assert not _use_persistent_topk_decode(256)


def test_decode_topk_max_seq_len_uses_dcp_local_lengths():
    common = _make_common_metadata(
        torch.tensor([100, 80, 0], dtype=torch.int32),
        max_seq_len=100,
        dcp_local_seq_lens_cpu=torch.tensor([13, 11, 0], dtype=torch.int32),
    )

    got = _decode_topk_max_seq_len(
        common,
        num_decodes=2,
        compress_ratio=4,
        dcp_world_size=8,
        dcp_rank=7,
        cp_interleave_size=16,
    )

    assert got == 3


def test_decode_topk_max_seq_len_derives_dcp_local_upper_bound():
    common = _make_common_metadata(
        torch.tensor([100, 65], dtype=torch.int32),
        max_seq_len=100,
        seq_lens_cpu_upper_bound=torch.tensor([100, 65], dtype=torch.int32),
    )

    got = _decode_topk_max_seq_len(
        common,
        num_decodes=2,
        compress_ratio=1,
        dcp_world_size=4,
        dcp_rank=1,
        cp_interleave_size=8,
    )

    assert got == 24


def _make_decode_builder(next_n: int) -> DeepseekV32IndexerMetadataBuilder:
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_seq_lens_buffer = torch.zeros(16, dtype=torch.int32)
    builder.decode_lens_buffer = torch.zeros(16, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros((16, 2), dtype=torch.int32)
    builder.arange_buffer = torch.arange(16, dtype=torch.int32)
    builder.offsets_buffer = torch.arange(next_n, dtype=torch.int32)
    builder.dcp_world_size = 2
    builder.dcp_rank = 1
    builder.cp_kv_cache_interleave_size = 2
    return builder


def test_prepare_decode_tensors_expands_global_then_localizes_for_dcp_spec():
    builder = _make_decode_builder(next_n=3)
    global_seq_lens = torch.tensor([10, 11], dtype=torch.int32)
    dcp_local_seq_lens = _dcp_local_indexer_seq_lens(
        global_seq_lens,
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=1,
        cp_interleave_size=2,
    )
    decode_lens = torch.tensor([3, 3], dtype=torch.int32)

    seq_lens, _, _, batch_size, requires_padding = builder._prepare_decode_tensors(
        seq_lens=global_seq_lens,
        dcp_local_seq_lens=dcp_local_seq_lens,
        block_table=torch.arange(4, dtype=torch.int32).view(2, 2),
        decode_lens=decode_lens.clone(),
        decode_lens_cpu=decode_lens,
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=6,
        use_native=True,
        next_n=3,
        max_decode_len=3,
    )

    global_per_token = torch.tensor(
        [
            [8, 9, 10],
            [9, 10, 11],
        ],
        dtype=torch.int32,
    )
    expected = _dcp_local_indexer_seq_lens(
        global_per_token.flatten(),
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=1,
        cp_interleave_size=2,
    ).view(2, 3)
    wrong_old_math = (
        dcp_local_seq_lens.view(2, 1) - 3 + torch.arange(1, 4, dtype=torch.int32)
    )

    assert batch_size == 2
    assert not requires_padding
    assert torch.equal(seq_lens, expected)
    assert not torch.equal(seq_lens, wrong_old_math)


def test_prepare_decode_tensors_uses_precomputed_dcp_local_for_plain_decode():
    builder = _make_decode_builder(next_n=1)
    global_seq_lens = torch.tensor([10, 11], dtype=torch.int32)
    dcp_local_seq_lens = _dcp_local_indexer_seq_lens(
        global_seq_lens,
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=1,
        cp_interleave_size=2,
    )
    decode_lens = torch.tensor([1, 1], dtype=torch.int32)

    seq_lens, _, _, batch_size, requires_padding = builder._prepare_decode_tensors(
        seq_lens=global_seq_lens,
        dcp_local_seq_lens=dcp_local_seq_lens,
        block_table=torch.arange(4, dtype=torch.int32).view(2, 2),
        decode_lens=decode_lens.clone(),
        decode_lens_cpu=decode_lens,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=2,
        use_native=True,
        next_n=1,
        max_decode_len=1,
    )

    assert batch_size == 2
    assert not requires_padding
    assert torch.equal(seq_lens, dcp_local_seq_lens)


@pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda() and torch.cuda.is_available()),
    reason="CUDA Triton kernel test requires a CUDA Triton runtime",
)
def test_localize_decode_seq_lens_inplace_cuda_matches_reference():
    seq_lens_cpu = torch.tensor(
        [
            [8, 9, 10],
            [9, 10, 11],
        ],
        dtype=torch.int32,
    )
    expected = _dcp_local_indexer_seq_lens(
        seq_lens_cpu.flatten(),
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=1,
        cp_interleave_size=2,
    ).view_as(seq_lens_cpu)
    seq_lens = seq_lens_cpu.cuda()
    data_ptr = seq_lens.data_ptr()

    _localize_decode_seq_lens_inplace(
        seq_lens,
        dcp_world_size=2,
        dcp_rank=1,
        cp_interleave_size=2,
    )

    assert seq_lens.data_ptr() == data_ptr
    assert torch.equal(seq_lens.cpu(), expected)


def test_pack_topk_candidates_preserves_score_bits_and_row_starts():
    topk_indices = torch.tensor(
        [
            [0, 2, -1, 1],
            [1, -1, 0, 3],
        ],
        dtype=torch.int32,
    )
    logits = torch.tensor(
        [
            [-100.0, -100.0, 1.0, 3.0, 5.0, 7.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0, -100.0, 2.0, 4.0, 6.0, 8.0],
        ],
        dtype=torch.float32,
    )
    row_starts = torch.tensor([2, 4], dtype=torch.int32)

    packed = _dcp_pack_topk_candidates(
        topk_indices,
        logits,
        topk_indices.shape[1],
        rank=1,
        world_size=3,
        interleave=2,
        row_starts=row_starts,
    )

    expected_pos = _local_to_global(
        torch.clamp(topk_indices, min=0),
        rank=1,
        world_size=3,
        interleave=2,
    )
    expected_pos = torch.where(
        topk_indices < 0, expected_pos.new_full((), -1), expected_pos
    )
    expected_scores = torch.tensor(
        [
            [1.0, 5.0, float("-inf"), 3.0],
            [4.0, float("-inf"), 2.0, 8.0],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(packed[:, 0, :], expected_pos)
    torch.testing.assert_close(packed[:, 1, :].view(torch.float32), expected_scores)


def test_finalize_topk_remap_matches_reference():
    all_candidates = torch.empty((2, 2, 6), dtype=torch.int32)
    all_candidates[:, 0, :] = torch.tensor(
        [
            [0, 2, 3, 4, 6, -1],
            [1, 5, 6, 7, 8, -1],
        ],
        dtype=torch.int32,
    )
    all_candidates[:, 1, :] = 0
    selected = torch.tensor([[1, 2, 4], [0, 3, 5]], dtype=torch.int64)
    out = torch.zeros((2, 3), dtype=torch.int32)

    _dcp_finalize_topk_remap(
        all_candidates,
        selected,
        out,
        topk_tokens=3,
        rank=1,
        world_size=2,
        interleave=2,
    )

    assert out.tolist() == [[0, 1, 2], [-1, 3, -1]]


@pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda() and torch.cuda.is_available()),
    reason="CUDA Triton kernel test requires a CUDA Triton runtime",
)
def test_triton_pack_topk_candidates_matches_cpu_reference():
    topk_indices_cpu = torch.tensor(
        [
            [0, 3, -1, 1],
            [2, -1, 0, 4],
        ],
        dtype=torch.int32,
    )
    logits_cpu = torch.tensor(
        [
            [-100.0, 1.0, 2.0, 3.0, 4.0, -100.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, -100.0],
        ],
        dtype=torch.float32,
    )
    row_starts_cpu = torch.tensor([1, 0], dtype=torch.int32)
    expected = _dcp_pack_topk_candidates(
        topk_indices_cpu,
        logits_cpu,
        topk_indices_cpu.shape[1],
        rank=1,
        world_size=4,
        interleave=2,
        row_starts=row_starts_cpu,
    )

    got = _dcp_pack_topk_candidates(
        topk_indices_cpu.cuda(),
        logits_cpu.cuda(),
        topk_indices_cpu.shape[1],
        rank=1,
        world_size=4,
        interleave=2,
        row_starts=row_starts_cpu.cuda(),
    )

    assert torch.equal(got.cpu(), expected)


@pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda() and torch.cuda.is_available()),
    reason="CUDA Triton kernel test requires a CUDA Triton runtime",
)
def test_triton_finalize_topk_remap_matches_cpu_reference():
    all_candidates_cpu = torch.empty((2, 2, 8), dtype=torch.int32)
    all_candidates_cpu[:, 0, :] = torch.tensor(
        [
            [0, 2, 3, 4, 6, 8, 9, -1],
            [1, 5, 6, 7, 8, 10, 11, -1],
        ],
        dtype=torch.int32,
    )
    all_candidates_cpu[:, 1, :] = 0
    selected_cpu = torch.tensor([[1, 2, 4, 6], [0, 3, 5, 7]], dtype=torch.int64)
    expected = torch.zeros((2, 4), dtype=torch.int32)
    _dcp_finalize_topk_remap(
        all_candidates_cpu,
        selected_cpu,
        expected,
        topk_tokens=4,
        rank=1,
        world_size=2,
        interleave=2,
    )

    got = torch.zeros_like(expected, device="cuda")
    _dcp_finalize_topk_remap(
        all_candidates_cpu.cuda(),
        selected_cpu.cuda(),
        got,
        topk_tokens=4,
        rank=1,
        world_size=2,
        interleave=2,
    )

    assert torch.equal(got.cpu(), expected)


@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_position_roundtrip(interleave, world_size):
    total = 3 * interleave * world_size
    g = torch.arange(total, dtype=torch.int32)
    owner = (g // interleave) % world_size
    for rank in range(world_size):
        present = g[owner == rank]
        for local_idx in range(present.numel()):
            gg = _local_to_global(
                torch.tensor([local_idx]), rank, world_size, interleave
            )
            # this global position should map back to local_idx on rank `rank`
            assert (
                _global_to_local_position(gg, interleave, world_size).item()
                == local_idx
            )
        # cross-check the owner implied by local->global matches `rank`
        gg_all = _local_to_global(
            torch.arange(present.numel(), dtype=torch.int32),
            rank,
            world_size,
            interleave,
        )
        assert (((gg_all // interleave) % world_size) == rank).all()


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [1, 2, 2, 2]),
        (1, [0, 0, 1, 2]),
    ],
)
def test_dcp_prefill_visible_lengths_use_global_positions(rank, expected):
    visible_global_lens = torch.arange(1, 5, dtype=torch.int32)

    got = _dcp_local_indexer_seq_lens(
        visible_global_lens,
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=rank,
        cp_interleave_size=2,
    )

    total = _dcp_local_indexer_seq_lens(
        torch.tensor([4], dtype=torch.int32),
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=rank,
        cp_interleave_size=2,
    ).item()
    assert got.tolist() == expected
    assert torch.all(got >= 0)
    assert torch.all(got <= total)


def test_dcp_prefill_visible_lengths_support_compressed_indexer_cache():
    visible_global_lens = torch.arange(1, 17, dtype=torch.int32)

    got = _dcp_local_indexer_seq_lens(
        visible_global_lens,
        compress_ratio=4,
        dcp_world_size=2,
        dcp_rank=0,
        cp_interleave_size=2,
    )

    assert got.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]


def _chunk_specs_as_tuples(chunks):
    return [(req.start, req.stop, query.start, query.stop) for req, query in chunks]


def test_dcp_prefill_chunking_uses_all_rank_max_lengths_for_collective_order():
    seq_lens_cpu = torch.tensor([1, 3], dtype=torch.int32)
    query_lens_cpu = torch.tensor([1, 1], dtype=torch.int32)
    workspace_size = 2
    max_logits_bytes = 1024 * 1024

    all_local = _dcp_local_indexer_seq_lens(
        seq_lens_cpu,
        compress_ratio=1,
        dcp_world_size=2,
        dcp_rank=None,
        cp_interleave_size=1,
    )

    rank_specs = [
        _chunk_specs_as_tuples(
            split_indexer_prefill_chunks(
                all_local[:, rank],
                query_lens_cpu,
                workspace_size,
                max_logits_bytes,
            )
        )
        for rank in range(2)
    ]
    safe_specs = _chunk_specs_as_tuples(
        split_indexer_prefill_chunks(
            all_local.max(dim=1).values,
            query_lens_cpu,
            workspace_size,
            max_logits_bytes,
        )
    )

    assert rank_specs[0] != rank_specs[1]
    assert safe_specs == [(0, 1, 0, 1), (1, 2, 0, 1)]


@pytest.mark.parametrize(
    ("input_heads", "kernel_heads", "output_heads"),
    [
        (4, 64, 4),
        (64, 64, 64),
    ],
)
def test_bf16_sparse_prefill_padding_uses_input_head_count(
    input_heads, kernel_heads, output_heads, monkeypatch
):
    impl = object.__new__(sparse_mla_mod.FlashMLASparseImpl)
    impl.num_heads = 4
    impl.prefill_padding = 64
    impl.softmax_scale = 1.0

    seen = {}

    def fake_flash_mla_sparse_fwd(q, kv_cache, topk_indices, softmax_scale):
        seen["q_shape"] = tuple(q.shape)
        seen["kv_shape"] = tuple(kv_cache.shape)
        seen["topk_shape"] = tuple(topk_indices.shape)
        seen["softmax_scale"] = softmax_scale
        out = torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype)
        max_logits = torch.zeros(q.shape[0], q.shape[1], dtype=q.dtype)
        lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32)
        return out, max_logits, lse

    monkeypatch.setattr(
        sparse_mla_mod,
        "flash_mla_sparse_fwd",
        fake_flash_mla_sparse_fwd,
    )

    q = torch.randn(3, input_heads, 576, dtype=torch.bfloat16)
    kv_cache = torch.randn(8, 576, dtype=torch.bfloat16)
    topk_indices = torch.zeros(3, 8, dtype=torch.int32)

    out, lse = sparse_mla_mod.FlashMLASparseImpl._bf16_flash_mla_kernel(
        impl, q, kv_cache, topk_indices
    )

    assert seen == {
        "q_shape": (3, kernel_heads, 576),
        "kv_shape": (8, 1, 576),
        "topk_shape": (3, 1, 8),
        "softmax_scale": 1.0,
    }
    assert out.shape == (3, output_heads, 512)
    assert lse.shape == (3, output_heads)


@pytest.mark.parametrize("interleave", [1, 2, 3])
@pytest.mark.parametrize("world_size", [2, 4])
def test_global_topk_reconstructs_reference(interleave, world_size, monkeypatch):
    torch.manual_seed(0)
    topk_tokens = 8
    num_rows = 4
    total_len = 6 * interleave * world_size
    full_logits = torch.randn(num_rows, total_len)

    # Shard full_logits to each rank per the interleave scheme. The shard's
    # column order is chosen so local index l maps to global via the formula.
    g = torch.arange(total_len)
    owner = (g // interleave) % world_size
    per_rank_logits = [
        full_logits[:, owner == rank].contiguous() for rank in range(world_size)
    ]

    # Pre-pass: compute each rank's (global_pos, score) candidates by running the
    # same local-topk + local->global logic the helper uses internally. When a
    # shard is shorter than topk_tokens, the real kernel returns only as many
    # valid indices as the shard holds and pads the rest with -1; emulate that.
    pos_list, scores_list = [], []
    for rank in range(world_size):
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        local_topk = lg.topk(k, dim=1).indices.to(torch.int32)
        pad = torch.full((local_topk.shape[0], topk_tokens - k), -1, dtype=torch.int32)
        local_topk = torch.cat([local_topk, pad], dim=1)
        invalid = local_topk < 0
        idx_safe = torch.clamp(local_topk, min=0)
        scores = lg.gather(1, idx_safe.long()).masked_fill(invalid, float("-inf"))
        gp = _local_to_global(idx_safe, rank, world_size, interleave)
        gp = torch.where(invalid, gp.new_full((), -1), gp)
        pos_list.append(gp.contiguous())
        scores_list.append(scores.contiguous())

    # Run the real helper once per rank with a fake group that returns the full
    # concatenation, capturing each rank's final local-index buffer.
    per_rank_final = []
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)

        topk_buf = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)
        # Seed the buffer with rank's local top-k (what the kernel would write).
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        seed = lg.topk(k, dim=1).indices.to(torch.int32)
        topk_buf[:, :k] = seed
        idx_mod._dcp_global_topk_remap(topk_buf, lg, topk_tokens, interleave)
        assert fake.all_gather_calls == 1
        per_rank_final.append(topk_buf)

    # Reference: global top-k of the FULL logits, as sets per row.
    ref = full_logits.topk(topk_tokens, dim=1).indices
    for row in range(num_rows):
        union = set()
        for rank in range(world_size):
            for local_idx in per_rank_final[rank][row].tolist():
                if local_idx < 0:
                    continue
                union.add(
                    _local_to_global(
                        torch.tensor([local_idx]), rank, world_size, interleave
                    ).item()
                )
        assert union == set(ref[row].tolist()), (
            row,
            sorted(union),
            sorted(ref[row].tolist()),
        )


def test_prefill_global_topk_uses_row_starts(monkeypatch):
    interleave, world_size, topk_tokens = 1, 2, 2
    full_logits = torch.tensor(
        [
            [1.0, 5.0, 2.0, 4.0, 3.0, 0.0],
            [0.1, 10.0, 0.2, 9.0, 0.3, 8.0],
        ],
        dtype=torch.float32,
    )
    num_rows = full_logits.shape[0]
    total_len = full_logits.shape[1]
    g = torch.arange(total_len)
    owner = (g // interleave) % world_size

    per_rank_logits = []
    row_starts_per_rank = []
    seed_topk = []
    pos_list, scores_list = [], []
    for rank in range(world_size):
        local_logits = full_logits[:, owner == rank].contiguous()
        local_len = local_logits.shape[1]
        row_starts = torch.arange(num_rows, dtype=torch.int32) * local_len
        logits = full_logits.new_full((num_rows, num_rows * local_len), float("-inf"))
        for row, row_start in enumerate(row_starts.tolist()):
            logits[row, row_start : row_start + local_len] = local_logits[row]

        local_topk = local_logits.topk(topk_tokens, dim=1).indices.to(torch.int32)
        idx_safe = torch.clamp(local_topk, min=0)
        score_idx = idx_safe.long() + row_starts.view(-1, 1).long()
        scores = logits.gather(1, score_idx)
        gp = _local_to_global(idx_safe, rank, world_size, interleave)

        per_rank_logits.append(logits)
        row_starts_per_rank.append(row_starts)
        seed_topk.append(local_topk)
        pos_list.append(gp.contiguous())
        scores_list.append(scores.contiguous())

    per_rank_final = []
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)

        topk_buf = seed_topk[rank].clone()
        idx_mod._dcp_global_topk_remap(
            topk_buf,
            per_rank_logits[rank],
            topk_tokens,
            interleave,
            row_starts=row_starts_per_rank[rank],
        )
        assert fake.all_gather_calls == 1
        per_rank_final.append(topk_buf)

    ref = full_logits.topk(topk_tokens, dim=1).indices
    for row in range(num_rows):
        union = set()
        for rank in range(world_size):
            for local_idx in per_rank_final[rank][row].tolist():
                if local_idx >= 0:
                    union.add(
                        _local_to_global(
                            torch.tensor([local_idx]), rank, world_size, interleave
                        ).item()
                    )
        assert union == set(ref[row].tolist())


def test_global_topk_remap_allows_empty_local_rank(monkeypatch):
    interleave, world_size, topk_tokens = 1, 2, 2
    pos_list = [
        torch.tensor([[0, 2]], dtype=torch.int32),
        torch.tensor([[-1, -1]], dtype=torch.int32),
    ]
    scores_list = [
        torch.tensor([[5.0, 4.0]], dtype=torch.float32),
        torch.full((1, topk_tokens), float("-inf"), dtype=torch.float32),
    ]

    per_rank_final = []
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)

        topk_buf = (
            torch.tensor([[0, 1]], dtype=torch.int32)
            if rank == 0
            else torch.full((1, topk_tokens), -1, dtype=torch.int32)
        )
        logits = torch.tensor([[5.0, 4.0]], dtype=torch.float32) if rank == 0 else None
        idx_mod._dcp_global_topk_remap(
            topk_buf,
            logits,
            topk_tokens,
            interleave,
        )
        assert fake.all_gather_calls == 1
        per_rank_final.append(topk_buf)

    assert per_rank_final[0].tolist() == [[0, 1]]
    assert per_rank_final[1].tolist() == [[-1, -1]]


def test_global_topk_ownership_partition(monkeypatch):
    """Every selected global position is owned by exactly one rank's output."""
    interleave, world_size, topk_tokens = 2, 2, 6
    num_rows = 2
    total_len = 4 * interleave * world_size
    full_logits = torch.randn(num_rows, total_len)
    g = torch.arange(total_len)
    owner = (g // interleave) % world_size
    per_rank_logits = [
        full_logits[:, owner == r].contiguous() for r in range(world_size)
    ]
    pos_list, scores_list = [], []
    for rank in range(world_size):
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        local_topk = lg.topk(k, dim=1).indices.to(torch.int32)
        pad = torch.full((local_topk.shape[0], topk_tokens - k), -1, dtype=torch.int32)
        local_topk = torch.cat([local_topk, pad], dim=1)
        idx_safe = torch.clamp(local_topk, min=0)
        scores = lg.gather(1, idx_safe.long()).masked_fill(
            local_topk < 0, float("-inf")
        )
        gp = _local_to_global(idx_safe, rank, world_size, interleave)
        gp = torch.where(local_topk < 0, gp.new_full((), -1), gp)
        pos_list.append(gp.contiguous())
        scores_list.append(scores.contiguous())
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        buf = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)
        buf[:, :k] = lg.topk(k, dim=1).indices.to(torch.int32)
        idx_mod._dcp_global_topk_remap(buf, lg, topk_tokens, interleave)
        assert fake.all_gather_calls == 1
        # owned positions get a real (>=0) local idx; others are -1
        for local_idx in buf.flatten().tolist():
            if local_idx >= 0:
                gg = _local_to_global(
                    torch.tensor([local_idx]), rank, world_size, interleave
                ).item()
                assert (gg // interleave) % world_size == rank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
