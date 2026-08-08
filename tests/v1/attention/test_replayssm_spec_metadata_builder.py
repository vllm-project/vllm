# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM speculative-decode ring cursors in BaseMambaAttentionMetadataBuilder.

The cursors are block-keyed and advanced on-device by commit_replayssm_spec, so
these tests drive builder.build() and read the resulting buffers back. Each case
builds one CommonAttentionMetadata and reuses it, because
create_common_attn_metadata draws fresh block ids on every call.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
SPEC_QUERY_LEN = 1 + NUM_SPEC
BUFFER_LEN = 8
# L = B + spec_query_len is both the logical threshold and physical ring.
FLUSH_THRESHOLD = BUFFER_LEN + SPEC_QUERY_LEN
RING_LEN = FLUSH_THRESHOLD

NHEADS, HEAD_DIM, DSTATE, NGROUPS = 2, 4, 8, 1
CONV_DIM = NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReplaySSM spec cursors are Triton kernels"
)


def _make_spec_mamba_spec() -> MambaSpec:
    # Standard and speculative ReplaySSM share the head-major x/dt/B layout.
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            (CONV_DIM, 3),
            (NHEADS, HEAD_DIM, DSTATE),
            (NHEADS, RING_LEN, HEAD_DIM),
            (NHEADS, RING_LEN),
            (NGROUPS, RING_LEN, DSTATE),
        ),
        dtypes=(torch.float32,),
    )


def _create_spec_builder(
    buffer_len: int = BUFFER_LEN,
    full_cuda_graph: bool = False,
) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        num_gpu_blocks=1024,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    # Set after construction so ReplaySSM validation does not run on the mock.
    vllm_config.cache_config.use_replayssm_spec = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    return MockMambaBuilder(_make_spec_mamba_spec(), ["layer0"], vllm_config, DEVICE)


def _make_common(
    seq_lens: list[int],
    decode_base: list[int],
    query_lens: list[int] | None = None,
):
    n = len(seq_lens)
    query_lens = query_lens if query_lens is not None else [SPEC_QUERY_LEN] * n
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    return create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False] * n, dtype=torch.bool),
        replayssm_decode_base_cpu=torch.tensor(decode_base, dtype=torch.int32),
    )


def _build(builder: MockMambaBuilder, common, num_accepted: list[int]):
    return builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor(
            num_accepted, dtype=torch.int32, device=DEVICE
        ),
    )


def _cursors(builder: MockMambaBuilder, blocks: list[int]):
    idx = torch.tensor(blocks, device=DEVICE)
    return (
        builder.spec_write_pos[idx].tolist(),
        builder.spec_post_origin[idx].tolist(),
        builder.spec_is_flush[idx].tolist(),
    )


def _seed(builder: MockMambaBuilder, blocks: list[int], write_pos: int, is_flush: int):
    for block in blocks:
        builder.spec_write_pos[block] = write_pos
        builder.spec_post_origin[block] = 0
        builder.spec_is_flush[block] = is_flush


def _prime(builder: MockMambaBuilder, common) -> list[int]:
    """Allocate the cursor buffers and return this batch's physical block ids."""
    meta = _build(builder, common, [1] * common.num_reqs)
    return meta.state_indices_tensor_d[:, 0].tolist()


def test_commit_advances_write_pos_by_accepted():
    builder = _create_spec_builder()
    # num_computed = 120 - 4 = 116 > decode_base, so no reset fires.
    common = _make_common([120], [100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    _build(builder, common, [3])

    write_pos, post_origin, _ = _cursors(builder, blocks)
    assert write_pos == [5]
    assert post_origin == [0]


def test_rejected_drafts_are_not_committed():
    """Rollback: only accepted tokens advance the cursor, the rest are dropped."""
    builder = _create_spec_builder()
    common = _make_common([120], [100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    # One accepted out of a 4-token window.
    _build(builder, common, [1])

    assert _cursors(builder, blocks)[0] == [3]


def test_flush_advances_origin_and_restarts_write_pos():
    builder = _create_spec_builder()
    common = _make_common([120], [100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=5, is_flush=1)

    _build(builder, common, [2])

    write_pos, post_origin, _ = _cursors(builder, blocks)
    # The flush consumed the 5 committed history tokens, so the ring restarts
    # past them and only the freshly accepted tokens remain.
    assert post_origin == [5]
    assert write_pos == [2]


def test_early_flush_keeps_room_for_a_full_window():
    """is_flush is set one window early: write_pos + 2 * spec_query_len > L."""
    builder = _create_spec_builder()
    common = _make_common([120], [100])
    blocks = _prime(builder, common)

    # write_pos 3 -> +1 = 4; 4 + 8 = 12 is not > 12, so no flush yet.
    _seed(builder, blocks, write_pos=3, is_flush=0)
    _build(builder, common, [1])
    assert _cursors(builder, blocks)[2] == [0]

    # write_pos 4 -> +1 = 5; 5 + 8 = 13 > 12, so the next step must flush.
    _seed(builder, blocks, write_pos=4, is_flush=0)
    _build(builder, common, [1])
    assert _cursors(builder, blocks)[2] == [1]


def test_first_decode_resets_a_recycled_block():
    builder = _create_spec_builder()
    # num_computed == decode_base marks the request's first verify step.
    common = _make_common([104], [100])
    blocks = _prime(builder, common)
    # Stale cursors left behind by whoever held this block before.
    _seed(builder, blocks, write_pos=6, is_flush=1)

    _build(builder, common, [1])

    assert _cursors(builder, blocks) == ([0], [0], [0])


def test_resumed_request_reanchors_on_decode_base():
    """Preemption: decode_base moves to the resume point, which re-fires the
    reset so the recomputed request does not inherit its old cursors."""
    builder = _create_spec_builder()
    # Resumed at 116 tokens of context: num_computed == decode_base again.
    common = _make_common([120], [116])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=7, is_flush=1)

    _build(builder, common, [1])

    write_pos, post_origin, _ = _cursors(builder, blocks)
    assert (write_pos, post_origin) == ([0], [0])


def test_steady_state_row_is_not_reset():
    builder = _create_spec_builder()
    # num_computed (116) > decode_base (100): mid-generation, keep the ring.
    common = _make_common([120], [100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    _build(builder, common, [1])

    assert _cursors(builder, blocks)[0] == [3]


def test_leftover_prompt_chunk_resets_and_forces_a_flush():
    """A final single-token prompt chunk is reclassified as a decode, so it runs
    the spec kernel one token short of decode_base. It must reset the ring and
    flush, or the checkpoint never advances past that token and the next step's
    reset drops it."""
    builder = _create_spec_builder()
    # num_computed = 100 - 1 = 99, one short of decode_base.
    common = _make_common([100], [100], query_lens=[1])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=6, is_flush=0)

    _build(builder, common, [1])

    write_pos, post_origin, is_flush = _cursors(builder, blocks)
    assert (write_pos, post_origin) == ([0], [0])
    assert is_flush == [2]


def test_forced_checkpoint_rejects_a_multi_token_query():
    """The forced checkpoint consumes the current token, so it is only valid
    for the single-token prompt tail it was introduced to handle."""
    builder = _create_spec_builder()
    common = _make_common([100], [100], query_lens=[SPEC_QUERY_LEN])

    with pytest.raises(ValueError, match="single-token query"):
        _build(builder, common, [1])


def test_mixed_batch_resets_only_the_entering_rows():
    builder = _create_spec_builder()
    # Row 0 steady state, row 1 first decode, row 2 steady state.
    common = _make_common([120, 104, 124], [100, 100, 100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=4, is_flush=0)

    _build(builder, common, [2, 2, 2])

    assert _cursors(builder, blocks)[0] == [6, 0, 6]


def test_zero_accept_row_commits_nothing():
    builder = _create_spec_builder()
    common = _make_common([120], [100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=3, is_flush=0)

    _build(builder, common, [0])

    assert _cursors(builder, blocks)[0] == [3]


def test_null_block_rows_are_skipped():
    builder = _create_spec_builder()
    common = _make_common([120, 120], [100, 100])
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=3, is_flush=0)
    # Mark row 1 as padding, the way cudagraph padding does.
    common.block_table_tensor[1, :] = NULL_BLOCK_ID

    _build(builder, common, [2, 2])

    write_pos, _, _ = _cursors(builder, blocks)
    assert write_pos[0] == 5
    assert write_pos[1] == 3


def test_cudagraph_scratch_widens_to_the_padded_batch():
    builder = _create_spec_builder(full_cuda_graph=True)
    common = _make_common([120, 120], [100, 100])
    meta = _build(builder, common, [1, 1])

    assert meta.spec_bc_pre_scratch is not None
    assert meta.spec_bc_pre_scratch.shape[0] == meta.num_reqs
    # It must be the persistent buffer, not a fresh allocation, or the captured
    # graph would replay against a dangling pointer.
    assert meta.spec_bc_pre_scratch.data_ptr() == builder.decode_spec_bc_pre.data_ptr()
    # The cursors are block-keyed and full-length, so they are handed through
    # whole rather than sliced to the batch.
    assert meta.spec_write_pos_d.shape[0] == builder.spec_write_pos.shape[0]
    assert meta.spec_write_pos_d.data_ptr() == builder.spec_write_pos.data_ptr()


def test_ring_geometry_matches_the_page():
    builder = _create_spec_builder()

    assert builder.spec_query_len == SPEC_QUERY_LEN
    assert builder.spec_flush_threshold == FLUSH_THRESHOLD
    assert builder.spec_ring_len == RING_LEN
    # ngroups comes directly from the shared head-major B cache.
    assert builder.decode_spec_bc_pre.shape[1] == NGROUPS
    assert builder.decode_spec_bc_pre.shape[2] == RING_LEN


def test_spec_page_shape_is_validated():
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.cache_config.use_replayssm_spec = True
    # Missing the shared B cache tensor.
    bad_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1), (1, 1, 1), (1, 8, 1), (1, 8)),
        dtypes=(torch.float32,),
    )
    with pytest.raises(ValueError, match="5-tensor Mamba2 page"):
        MockMambaBuilder(bad_spec, ["layer0"], vllm_config, DEVICE)
