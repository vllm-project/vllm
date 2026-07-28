# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ReplaySSM spec ring cursors in BaseMambaAttentionMetadataBuilder.

Separate from the Triton suite: the FlashInfer ring is exactly B + T (wrapping
by subtraction, not a mask), its flush decision uses each row's actual length
rather than the maximum T, and decode entry is driven by a once-per-admission
flag instead of a decode_base comparison.

The admission flag is the DSpark bug class: #49847 plumbed its decode_base only
through the classic worker, so the reset never fired on the V2 runner and the
ring replayed from the wrong origin. These tests pin the classic-runner half of
the lifecycle; the V2 half is covered by the engine gate.
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
from vllm.config.mamba import MambaBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
MAX_SPEC_LEN = 1 + NUM_SPEC
BUFFER_LEN = 8
RING_LEN = BUFFER_LEN + MAX_SPEC_LEN  # 12, deliberately not a power of two

NHEADS, HEAD_DIM, DSTATE, NGROUPS = 2, 4, 8, 1
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReplaySSM spec cursors are Triton kernels"
)


def _make_flashinfer_spec() -> MambaSpec:
    # Five-tensor page: (conv, ssm, x_cache, B_cache, dt_cache).
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            (NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE, 3),
            (NHEADS, HEAD_DIM, DSTATE),
            (NHEADS, RING_LEN, HEAD_DIM),
            (NGROUPS, RING_LEN, DSTATE),
            (NHEADS, RING_LEN),
        ),
        dtypes=(
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
        ),
    )


def _create_builder(algorithm: str = "monolith") -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        num_gpu_blocks=1024,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    # Set after construction so validate_mamba_cached_spec_kernel does not run
    # against the mock model.
    vllm_config.cache_config.use_replayssm_spec = True
    vllm_config.cache_config.replayssm_buffer_len = BUFFER_LEN
    vllm_config.mamba_config.backend = MambaBackendEnum.FLASHINFER
    vllm_config.mamba_config.replayssm_spec_algorithm = algorithm
    return MockMambaBuilder(_make_flashinfer_spec(), ["layer0"], vllm_config, DEVICE)


def _make_common(seq_lens, needs_reset, query_lens=None):
    n = len(seq_lens)
    query_lens = query_lens if query_lens is not None else [MAX_SPEC_LEN] * n
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    return create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False] * n, dtype=torch.bool),
        replayssm_decode_base_cpu=torch.zeros(n, dtype=torch.int32),
        replayssm_needs_reset_cpu=torch.tensor(needs_reset, dtype=torch.int8),
    )


def _build(builder, common, num_accepted):
    return builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor(
            num_accepted, dtype=torch.int32, device=DEVICE
        ),
    )


def _cursors(builder, blocks):
    idx = torch.tensor(blocks, device=DEVICE)
    return (
        builder.spec_write_pos[idx].tolist(),  # history_len
        builder.spec_post_origin[idx].tolist(),  # ring_start
        builder.spec_is_flush[idx].tolist(),
    )


def _seed(builder, blocks, history, origin, is_flush):
    for block in blocks:
        builder.spec_write_pos[block] = history
        builder.spec_post_origin[block] = origin
        builder.spec_is_flush[block] = is_flush


def _prime(builder, common):
    """Allocate the cursor buffers and return this batch's physical block ids."""
    meta = _build(builder, common, [1] * common.num_reqs)
    return meta.state_indices_tensor_d[:, 0].tolist()


def test_non_flush_accumulates_history():
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0])
    blocks = _prime(builder, common)
    _seed(builder, blocks, history=2, origin=0, is_flush=0)

    _build(builder, common, [3])

    history, origin, _ = _cursors(builder, blocks)
    assert history == [5]
    assert origin == [0]


def test_flush_advances_origin_by_the_replayed_count():
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0])
    blocks = _prime(builder, common)
    _seed(builder, blocks, history=5, origin=0, is_flush=1)

    _build(builder, common, [2])

    history, origin, _ = _cursors(builder, blocks)
    assert origin == [5]
    assert history == [2]


def test_origin_wraps_by_subtraction_not_by_mask():
    """R = 12 here: origin 10 + history 5 must wrap to 3, and 15 & 11 == 11."""
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0])
    blocks = _prime(builder, common)
    _seed(builder, blocks, history=5, origin=10, is_flush=1)

    _build(builder, common, [2])

    history, origin, _ = _cursors(builder, blocks)
    assert origin == [3]
    assert history == [2]


@pytest.mark.parametrize(
    "history,accepted,query_len,expected_flush",
    [
        (3, 1, MAX_SPEC_LEN, 0),  # 4 + 4 == 8, not > B
        (4, 1, MAX_SPEC_LEN, 1),  # 5 + 4 == 9 > B
        (4, 1, 1, 0),  # same history, a one-token row does not flush
        (7, 1, 1, 0),  # 8 + 1 == 9 > 8 -> flush
    ],
)
def test_flush_uses_the_actual_row_length(history, accepted, query_len, expected_flush):
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0], query_lens=[query_len])
    blocks = _prime(builder, common)
    _seed(builder, blocks, history=history, origin=0, is_flush=0)

    _build(builder, common, [accepted])

    assert _cursors(builder, blocks)[2] == [expected_flush]


def test_zero_acceptance_freezes_cursors_but_recomputes_flush():
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0])
    blocks = _prime(builder, common)
    _seed(builder, blocks, history=6, origin=4, is_flush=1)

    _build(builder, common, [0])

    history, origin, is_flush = _cursors(builder, blocks)
    assert (history, origin) == ([6], [4])
    # 6 + 4 > 8, decided fresh for this call rather than inherited.
    assert is_flush == [1]


def test_admission_flag_resets_a_recycled_block():
    builder = _create_builder()
    common = _make_common([120], needs_reset=[1])
    blocks = _prime(builder, common)
    # Stale cursors from whoever held this block before.
    _seed(builder, blocks, history=6, origin=9, is_flush=1)

    _build(builder, common, [2])

    history, origin, is_flush = _cursors(builder, blocks)
    assert history == [0]
    assert origin == [0]
    # An empty ring cannot overflow: 0 + query_len <= T <= B.
    assert is_flush == [0]


def test_reset_only_touches_rows_whose_flag_is_set():
    """A mixed batch: one row entering decode, one mid-stream."""
    builder = _create_builder()
    common = _make_common([120, 120], needs_reset=[1, 0])
    blocks = _prime(builder, common)
    _seed(builder, blocks[:1], history=6, origin=9, is_flush=0)
    _seed(builder, blocks[1:], history=2, origin=1, is_flush=0)

    _build(builder, common, [2, 3])

    history, origin, _ = _cursors(builder, blocks)
    assert history[0] == 0 and origin[0] == 0
    assert history[1] == 5 and origin[1] == 1


def test_cursors_are_block_keyed_so_padded_rows_are_untouched():
    builder = _create_builder()
    common = _make_common([120], needs_reset=[0])
    blocks = _prime(builder, common)
    other = (blocks[0] + 1) % 1024
    _seed(builder, [other], history=7, origin=2, is_flush=1)
    _seed(builder, blocks, history=1, origin=0, is_flush=0)

    _build(builder, common, [1])

    assert _cursors(builder, [other])[0] == [7]


@pytest.mark.parametrize("algorithm", ["monolith", "auto", "two-kernel"])
def test_scratch_is_present_iff_the_two_kernel_path_is_available(algorithm):
    builder = _create_builder(algorithm)
    common = _make_common([120], needs_reset=[0])
    meta = _build(builder, common, [1])

    present = [
        meta.spec_fi_cb_scaled_scratch is not None,
        meta.spec_fi_cumadt_scratch is not None,
        meta.spec_fi_cb_old_scratch is not None,
    ]
    if algorithm == "monolith":
        assert present == [False, False, False]
    else:
        assert present == [True, True, True]
        assert meta.spec_fi_cb_scaled_scratch.shape[0] == meta.num_decodes
