# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM speculative-decode ring cursors in GDNAttentionMetadataBuilder.

The cursors are block-keyed and advanced on-device by commit_gdn_replayssm_spec,
so these tests drive builder.build() and read the resulting buffers back. Each
case builds one CommonAttentionMetadata and reuses it, because
create_common_attn_metadata draws fresh block ids on every call.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
MAX_SPEC_LEN = 1 + NUM_SPEC
BUFFER_LEN = 8
FLUSH_THRESHOLD = BUFFER_LEN + MAX_SPEC_LEN
RING_LEN = 16
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReplaySSM spec cursors are Triton kernels"
)


def _create_spec_builder(
    buffer_len: int = BUFFER_LEN,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
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
    # Set after construction so validate_mamba_cached_spec_kernel (Triton only)
    # does not run against the mock model.
    vllm_config.cache_config.use_replayssm_spec = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _make_common(seq_lens: list[int], decode_base: list[int], query_lens: list[int]):
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    return create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False] * len(seq_lens), dtype=torch.bool),
        replayssm_decode_base_cpu=torch.tensor(decode_base, dtype=torch.int32),
    )


def _build(builder, common, num_accepted: list[int]):
    n = len(num_accepted)
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor(
            num_accepted, dtype=torch.int32, device=DEVICE
        ),
        num_decode_draft_tokens_cpu=torch.full((n,), NUM_SPEC, dtype=torch.int32),
    )


def _cursors(builder, blocks: list[int]):
    idx = torch.tensor(blocks, device=DEVICE)
    return (
        builder.spec_write_pos[idx].tolist(),
        builder.spec_cache_base[idx].tolist(),
        builder.spec_is_flush[idx].tolist(),
    )


def _seed(builder, blocks: list[int], write_pos: int, is_flush: int):
    for block in blocks:
        builder.spec_write_pos[block] = write_pos
        builder.spec_cache_base[block] = 0
        builder.spec_is_flush[block] = is_flush


def _prime(builder, common) -> list[int]:
    """Allocate the cursor buffers and return this batch's physical block ids."""
    meta = _build(builder, common, [1] * common.num_reqs)
    return meta.spec_state_indices_tensor[: common.num_reqs, 0].tolist()


def _spec_window(n: int = 1):
    return [MAX_SPEC_LEN] * n


def test_commit_advances_write_pos_by_accepted():
    builder = _create_spec_builder()
    # num_computed = 120 - 4 = 116 > decode_base, so no reset fires.
    common = _make_common([120], [100], _spec_window())
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    _build(builder, common, [3])

    write_pos, cache_base, _ = _cursors(builder, blocks)
    assert write_pos == [5]
    assert cache_base == [0]


def test_rejected_drafts_are_not_committed():
    """Rollback: only accepted tokens advance the cursor, the rest are dropped."""
    builder = _create_spec_builder()
    common = _make_common([120], [100], _spec_window())
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    _build(builder, common, [1])

    assert _cursors(builder, blocks)[0] == [3]


def test_flush_advances_base_and_restarts_write_pos():
    builder = _create_spec_builder()
    common = _make_common([120], [100], _spec_window())
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=5, is_flush=1)

    _build(builder, common, [2])

    write_pos, cache_base, _ = _cursors(builder, blocks)
    # The flush consumed the 5 committed history tokens, so the ring restarts
    # past them and only the freshly accepted tokens remain.
    assert cache_base == [5]
    assert write_pos == [2]


def test_early_flush_keeps_room_for_a_full_window():
    """is_flush is set one window early: write_pos + 2 * max_spec_len > L."""
    builder = _create_spec_builder()
    common = _make_common([120], [100], _spec_window())
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
    common = _make_common([104], [100], _spec_window())
    blocks = _prime(builder, common)
    # Stale cursors left behind by whoever held this block before.
    _seed(builder, blocks, write_pos=6, is_flush=1)

    _build(builder, common, [1])

    assert _cursors(builder, blocks) == ([0], [0], [0])


def test_resumed_request_reanchors_on_decode_base():
    """Preemption: decode_base moves to the resume point, which re-fires the
    reset so the recomputed request does not inherit its old cursors."""
    builder = _create_spec_builder()
    common = _make_common([120], [116], _spec_window())
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=7, is_flush=1)

    _build(builder, common, [1])

    write_pos, cache_base, _ = _cursors(builder, blocks)
    assert (write_pos, cache_base) == ([0], [0])


def test_steady_state_row_is_not_reset():
    builder = _create_spec_builder()
    common = _make_common([120], [100], _spec_window())
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=2, is_flush=0)

    _build(builder, common, [1])

    assert _cursors(builder, blocks)[0] == [3]


def test_mixed_batch_resets_only_the_entering_rows():
    builder = _create_spec_builder()
    # Row 0 steady state, row 1 first decode, row 2 steady state.
    common = _make_common([120, 104, 124], [100, 100, 100], _spec_window(3))
    blocks = _prime(builder, common)
    _seed(builder, blocks, write_pos=4, is_flush=0)

    _build(builder, common, [2, 2, 2])

    assert _cursors(builder, blocks)[0] == [6, 0, 6]


def test_draft_less_rows_route_through_the_spec_kernel():
    """A row with no drafts is a T=1 window, not a baseline decode. Routing it
    to the baseline path would read a checkpoint the ring has moved past."""
    builder = _create_spec_builder()
    common = _make_common([120, 120], [100, 100], query_lens=[1, 1])
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
        # Stale on draft-less steps, so it must not drive the routing mask.
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1], dtype=torch.int32),
    )

    assert meta.num_spec_decodes == 2
    assert meta.num_decodes == 0
    assert meta.spec_write_pos_d is not None


def test_prefilling_rows_stay_on_the_prefill_path():
    """A still-prefilling chunk is excluded from the spec mask. It cannot reach
    a captured decode graph either: _is_uniform_decode needs every row to be
    exactly 1 + num_spec tokens wide, which a 1-token chunk breaks."""
    builder = _create_spec_builder()
    batch = BatchSpec(seq_lens=[120, 100], query_lens=[MAX_SPEC_LEN, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, True], dtype=torch.bool),
        replayssm_decode_base_cpu=torch.tensor([100, 100], dtype=torch.int32),
    )
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC, -1], dtype=torch.int32),
    )

    assert meta.num_spec_decodes == 1
    assert meta.num_prefills == 1


def test_ring_geometry_matches_the_page():
    builder = _create_spec_builder()

    assert builder.max_spec_len == MAX_SPEC_LEN
    assert builder.spec_flush_threshold == FLUSH_THRESHOLD
    assert builder.spec_ring_len == RING_LEN


def test_cursors_are_the_persistent_block_keyed_buffers():
    builder = _create_spec_builder(full_cuda_graph=True)
    common = _make_common([120, 120], [100, 100], _spec_window(2))
    meta = _build(builder, common, [1, 1])

    # Block-keyed and full-length, so a captured graph replays against a fixed
    # address rather than a per-step slice.
    assert meta.spec_write_pos_d.data_ptr() == builder.spec_write_pos.data_ptr()
    assert meta.spec_cache_base_d.data_ptr() == builder.spec_cache_base.data_ptr()
    assert meta.spec_is_flush_d.data_ptr() == builder.spec_is_flush.data_ptr()
    assert meta.spec_write_pos_d.shape[0] == 1024


def test_layer_and_config_pages_agree():
    """The layer sizes its kv_cache from get_state_shape/get_state_dtype while
    the engine sizes the page from the config classmethods. If the two disagree
    the page silently truncates, because abstract.py zips shapes with dtypes.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B")
    vllm_config.cache_config.use_replayssm_spec = True
    vllm_config.cache_config.replayssm_buffer_len = BUFFER_LEN
    hf = vllm_config.model_config.hf_text_config

    base_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1,
        hf.linear_num_key_heads,
        hf.linear_num_value_heads,
        hf.linear_key_head_dim,
        hf.linear_value_head_dim,
        hf.linear_conv_kernel_dim,
        NUM_SPEC,
    )
    shapes = MambaStateShapeCalculator.append_gated_delta_net_replayssm_spec_ring(
        base_shape,
        hf.linear_num_key_heads,
        hf.linear_num_value_heads,
        hf.linear_key_head_dim,
        hf.linear_value_head_dim,
        1,
        BUFFER_LEN,
        NUM_SPEC,
    )
    base_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype, "auto", "auto"
    )
    dtypes = MambaStateDtypeCalculator.append_gated_delta_net_replayssm_spec_ring(
        base_dtype, vllm_config.model_config.dtype
    )

    assert len(shapes) == 5
    assert len(shapes) == len(dtypes)
