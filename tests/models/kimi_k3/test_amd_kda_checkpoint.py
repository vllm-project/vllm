# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm KDA prefill checkpoint: where it is taken and what it stores.

Two things have to line up for a checkpoint to be safe: the metadata must name
a position the chunk kernel will actually write, and the conv window saved
there must be the one a resumed prefill would have left.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadataBuilder
from vllm.models.kimi_k3.amd.ops.kda_checkpoint import store_conv_checkpoints
from vllm.models.kimi_k3.amd.ops.kda_chunk import KDA_CHECKPOINT_ALIGNMENT
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and torch.cuda.is_available()),
    reason="The ROCm KDA prefill checkpoint needs a ROCm device",
)

# A mamba block size the chunk walk lands on. The real K3 value is 768; any
# multiple of the kernel's 64-token chunk behaves the same and keeps the
# fixtures small.
BLOCK_SIZE = 64


def _builder(
    num_prefill_checkpoint_blocks: int = 1,
    mamba_cache_mode: str = "align",
    block_size: int = BLOCK_SIZE,
    prefix_match_unit: int | None = None,
) -> KimiK3ROCmKDAMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=block_size
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.cache_config.prefix_match_unit = prefix_match_unit
    return KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode=mamba_cache_mode,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=(
                KDA_CHECKPOINT_ALIGNMENT if num_prefill_checkpoint_blocks else None
            ),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def _build(batch: BatchSpec, block_size: int = BLOCK_SIZE, **kw):
    common = create_common_attn_metadata(
        batch, block_size, torch.device("cuda"), arange_block_indices=True
    )
    return _builder(block_size=block_size, **kw).build(0, common)


def _i32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def test_checkpoint_metadata_targets_last_aligned_boundary() -> None:
    # Slot p holds the state after (p + 1) * block_size tokens, so a 200-token
    # prompt checkpoints at 192 into slot 2 and a 128-token one at 64 into slot
    # 0. The block table is an arange, so request 1 owns blocks 4..7.
    md = _build(BatchSpec(seq_lens=[200, 128], query_lens=[200, 128]))

    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.checkpoint_offsets, _i32([192, 64]))
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([2, 4]))


def test_checkpoint_offset_is_relative_to_query_start() -> None:
    # 256 tokens computed, 144 more this step: the boundary at 384 lands 128
    # tokens into the chunk, and cdiv(400, 64) - 2 = 5 is its slot.
    md = _build(BatchSpec(seq_lens=[400, 64], query_lens=[144, 64]))

    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.checkpoint_offsets, _i32([128, 0]))
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([5, -1]))


def test_checkpoint_metadata_avoids_live_state_row() -> None:
    # Writing the row the walk itself owns would race it.
    md = _build(BatchSpec(seq_lens=[200, 130], query_lens=[200, 130]))

    assert md.checkpoint is not None
    # `mamba_get_block_table_tensor` hands the walk block (seq_len - 1) // bs.
    live = torch.tensor([(200 - 1) // BLOCK_SIZE, (130 - 1) // BLOCK_SIZE])
    table = torch.arange(2 * 4).view(2, 4)
    live_rows = table[torch.arange(2), live].cuda()
    assert not bool((md.checkpoint.state_indices == live_rows).any())


def test_checkpoint_metadata_skips_chunk_without_boundary() -> None:
    # 128 computed, 40 more: the only boundary at 128 is the state this chunk
    # started from, so there is nothing new to snapshot.
    assert _build(BatchSpec(seq_lens=[168], query_lens=[40])).checkpoint is None


def test_checkpoint_metadata_skips_null_reservation() -> None:
    # Only the chunk that finishes a prompt gets a block reserved; every
    # earlier one leaves the shared null block, which nothing hashes and
    # writing it would corrupt every other request's scratch row.
    batch = BatchSpec(seq_lens=[200], query_lens=[200])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, torch.device("cuda"), arange_block_indices=True
    )
    # cdiv(200, 64) - 2 = 2 is the checkpoint slot.
    common.block_table_tensor[0, 2] = NULL_BLOCK_ID

    md = _builder().build(0, common)

    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([-1]))


def test_checkpoint_metadata_absent_without_reserved_block() -> None:
    md = _build(
        BatchSpec(seq_lens=[200], query_lens=[200]),
        num_prefill_checkpoint_blocks=0,
    )
    assert getattr(md, "checkpoint", None) is None


def test_checkpoint_metadata_absent_outside_align_mode() -> None:
    md = _build(
        BatchSpec(seq_lens=[200], query_lens=[200]),
        mamba_cache_mode="none",
    )
    assert getattr(md, "checkpoint", None) is None


# ---------------------------------------------------------------------------
# Conv half
# ---------------------------------------------------------------------------

DIM = 384
WIDTH = 4
STATE_LEN = WIDTH - 1


def _conv_fixture(seqlens: list[int], slots: int = 16, seed: int = 0):
    """A cache row as wide as the layer allocates it."""
    torch.manual_seed(seed)
    total = sum(seqlens)
    x = torch.randn(total, DIM, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    conv_state = torch.full(
        (slots, DIM, STATE_LEN), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    return x, cu, conv_state


def test_conv_checkpoint_saves_window_before_offset() -> None:
    seqlens = [200, 130]
    x, cu, conv_state = _conv_fixture(seqlens)
    offsets = _i32([192, 128])
    rows = _i32([5, 9])

    store_conv_checkpoints(x, conv_state, cu, offsets, rows, STATE_LEN)

    for seq, (offset, row) in enumerate(zip([192, 128], [5, 9])):
        end = int(cu[seq]) + offset
        expected = x[end - STATE_LEN : end].transpose(0, 1)
        torch.testing.assert_close(conv_state[row], expected)


def test_conv_checkpoint_skips_opted_out_sequences() -> None:
    x, cu, conv_state = _conv_fixture([200, 200])
    before = conv_state.clone()

    store_conv_checkpoints(x, conv_state, cu, _i32([0, 192]), _i32([5, -1]), STATE_LEN)

    assert torch.equal(conv_state.isnan(), before.isnan())


def test_conv_checkpoint_matches_prefill_truncated_at_offset() -> None:
    # causal_conv1d_fn's own state write is the reference: a prefill that
    # stopped at the offset leaves exactly the row a later prefix-cache hit
    # reads back.
    seqlens = [200, 130]
    offsets = [192, 64]
    x, cu, conv_state = _conv_fixture(seqlens, seed=3)
    weight = torch.randn(DIM, WIDTH, device="cuda", dtype=torch.bfloat16) * 0.3
    bias = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)
    rows = [5, 9]

    store_conv_checkpoints(x, conv_state, cu, _i32(offsets), _i32(rows), STATE_LEN)

    # A prefill of just the prefix, into its own cache. Row 0 is the null block
    # the conv kernel skips by design, so the reference rows start at 1.
    prefix = torch.cat(
        [x[int(cu[n]) : int(cu[n]) + off] for n, off in enumerate(offsets)]
    )
    prefix_cu = torch.tensor(
        [0] + torch.tensor(offsets).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    reference = torch.zeros_like(conv_state)
    causal_conv1d_fn(
        prefix.transpose(0, 1),
        weight,
        bias,
        activation="silu",
        conv_states=reference,
        has_initial_state=torch.zeros(2, dtype=torch.bool, device="cuda"),
        cache_indices=_i32([1, 2]),
        query_start_loc=prefix_cu,
    )

    torch.testing.assert_close(conv_state[rows[0]], reference[1])
    torch.testing.assert_close(conv_state[rows[1]], reference[2])


def test_conv_checkpoint_honours_transposed_cache_view() -> None:
    # The SD conv layout reaches the layer transposed, not contiguous.
    x, cu, _ = _conv_fixture([200])
    raw = torch.full(
        (16, STATE_LEN, DIM), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    conv_state = raw.transpose(-1, -2)
    assert not conv_state.is_contiguous()

    store_conv_checkpoints(x, conv_state, cu, _i32([192]), _i32([7]), STATE_LEN)

    torch.testing.assert_close(conv_state[7], x[192 - STATE_LEN : 192].transpose(0, 1))
    untouched = torch.ones(16, dtype=torch.bool, device="cuda")
    untouched[7] = False
    assert bool(raw[untouched].isnan().all())


def test_checkpoint_metadata_indexes_prefill_rows_after_decodes() -> None:
    # The chunk kernel receives the prefill tail alone, so the metadata has to
    # be indexed the same way or it snapshots the wrong request's boundary.
    md = _build(BatchSpec(seq_lens=[64, 200], query_lens=[1, 200]))

    assert md.num_decodes == 1 and md.num_prefills == 1
    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.checkpoint_offsets, _i32([192]))
    # Request 1 owns blocks 4..7, so its slot-2 state block is 6.
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([6]))


def test_conv_checkpoint_addresses_row_past_int32_limit() -> None:
    # state_stride_0 is the whole mamba page, padded up to the attention page,
    # so row * stride passes 2**31 a few thousand blocks into a large cache and
    # an int32 wrap stores outside the tensor. The margin on the plain prefill
    # path is 1.6x, which is luck: a bigger cache, a higher utilization or the
    # wider row speculative decoding allocates all close it.
    stride_row = DIM * (STATE_LEN + 8)
    row = 2**31 // stride_row + 1
    need = (row + 1) * stride_row * 2
    free, _ = torch.cuda.mem_get_info()
    if free < need + (4 << 30):
        pytest.skip(f"needs {need >> 30} GiB of free VRAM to reach the wrap")

    x, cu, _ = _conv_fixture([200])
    conv_state = torch.zeros(
        row + 1, DIM, STATE_LEN + 8, device="cuda", dtype=torch.bfloat16
    )
    assert conv_state.stride(0) * row > 2**31, "fixture does not reach the wrap"

    store_conv_checkpoints(x, conv_state, cu, _i32([192]), _i32([row]), STATE_LEN)

    torch.testing.assert_close(
        conv_state[row, :, :STATE_LEN], x[192 - STATE_LEN : 192].transpose(0, 1)
    )
    assert not bool(conv_state[:row].any()), "the store wrote outside its row"


def test_conv_checkpoint_reads_row_strided_projection() -> None:
    # mixed_qkv is a band of the fused QKVGFAB projection, so its row stride is
    # the whole projection width, not DIM.
    torch.manual_seed(5)
    total, pad = 200, 177
    projection = torch.randn(total, DIM + pad, device="cuda", dtype=torch.bfloat16)
    x = projection[:, :DIM]
    assert x.stride(0) == DIM + pad

    cu = _i32([0, total])
    conv_state = torch.full(
        (16, DIM, STATE_LEN), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    store_conv_checkpoints(x, conv_state, cu, _i32([192]), _i32([4]), STATE_LEN)

    torch.testing.assert_close(conv_state[4], x[192 - STATE_LEN : 192].transpose(0, 1))


def test_finer_prefix_match_unit_moves_checkpoint_to_hash_grid() -> None:
    # A partial prefix hit matches at prefix_match_unit, so a state captured
    # only at block boundaries is too coarse to resume from. Block 256 with
    # unit 128 moves a 700-token prompt's checkpoint from 512 to 640.
    md = _build(
        BatchSpec(seq_lens=[700], query_lens=[700]),
        block_size=256,
        prefix_match_unit=128,
    )

    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.checkpoint_offsets, _i32([640]))
    # Slot is still block-indexed: cdiv(700, 256) - 2 = 1, request 0 owns 0..2.
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([1]))


def test_checkpoint_metadata_skips_unaligned_offset() -> None:
    # The scheduler declines a boundary off the published alignment and
    # reserves nothing, so the builder has to reach the same verdict or it
    # writes a block nobody hashed.
    # Block 192, unit 96: a 700-token prompt's last hash boundary is 672, which
    # is 32 tokens off the chunk grid.
    md = _build(
        BatchSpec(seq_lens=[700], query_lens=[700]),
        block_size=192,
        prefix_match_unit=96,
    )

    assert md.checkpoint is None


# ---------------------------------------------------------------------------
# Opting in
# ---------------------------------------------------------------------------


def _spec_for(monkeypatch, **attrs) -> MambaSpec:
    """Run the layer's `get_kv_cache_spec` over a stubbed base spec."""
    base = MambaSpec(block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,))
    monkeypatch.setattr(
        GatedDeltaNetAttention, "get_kv_cache_spec", lambda self, cfg: base
    )
    layer = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
    for name, value in dict(
        use_prefill_checkpoint=True,
        use_fused_chunk=True,
        num_spec=0,
        use_safe_gate=True,
    ).items():
        setattr(layer, name, attrs.get(name, value))
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B")
    vllm_config.cache_config.mamba_cache_mode = "align"
    return layer.get_kv_cache_spec(vllm_config)


def test_opt_in_publishes_chunk_walk_alignment(monkeypatch) -> None:
    # The scheduler learns the kernel's granularity only from the spec, so
    # reserving a block without publishing the alignment would hash a state
    # that is never written.
    spec = _spec_for(monkeypatch)

    assert spec.num_prefill_checkpoint_blocks == 1
    assert spec.prefill_checkpoint_alignment == KDA_CHECKPOINT_ALIGNMENT


@pytest.mark.parametrize(
    "attr,value",
    [
        ("use_prefill_checkpoint", False),
        ("use_fused_chunk", False),
        ("use_safe_gate", False),
        ("num_spec", 8),
    ],
)
def test_paths_that_cannot_export_stay_opted_out(monkeypatch, attr, value) -> None:
    spec = _spec_for(monkeypatch, **{attr: value})

    assert spec.num_prefill_checkpoint_blocks == 0
    assert spec.prefill_checkpoint_alignment is None
