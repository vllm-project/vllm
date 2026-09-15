# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which path GDN metadata takes when a decode row carries no draft tokens.

`build_for_cudagraph_capture` derives the draft count from the query length
(`num_decode_draft_tokens = query_len - 1`), so a capture shape of one token
per request describes rows with zero drafts. `GDNAttentionMetadataBuilder`
routes every row with `num_decode_draft_tokens_cpu >= 0` through the
speculative path, zero drafts included: that path is the only one that
applies each row's accepted-token offset to the recurrent state, so a
zero-draft decode row must not fall back to the plain decode path.
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
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
NUM_SPEC = 3


def _builder(
    use_spec_decode: bool,
    mamba_cache_mode: str = "none",
    shapes: tuple[tuple[int, ...], ...] = ((16, 64),),
    dtypes: tuple[torch.dtype, ...] = (torch.float16,),
) -> GDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = (
        SpeculativeConfig(method="ngram", num_speculative_tokens=NUM_SPEC)
        if use_spec_decode
        else None
    )
    # "none" keeps the block table as-is; the align gather needs columns for
    # the speculative blocks, which `_decode_metadata` appends on request.
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=dtypes,
        num_speculative_blocks=NUM_SPEC,
    )

    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _capture_metadata(builder, query_len: int, batch_size: int = 2):
    """Metadata as produced during CUDA-graph capture of a decode shape."""
    batch = BatchSpec(
        seq_lens=[64] * batch_size,
        query_lens=[query_len] * batch_size,
    )
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    return builder.build_for_cudagraph_capture(common)


def _decode_metadata(builder, num_accepted: list[int], seq_len: int = 64):
    """Metadata for a live decode batch where no row was given drafts.

    Every row schedules a single token, so `num_decode_draft_tokens_cpu` is all
    zeros -- the shape the runner produces once `zero_draft_decode_mask` has
    promoted zero-draft decode rows out of the `-1` placeholder.
    `num_accepted` is what the previous step accepted (bonus token included,
    so one is the minimum), which is what decides the state slot.
    """
    batch = BatchSpec(
        seq_lens=[seq_len] * len(num_accepted), query_lens=[1] * len(num_accepted)
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    if builder.vllm_config.cache_config.mamba_cache_mode == "align":
        # Align gathers the last 1 + num_speculative_blocks columns starting at
        # the request's current block, so the table has to reach past the end of
        # the sequence -- at runtime it is sized by max_model_len, while the
        # test helper sizes it by seq_len.
        rows, cols = common.block_table_tensor.shape
        extra = (
            torch.arange(
                rows * NUM_SPEC, dtype=common.block_table_tensor.dtype, device=DEVICE
            ).view(rows, NUM_SPEC)
            + rows * cols
        )
        common.block_table_tensor = torch.cat([common.block_table_tensor, extra], dim=1)
    accepted = torch.tensor(num_accepted, dtype=torch.int32, device=DEVICE)
    num_decode_draft_tokens_cpu = torch.zeros(len(num_accepted), dtype=torch.int32)
    return builder.build(0, common, accepted, num_decode_draft_tokens_cpu)


def test_zero_draft_capture_shape_routes_through_the_speculative_path():
    """The one-token capture shape is what breaks backends whose decode kernels
    assume a full num_spec + 1 query window: `num_decode_draft_tokens_cpu =
    query_len - 1 = 0` for every row, which is `>= 0` and must still take the
    speculative path (not the plain decode path that drops the state offset),
    with a state tensor sized for the full window even though every row only
    supplies a single query token.

    Only `query_len=1` (zero drafts) discriminates this from the plain-decode
    path before the fix: for any `query_len > 1`, `num_decode_draft_tokens_cpu`
    is already `> 0` and both the pre-fix and post-fix code agree, so this
    does not need parametrizing over larger draft counts.
    """
    meta = _capture_metadata(_builder(use_spec_decode=True), query_len=1)

    assert meta.num_spec_decodes == 2
    assert meta.num_decodes == 0
    slots = meta.spec_state_indices_tensor
    assert slots is not None
    # The kernels index columns 0..num_spec of this tensor.
    assert slots.shape[1] == NUM_SPEC + 1
    assert int(slots.min()) >= 0
    query_lens = torch.diff(meta.spec_query_start_loc)
    assert torch.all(query_lens == 1)


def test_plain_decode_without_spec_decode_skips_the_speculative_path():
    """Without a speculative config the builder never builds spec metadata,
    so a one-token-per-row decode capture takes the plain decode path.

    Not a regression check for this patch (this branch is untouched by it),
    but it is what keeps the test above from passing vacuously: it would also
    pass if the builder routed every row through the speculative path
    unconditionally, regardless of whether speculative decoding is even
    configured.
    """
    meta = _capture_metadata(_builder(use_spec_decode=False), query_len=1)

    assert meta.num_spec_decodes == 0
    assert meta.num_decodes == 2


@pytest.mark.parametrize(
    "mamba_cache_mode, shapes, dtypes",
    [
        ("none", ((16, 64),), (torch.float16,)),
        # Kimi-Linear's KDA layers subclass GatedDeltaNetAttention and consume
        # this same metadata, and prefix caching (which Kimi enables by
        # default) puts the cache in align mode -- so the guarantee has to hold
        # for that state shape and mode too.
        (
            "align",
            MambaStateShapeCalculator.kda_state_shape(
                tp_world_size=1, num_heads=4, head_dim=32, num_spec=NUM_SPEC
            ),
            (torch.bfloat16, torch.bfloat16),
        ),
    ],
    ids=["gdn_none", "kimi_kda_align"],
)
def test_zero_draft_batch_still_applies_the_accepted_token_offset(
    mamba_cache_mode, shapes, dtypes
):
    """The numeric difference the fix makes, on the batch shape that triggers it.

    The recurrent kernels read their state at column `num_accepted_tokens - 1`
    of `spec_state_indices_tensor`. Before the fix, a batch in which *every*
    speculative row drafted nothing was pushed onto the non-speculative path,
    which passes `non_spec_state_indices_tensor` -- always column 0 -- and
    drops `num_accepted_tokens` entirely. A row that accepted drafts on the
    previous step wrote its fresh state further along, so column 0 is stale.
    """
    accepted = [NUM_SPEC, 1]
    meta = _decode_metadata(
        _builder(
            use_spec_decode=True,
            mamba_cache_mode=mamba_cache_mode,
            shapes=shapes,
            dtypes=dtypes,
        ),
        accepted,
    )

    assert meta.num_spec_decodes == len(accepted)
    assert meta.num_decodes == 0
    # Dropping these is what silently sent the kernels to the wrong slot.
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist()[: len(accepted)] == accepted
    assert meta.non_spec_state_indices_tensor is None

    slots = meta.spec_state_indices_tensor
    assert slots is not None
    for row, num_accepted in enumerate(accepted):
        read = int(slots[row, num_accepted - 1])
        fallback = int(slots[row, 0])
        # Only the row that accepted a draft moved off column 0; the row that
        # accepted nothing but its bonus token legitimately stays there. This
        # also confirms the columns are genuinely distinct blocks (via
        # `arange_block_indices=True` in `_decode_metadata`), not just
        # distinct labels for the same one -- otherwise `read != fallback`
        # here would mean nothing.
        assert (read != fallback) == (num_accepted > 1)
