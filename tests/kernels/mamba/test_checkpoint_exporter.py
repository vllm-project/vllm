# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MambaPrefillCheckpointExporter, called the way each of its callers calls it.

Every Mamba layer that exports internal prefill checkpoints shares this kernel,
so a change for one layer must leave the others' exports intact.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
)
from vllm.models.kimi_k3.amd.ops.kda_checkpoint import (
    KimiK3ROCmKDAPrefillCheckpointExporter,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

QUERY_LENS = [1, 160, 96, 200]
# A decode row, two checkpoints, and one whose block was not allocated.
OFFSETS = [0, 64, 64, 128]
STATE_SLOTS = [NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 3]
NUM_SLOTS = 8
RECURRENT_SHAPE = (2, 16, 16)


@dataclass(frozen=True)
class _Caller:
    exporter: MambaPrefillCheckpointExporter
    width: int
    conv_window: int
    num_spec: int = 0
    # "none", "per_row" (row i holds checkpoint i), or "rows" (a row map).
    recurrent: str = "none"
    # SSM cache dtype; the emitted states are FP32.
    recurrent_dtype: torch.dtype = torch.float32
    token_map: bool = False


CALLERS = {
    # Qwen3.5 and Qwen4Exp default to an FP32 SSM cache, Qwen3-Next to BF16.
    "gdn-flashinfer-fp32-ssm": _Caller(
        MambaPrefillCheckpointExporter(state_len=3),
        width=96,
        conv_window=3,
        num_spec=2,
        recurrent="rows",
    ),
    "gdn-flashinfer-bf16-ssm": _Caller(
        MambaPrefillCheckpointExporter(state_len=3),
        width=96,
        conv_window=3,
        num_spec=2,
        recurrent="rows",
        recurrent_dtype=torch.bfloat16,
    ),
    # Leaves state_len unset: its conv cache has no spec slots.
    "kimi-k3-flashkda": _Caller(
        FlashKDAPrefillCheckpointExporter(),
        width=96,
        conv_window=3,
        recurrent="per_row",
    ),
    "glm5next-flashkda": _Caller(
        FlashKDAPrefillCheckpointExporter(state_len=3),
        width=96,
        conv_window=3,
        num_spec=2,
        recurrent="per_row",
    ),
    "kimi-k3-rocm": _Caller(
        KimiK3ROCmKDAPrefillCheckpointExporter(state_len=3),
        width=96,
        conv_window=3,
        num_spec=2,
    ),
    "qwen4exp-ple": _Caller(
        MambaPrefillCheckpointExporter(state_len=9),
        width=640,
        conv_window=9,
        num_spec=3,
        token_map=True,
    ),
}


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("state_layout", ["DS", "SD"])
@pytest.mark.parametrize("caller", list(CALLERS))
def test_exporter_stores_each_callers_checkpoints(caller: str, state_layout: str):
    """Each checkpoint gets the inputs before its offset as its conv window and
    its emitted recurrent state; nothing else in either cache changes."""
    c = CALLERS[caller]
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(QUERY_LENS).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_tokens = sum(QUERY_LENS)
    # The token map interleaves other rows' tokens, as in a mixed spec batch.
    num_inputs = 2 * num_tokens if c.token_map else num_tokens
    raw_qkv = torch.randn(
        num_inputs, c.width, device=device, dtype=torch.bfloat16, generator=gen
    )
    token_indices = (
        torch.randperm(num_inputs, device=device, generator=gen)[:num_tokens]
        if c.token_map
        else torch.arange(num_tokens, device=device)
    )

    state_width = c.conv_window + c.num_spec
    if state_layout == "DS":
        conv_state = torch.randn(
            NUM_SLOTS, c.width, state_width, device=device, dtype=torch.bfloat16
        )
    else:
        conv_state = torch.randn(
            NUM_SLOTS, state_width, c.width, device=device, dtype=torch.bfloat16
        ).transpose(-1, -2)
    recurrent_state = torch.randn(
        NUM_SLOTS, *RECURRENT_SHAPE, device=device, dtype=c.recurrent_dtype
    )
    # FP32 like the kernels emit, with values the cache dtype holds exactly.
    emitted = (
        torch.randn(2 * len(OFFSETS), *RECURRENT_SHAPE, device=device, generator=gen)
        .bfloat16()
        .float()
    )
    rows = torch.randperm(emitted.shape[0], device=device, generator=gen)[
        : len(OFFSETS)
    ]
    expected_conv = conv_state.clone()
    expected_recurrent = recurrent_state.clone()
    for i, (offset, slot) in enumerate(zip(OFFSETS, STATE_SLOTS)):
        if offset == 0 or slot == NULL_BLOCK_ID:
            continue
        end = int(cu_seqlens[i]) + offset
        window = token_indices[end - c.conv_window : end]
        expected_conv[slot, :, : c.conv_window] = raw_qkv[window].T
        if c.recurrent != "none":
            row = rows[i] if c.recurrent == "rows" else i
            expected_recurrent[slot] = emitted[row].to(c.recurrent_dtype)

    checkpoint = MambaPrefillCheckpointMetadata(
        torch.tensor(OFFSETS, dtype=torch.int32, device=device),
        torch.tensor(STATE_SLOTS, dtype=torch.int32, device=device),
    )
    kwargs: dict = {}
    if c.recurrent != "none":
        kwargs.update(recurrent_checkpoint=emitted, recurrent_state=recurrent_state)
    if c.recurrent == "rows":
        kwargs.update(recurrent_rows=rows)
    if c.token_map:
        kwargs.update(token_indices=token_indices)
    c.exporter.export(
        checkpoint,
        raw_qkv=raw_qkv,
        conv_state=conv_state,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )

    assert torch.equal(conv_state, expected_conv)
    assert torch.equal(recurrent_state, expected_recurrent)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
def test_exporter_int64_token_offset():
    """Conv windows of rows past 2^31 // row stride need int64 input offsets.

    Qwen3.5 passes ``raw_qkv`` as the qkv slice of the packed qkvz projection,
    so its row stride is wider than its width.
    """
    width, row_stride, state_len, slot = 8192, 12288, 3, 5
    num_tokens = 2**31 // row_stride + 64
    free, _ = current_platform.mem_get_info()
    if free < num_tokens * row_stride * 2 + (1 << 30):
        pytest.skip("not enough free device memory")

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    qkvz = torch.randn(
        num_tokens, row_stride, device=device, dtype=torch.bfloat16, generator=gen
    )
    raw_qkv = qkvz[:, :width]
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    conv_state = torch.zeros(
        NUM_SLOTS, width, state_len, device=device, dtype=torch.bfloat16
    )
    offset = num_tokens - 16
    expected_conv = conv_state.clone()
    expected_conv[slot] = raw_qkv[offset - state_len : offset].T

    checkpoint = MambaPrefillCheckpointMetadata(
        torch.tensor([offset], dtype=torch.int32, device=device),
        torch.tensor([slot], dtype=torch.int32, device=device),
    )
    MambaPrefillCheckpointExporter(state_len=state_len).export(
        checkpoint, raw_qkv=raw_qkv, conv_state=conv_state, cu_seqlens=cu_seqlens
    )

    assert torch.equal(conv_state, expected_conv)
