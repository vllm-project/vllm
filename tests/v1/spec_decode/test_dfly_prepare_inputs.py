# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused DFly input kernels must reproduce the torch preparation they
replace: query block, sampling state, query slots and sparse reuse indices."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="HY4 DFly requires CUDA or ROCm"
)

QUERY_WIDTH = 4
MASK_TOKEN_ID = 99
MAX_MODEL_LEN = 106
BLOCK_SIZE = 4


def _reference_reuse_indices(snapshot_rows, anchors, local_window, active, width):
    """The torch `build_reuse_indices` the kernel replaced."""
    batch, topk = snapshot_rows.shape
    rows = torch.full(
        (batch, QUERY_WIDTH, width), -1, dtype=torch.int32, device=anchors.device
    )
    starts = anchors[:, None] - local_window
    history = torch.where(
        (snapshot_rows >= 0) & (snapshot_rows < starts), snapshot_rows, -1
    )
    rows[:, :, :topk] = history[:, None, :]
    window = starts + torch.arange(local_window, device=anchors.device)
    rows[:, :, topk : topk + local_window] = window.clamp(min=-1)[:, None, :]
    offsets = torch.arange(QUERY_WIDTH, device=anchors.device)
    block = anchors[:, None] + offsets
    causal = offsets[None, :] <= offsets[:, None]
    rows[:, :, topk + local_window : topk + local_window + QUERY_WIDTH] = torch.where(
        causal[None], block[:, None, :], -1
    )
    rows.masked_fill_(~active[:, None, None], -1)
    return rows.view(batch * QUERY_WIDTH, width)


@pytest.fixture
def prepared():
    from vllm.models.hy_v4.common.dfly_inputs import prepare_dfly_inputs

    device = torch.device("cuda")
    max_reqs, max_tokens, hidden_size, padded_batch = 8, 64, 16, 5
    # req0 drafts after one rejection; req1's block would pass max_model_len;
    # req2 is mid-prefill (nothing sampled). Rows 3 and 4 are graph padding.
    target_qsl = torch.tensor([0, 4, 8, 9], dtype=torch.int32, device=device)
    target_positions = torch.tensor(
        [10, 11, 12, 13, 100, 101, 102, 103, 50], dtype=torch.int64, device=device
    )
    num_rejected = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
    num_sampled = torch.tensor([2, 4, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([5, 1, 6], dtype=torch.int32, device=device)
    last_sampled = torch.arange(max_reqs * 2, device=device).view(max_reqs, 2) + 1000
    last_hidden = torch.randn(9, hidden_size, dtype=torch.bfloat16, device=device)
    mrope_delta = torch.arange(max_reqs, dtype=torch.int32, device=device) * 7
    temperature = torch.rand(max_reqs, device=device) + 0.5
    seeds = torch.arange(max_reqs, dtype=torch.int64, device=device) + 17

    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device=device)

    out = dict(
        input_ids=full((max_tokens,), -7, torch.int32),
        positions=full((max_tokens,), -7, torch.int64),
        is_padding=full((max_tokens,), False, torch.bool),
        query_start_loc=full((max_reqs + 1,), -7, torch.int32),
        seq_lens=full((max_reqs,), -7, torch.int32),
        active=full((max_reqs,), True, torch.bool),
        source_rows=full((max_reqs,), -7, torch.int64),
        anchors=full((max_reqs,), -7, torch.int64),
        target_hidden_out=torch.randn(
            max_reqs, hidden_size, dtype=torch.bfloat16, device=device
        ),
        mrope_positions=full((3, max_tokens), -7, torch.int64),
        draft_idx_mapping=full((max_reqs,), -7, torch.int64),
        out_temperature=full((max_reqs,), 0.0, torch.float32),
        out_seeds=full((max_reqs,), 0, torch.int64),
    )
    prepare_dfly_inputs(
        **out,
        target_query_start_loc=target_qsl,
        target_positions=target_positions,
        idx_mapping=idx_mapping,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        target_hidden=last_hidden,
        mrope_delta=mrope_delta,
        temperature=temperature,
        seeds=seeds,
        max_model_len=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=device),
        mask_token_id=MASK_TOKEN_ID,
        query_width=QUERY_WIDTH,
        num_reqs=3,
        padded_batch=padded_batch,
    )
    inputs = dict(
        target_qsl=target_qsl,
        target_positions=target_positions,
        num_rejected=num_rejected,
        num_sampled=num_sampled,
        idx_mapping=idx_mapping,
        last_sampled=last_sampled,
        last_hidden=last_hidden,
        mrope_delta=mrope_delta,
        temperature=temperature,
        seeds=seeds,
        padded_batch=padded_batch,
    )
    return out, inputs


def test_prepare_inputs_matches_torch_reference(prepared):
    out, inp = prepared
    device = out["positions"].device
    batch, padded_batch = 3, inp["padded_batch"]
    count = padded_batch * QUERY_WIDTH
    offsets = torch.arange(QUERY_WIDTH, device=device)

    source_rows = (inp["target_qsl"][1:] - inp["num_rejected"] - 1).long()
    anchors = inp["target_positions"][source_rows] + 1
    active = (inp["num_sampled"] > 0) & (anchors + QUERY_WIDTH <= MAX_MODEL_LEN)
    assert active.tolist() == [True, False, False]
    positions = torch.where(active[:, None], anchors[:, None] + offsets, offsets)
    ids = torch.full((batch, QUERY_WIDTH), MASK_TOKEN_ID, device=device)
    ids[:, 0] = torch.where(active, inp["last_sampled"][inp["idx_mapping"], 0], 0)
    delta = inp["mrope_delta"][inp["idx_mapping"].long()]
    rope = torch.where(active[:, None], positions + delta[:, None], 0)

    pad = padded_batch - batch
    pad_ids = torch.full((pad, QUERY_WIDTH), MASK_TOKEN_ID, device=device)
    pad_ids[:, 0] = 0
    expect_ids = torch.cat([ids, pad_ids]).flatten()
    expect_positions = torch.cat([positions, offsets.expand(pad, -1)]).flatten()
    expect_padding = torch.cat(
        [
            (~active)[:, None].expand(-1, QUERY_WIDTH),
            torch.ones(pad, QUERY_WIDTH, device=device),
        ]
    ).flatten()
    expect_rope = torch.cat(
        [rope, torch.zeros(pad, QUERY_WIDTH, device=device)]
    ).flatten()

    torch.testing.assert_close(out["input_ids"][:count].long(), expect_ids.long())
    torch.testing.assert_close(out["positions"][:count], expect_positions)
    assert out["is_padding"][:count].tolist() == expect_padding.bool().tolist()
    for dim in range(3):
        torch.testing.assert_close(
            out["mrope_positions"][dim, :count], expect_rope.long()
        )
    assert out["seq_lens"][:padded_batch].tolist() == (
        (positions[:, -1] + 1).tolist() + [QUERY_WIDTH] * pad
    )
    assert out["query_start_loc"][: padded_batch + 1].tolist() == [
        i * QUERY_WIDTH for i in range(padded_batch + 1)
    ]
    assert out["active"][:padded_batch].tolist() == active.tolist() + [False] * pad
    torch.testing.assert_close(
        out["target_hidden_out"][:batch], inp["last_hidden"][source_rows]
    )
    assert not out["target_hidden_out"][batch:padded_batch].any()
    assert out["source_rows"][:batch].tolist() == source_rows.tolist()
    assert out["anchors"][:batch].tolist() == anchors.tolist()

    # Sampling state: only active rows sample, read from fixed buffers.
    assert out["draft_idx_mapping"][:padded_batch].tolist() == [5, -1, -1, -1, -1]
    states = inp["idx_mapping"].long()
    torch.testing.assert_close(
        out["out_temperature"][states], inp["temperature"][states]
    )
    torch.testing.assert_close(out["out_seeds"][states], inp["seeds"][states])


def test_query_slots_skip_inactive_and_null_blocks(prepared):
    from vllm.models.hy_v4.common.dfly_inputs import write_dfly_query_slots

    out, inp = prepared
    device = out["positions"].device
    padded_batch = inp["padded_batch"]
    # req0's block covers positions 13..16 across blocks 3 and 4; block 4 of its
    # table is the null block.
    block_table = torch.arange(1, 41, dtype=torch.int32, device=device).view(8, 5)
    block_table[0, 4] = 0
    slots = torch.full((64,), -7, dtype=torch.int64, device=device)
    write_dfly_query_slots(
        slots,
        out["positions"],
        out["active"],
        block_table,
        BLOCK_SIZE,
        torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=device),
        QUERY_WIDTH,
        padded_batch,
    )
    req0 = [
        int(block_table[0, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE
        for p in (13, 14, 15)
    ] + [PAD_SLOT_ID]
    expected = req0 + [PAD_SLOT_ID] * (QUERY_WIDTH * (padded_batch - 1))
    assert slots[: padded_batch * QUERY_WIDTH].tolist() == expected


@pytest.mark.parametrize("layer_slots", [[0, 0], [1, 0]])
def test_reuse_indices_match_torch_reference(prepared, layer_slots):
    from vllm.models.hy_v4.common.dfly_inputs import build_dfly_reuse_indices

    out, inp = prepared
    device = out["positions"].device
    padded_batch = inp["padded_batch"]
    target_topk, local_window, width = 8, 3, 128
    snapshots = torch.randint(-1, 20, (2, 9, target_topk), device=device).int()
    indices = torch.full((2, 64, width), -7, dtype=torch.int32, device=device)
    slots = torch.tensor(layer_slots, dtype=torch.int32, device=device)
    build_dfly_reuse_indices(
        indices,
        snapshots,
        slots,
        out["active"],
        out["source_rows"],
        out["anchors"],
        target_topk,
        local_window,
        QUERY_WIDTH,
        padded_batch,
    )
    count = padded_batch * QUERY_WIDTH
    for layer, slot in enumerate(layer_slots):
        expected = _reference_reuse_indices(
            snapshots[slot][out["source_rows"][:padded_batch]],
            out["anchors"][:padded_batch],
            local_window,
            out["active"][:padded_batch],
            width,
        )
        torch.testing.assert_close(indices[layer, :count], expected)


def test_all_padding_batch_without_mrope():
    """Dummy runs and graph capture prepare an all-padding batch with no
    requests and no M-RoPE; every row must be inactive padding and the reuse
    rows must be dead."""
    from vllm.models.hy_v4.common.dfly_inputs import (
        build_dfly_reuse_indices,
        prepare_dfly_inputs,
    )

    device = torch.device("cuda")
    max_reqs, padded_batch, hidden_size = 8, 3, 16
    count = padded_batch * QUERY_WIDTH

    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device=device)

    out = dict(
        input_ids=full((64,), -7, torch.int32),
        positions=full((64,), -7, torch.int64),
        is_padding=full((64,), False, torch.bool),
        query_start_loc=full((max_reqs + 1,), -7, torch.int32),
        seq_lens=full((max_reqs,), -7, torch.int32),
        active=full((max_reqs,), True, torch.bool),
        source_rows=full((max_reqs,), -7, torch.int64),
        anchors=full((max_reqs,), -7, torch.int64),
        target_hidden_out=torch.randn(
            max_reqs, hidden_size, dtype=torch.bfloat16, device=device
        ),
        mrope_positions=None,
        draft_idx_mapping=full((max_reqs,), -7, torch.int64),
        out_temperature=full((max_reqs,), 0.0, torch.float32),
        out_seeds=full((max_reqs,), 0, torch.int64),
    )
    empty = torch.empty(0, dtype=torch.int32, device=device)
    prepare_dfly_inputs(
        **out,
        target_query_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
        target_positions=torch.empty(0, dtype=torch.int64, device=device),
        idx_mapping=empty,
        num_sampled=empty,
        num_rejected=empty,
        last_sampled=torch.zeros(max_reqs, 2, dtype=torch.int64, device=device),
        target_hidden=torch.empty(0, hidden_size, dtype=torch.bfloat16, device=device),
        mrope_delta=None,
        temperature=torch.rand(max_reqs, device=device),
        seeds=torch.arange(max_reqs, device=device),
        max_model_len=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=device),
        mask_token_id=MASK_TOKEN_ID,
        query_width=QUERY_WIDTH,
        num_reqs=0,
        padded_batch=padded_batch,
    )

    block = [0] + [MASK_TOKEN_ID] * (QUERY_WIDTH - 1)
    assert out["input_ids"][:count].tolist() == block * padded_batch
    assert out["positions"][:count].tolist() == list(range(QUERY_WIDTH)) * padded_batch
    assert out["is_padding"][:count].all()
    assert out["query_start_loc"][: padded_batch + 1].tolist() == [
        i * QUERY_WIDTH for i in range(padded_batch + 1)
    ]
    assert out["seq_lens"][:padded_batch].tolist() == [QUERY_WIDTH] * padded_batch
    assert not out["active"][:padded_batch].any()
    assert not out["target_hidden_out"][:padded_batch].any()
    assert out["draft_idx_mapping"][:padded_batch].tolist() == [-1] * padded_batch
    assert not out["out_temperature"].any()

    indices = torch.full((2, 64, 128), -7, dtype=torch.int32, device=device)
    build_dfly_reuse_indices(
        indices,
        torch.zeros(1, 8, 8, dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        out["active"],
        out["source_rows"],
        out["anchors"],
        8,
        3,
        QUERY_WIDTH,
        padded_batch,
    )
    assert (indices[:, :count] == -1).all()
    assert (indices[:, count:] == -7).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA DFly2 KV layout")
@pytest.mark.parametrize("record", [528, 584])
@pytest.mark.parametrize("large_offset", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_dsv41_context_excludes_rejected_and_inactive_tokens(
    record, large_offset, replay
):
    """Read accepted KV across pages, including byte offsets beyond int32.

    Scratch previously belonged to other requests. Invalid top-k, rejected
    positions and inactive rows must overwrite it, and leave target KV intact.
    """
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.models.deepseek_v41.nvidia.ops.dfly import gather_dfly_context

    device = "cuda"
    block_size = 16
    stride = (2**31 if large_offset else 0) + block_size * record + 128
    storage = torch.empty(
        stride + block_size * record, device=device, dtype=torch.uint8
    )
    cache = storage.as_strided((2, block_size, record), (stride, record, 1))
    torch.manual_seed(42)
    latent = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    quantize_and_insert_k_cache(
        latent,
        cache,
        torch.arange(32, device=device),
        block_size=block_size,
        bytes_per_token=record,
    )
    before = cache.clone()
    # Decode the two record formats independently of the gather kernel.
    if record == 528:
        fp8 = before[:, :, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = torch.exp2(before[:, :, 512:].float() - 127).repeat_interleave(32, -1)
        expected = (fp8 * scales).to(torch.bfloat16).reshape(32, 512)
    else:
        pages = before.reshape(2, -1)
        data = pages[:, : block_size * 576].reshape(2, block_size, 576)
        fp8 = data[:, :, :448].contiguous().view(torch.float8_e4m3fn).float()
        scales = pages[:, block_size * 576 :].reshape(2, block_size, 8)[:, :, :7]
        scales = torch.exp2(scales.float() - 127).repeat_interleave(64, -1)
        tail = data[:, :, 448:].contiguous().view(torch.bfloat16)
        expected = torch.cat(((fp8 * scales).to(torch.bfloat16), tail), -1).reshape(
            32, 512
        )
    # Offset views and strided tables exercise the same contracts as batch buffers.
    tables = torch.tensor(
        [[9, 9, 9], [1, 0, 9], [0, 1, 9], [0, 1, 9]], device=device, dtype=torch.int32
    )[1:, :2]
    topk = torch.tensor(
        [[9, 9, 9, 9], [0, 15, 16, 18], [2, -1, 32, 3]],
        device=device,
        dtype=torch.int32,
    )
    rows = torch.tensor([99, 1, 2, 100], device=device)[1:]
    replay_rows = None
    if replay:
        rows = torch.tensor([37, 108, 100], device=device)
        replay_rows = torch.tensor([30, 37, 108], device=device)
    anchors = torch.tensor([18, 3, 16], device=device)
    active = torch.tensor([True, True, False], device=device)
    output = torch.full((3, 10, 512), 17, device=device, dtype=torch.bfloat16)
    indices = torch.full((3, 128), 17, device=device, dtype=torch.int32)
    for offset, window in ((0, False), (4, True)):
        gather_dfly_context(
            cache,
            tables,
            block_size,
            topk,
            rows,
            anchors,
            active,
            output,
            indices,
            offset=offset,
            width=4,
            window=window,
            replay_rows=replay_rows,
        )
    expected_positions = [
        [16, 31, 0, -1, 30, 31, 0, 1],
        [2, -1, -1, -1, -1, 0, 1, 2],
        [-1] * 8,
    ]
    expected_output = torch.zeros_like(output[:, :8])
    expected_indices = torch.full_like(indices[:, :8], -1)
    for request, positions in enumerate(expected_positions):
        for col, physical in enumerate(positions):
            if physical >= 0:
                expected_output[request, col] = expected[physical]
                expected_indices[request, col] = request * 10 + col
    torch.testing.assert_close(output[:, :8], expected_output, rtol=0, atol=0)
    torch.testing.assert_close(indices[:, :8], expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    assert (output[:, 8:] == 17).all()
    assert (indices[:, 8:] == 17).all()
