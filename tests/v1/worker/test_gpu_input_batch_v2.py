# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.gpu import cp_utils
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, post_update

DEVICE = current_platform.device_type


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
@pytest.mark.parametrize("with_drafts", [False, True])
def test_post_update_preserves_freed_slots_and_maps_pp_drafts(with_drafts: bool):
    """Deferred PP updates must write live request slots, never freed rows."""
    device = torch.device(DEVICE)
    indices = torch.tensor([-1, 3, 1], dtype=torch.int32, device=device)
    computed = torch.full((4,), 10, dtype=torch.int32, device=device)
    last = torch.full((4,), -1, dtype=torch.int64, device=device)
    total_len = torch.full((4,), 4, dtype=torch.int32, device=device)
    tokens = torch.full((4, 16), -1, dtype=torch.int32, device=device)
    sampled = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]], device=device)
    drafts = torch.tensor([[101, 102], [201, 202], [301, 302]], device=device)
    draft_table = torch.full((4, 2), -1, dtype=torch.int64, device=device)

    post_update(
        indices,
        computed,
        last,
        None,
        sampled,
        torch.tensor([1, 2, 1], dtype=torch.int32, device=device),
        torch.tensor([0, 2, 1], dtype=torch.int32, device=device),
        None,
        tokens,
        total_len,
        drafts if with_drafts else None,
        draft_table if with_drafts else None,
    )

    assert computed.tolist() == [10, 9, 10, 8]
    assert last.tolist() == [-1, 31, -1, 22]
    assert total_len.tolist() == [4, 5, 4, 6]
    expected_tokens = torch.full_like(tokens, -1)
    expected_tokens[1, 4] = 31
    expected_tokens[3, 4:6] = sampled[1, :2]
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    expected_drafts = torch.full_like(draft_table, -1)
    if with_drafts:
        expected_drafts[1] = drafts[2]
        expected_drafts[3] = drafts[1]
    torch.testing.assert_close(draft_table, expected_drafts, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA top-k.")
@pytest.mark.parametrize("is_padding", [True, False])
def test_make_dummy_padding_controls_moe_routing(monkeypatch, is_padding: bool):
    """Dummy tokens marked as padding are dropped by the MoE router (top-k id
    -1), which is wanted for idle DP ranks. Profile runs must not mark them, or
    no token reaches the experts and MoE memory is never profiled."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    num_tokens = 16
    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(4, num_tokens, buffers, is_padding=is_padding)
    hidden_states = torch.randn(num_tokens, 4, device=DEVICE)
    router_logits = torch.randn(num_tokens, 8, device=DEVICE)

    with set_forward_context(None, VllmConfig(), is_padding=batch.is_padding):
        _, topk_ids, _ = fused_topk(hidden_states, router_logits, 2, False)

    assert bool((topk_ids == -1).all()) is is_padding
    assert bool((topk_ids >= 0).all()) is not is_padding


def test_prepare_dcp_local_seq_lens_uses_shared_buffer(monkeypatch):
    """The batch must view the caller-owned buffer, sliced to padded length.

    Runtime (and capture) paths all funnel through this helper; attention
    metadata indexes padded rows, so the view must reach
    num_reqs_after_padding, and it must alias the persistent buffer so CUDA
    graph replay sees the recomputed values.
    """
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=4, device=torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 4, buffers)
    batch.num_reqs_after_padding = 4

    def fake_kernel(
        output,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        block_size,
    ):
        assert output is buffers.dcp_local_seq_lens
        assert seq_lens is batch.seq_lens
        assert (num_reqs, dcp_size, dcp_rank, cp_interleave) == (2, 4, 1, 16)
        assert (max_num_reqs, block_size) == (4, 128)
        output[:] = torch.tensor([1, 2, 0, 0], dtype=output.dtype)

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)
            return fake_kernel

    monkeypatch.setattr(cp_utils, "_dcp_local_seq_lens_kernel", FakeKernel())
    batch.dcp_local_seq_lens = cp_utils.prepare_dcp_local_seq_lens(
        buffers.dcp_local_seq_lens,
        batch.seq_lens,
        batch.num_reqs,
        4,
        1,
        16,
        num_reqs_padded=batch.num_reqs_after_padding,
    )

    assert batch.dcp_local_seq_lens is not None
    assert batch.dcp_local_seq_lens.data_ptr() == buffers.dcp_local_seq_lens.data_ptr()
    assert batch.dcp_local_seq_lens.tolist() == [1, 2, 0, 0]


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="triton kernel needs CUDA or XPU",
)
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_interleave", [1, 16])
def test_prepare_dcp_local_seq_lens_matches_reference(
    dcp_size: int, cp_interleave: int
):
    """Every rank's local lengths must equal the reference torch formula."""
    device = torch.device(DEVICE)
    seq_lens_np = np.array([7, 16, 33, 64, 512, 1023], dtype=np.int32)
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=32, device=device)
    batch = InputBatch.make_dummy(6, 12, buffers)
    buffers.seq_lens[: len(seq_lens_np)] = torch.from_numpy(seq_lens_np).to(device)

    for dcp_rank in range(dcp_size):
        buffers.dcp_local_seq_lens.fill_(-1)
        batch.dcp_local_seq_lens = cp_utils.prepare_dcp_local_seq_lens(
            buffers.dcp_local_seq_lens,
            batch.seq_lens,
            batch.num_reqs,
            dcp_size,
            dcp_rank,
            cp_interleave,
            num_reqs_padded=batch.num_reqs_after_padding,
        )
        expected = get_dcp_local_seq_lens(
            torch.from_numpy(seq_lens_np), dcp_size, dcp_rank, cp_interleave
        )
        assert batch.dcp_local_seq_lens is not None
        assert torch.equal(batch.dcp_local_seq_lens.cpu(), expected.to(torch.int32))
