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
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

DEVICE = current_platform.device_type


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


def test_maybe_prepare_dcp_local_seq_lens_uses_shared_buffer(monkeypatch):
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
    batch.dcp_local_seq_lens = cp_utils.maybe_prepare_dcp_local_seq_lens(
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


def test_maybe_prepare_dcp_local_seq_lens_clears_stale_metadata(monkeypatch):
    """dcp_size == 1 resets the field to None instead of leaving a leftover.

    DCP is toggled per deployment, and batches are recycled; a value from an
    earlier DCP batch would otherwise be consumed as if it were current.
    """
    buffers = InputBuffers(max_num_reqs=2, max_num_tokens=2, device=torch.device("cpu"))
    batch = InputBatch.make_dummy(1, 1, buffers)
    batch.dcp_local_seq_lens = buffers.dcp_local_seq_lens[:1]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not run with dcp_size == 1")

    monkeypatch.setattr(cp_utils, "_dcp_local_seq_lens_kernel", fail_if_called)
    batch.dcp_local_seq_lens = cp_utils.maybe_prepare_dcp_local_seq_lens(
        buffers.dcp_local_seq_lens,
        batch.seq_lens,
        batch.num_reqs,
        1,
        0,
        1,
        num_reqs_padded=batch.num_reqs_after_padding,
    )

    assert batch.dcp_local_seq_lens is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_interleave", [1, 16])
def test_maybe_prepare_dcp_local_seq_lens_matches_reference(
    dcp_size: int, cp_interleave: int
):
    """Every rank's local lengths must equal the reference torch formula."""
    device = torch.device("cuda:0")
    seq_lens_np = np.array([7, 16, 33, 64, 512, 1023], dtype=np.int32)
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=32, device=device)
    batch = InputBatch.make_dummy(6, 12, buffers)
    buffers.seq_lens[: len(seq_lens_np)] = torch.from_numpy(seq_lens_np).to(device)

    for dcp_rank in range(dcp_size):
        buffers.dcp_local_seq_lens.fill_(-1)
        batch.dcp_local_seq_lens = cp_utils.maybe_prepare_dcp_local_seq_lens(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize(
    "num_new_sampled,drafts_per_req",
    [(1, [0, 0, 0, 0]), (1, [0, 3, 2, 1]), (0, [1, 3, 2, 1])],
    ids=["bonus-only", "bonus-plus-drafts", "drafts-only"],
)
def test_prepare_decode_inputs_matches_per_request_reference(
    num_new_sampled, drafts_per_req
):
    """The fused kernel must write what the three per-request kernels wrote.

    Covers positions, seq_lens, sampled/draft input ids, logits indices and the
    expanded per-logit request mapping. Request 0 is mid-prefill so its ids must
    survive untouched, and request 2 has more query rows than logits rows so the
    ``end - num_logits`` placement is exercised.
    """
    from vllm.v1.worker.gpu.input_batch import prepare_decode_inputs

    torch.manual_seed(0)
    device = torch.device("cuda")
    max_num_reqs, num_spec = 8, 3
    idx_mapping = torch.tensor([5, 2, 7, 0], dtype=torch.int32)
    num_computed = torch.tensor([9, 0, 40, 0, 0, 16, 0, 60], dtype=torch.int32)
    prefill_len = torch.tensor([0, 0, 0, 9, 0, 999, 0, 0], dtype=torch.int32)
    num_logits = [d + num_new_sampled for d in drafts_per_req]
    query_lens = [n + (r == 2) for r, n in enumerate(num_logits)]
    cu = torch.tensor([0, *torch.tensor(num_logits).cumsum(0)], dtype=torch.int32)
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0)], dtype=torch.int32)
    num_tokens, total_logits = int(qsl[-1]), int(cu[-1])
    last_sampled = torch.randint(1, 999, (max_num_reqs,), dtype=torch.int64)
    drafts = torch.randint(1, 999, (max_num_reqs, num_spec), dtype=torch.int64)

    pos_ref = torch.zeros(num_tokens, dtype=torch.int64)
    seq_ref = torch.zeros(max_num_reqs, dtype=torch.int32)
    ids_ref = torch.zeros(num_tokens, dtype=torch.int32)
    li_ref, ex_ref, lp_ref = [], [], []
    for r, st in enumerate(idx_mapping.tolist()):
        start, end = int(qsl[r]), int(qsl[r + 1])
        seq = int(num_computed[st]) + end - start
        seq_ref[r] = seq
        pos_ref[start:end] = torch.arange(int(num_computed[st]), seq)
        n = int(cu[r + 1]) - int(cu[r])
        li_ref += list(range(end - n, end))
        ex_ref += [st] * n
        lp_ref += list(range(n))
        if seq <= int(prefill_len[st]):
            continue
        if num_new_sampled and seq - n >= int(prefill_len[st]):
            ids_ref[end - n] = last_sampled[st]
        if n - num_new_sampled:
            ids_ref[end - n + num_new_sampled : end] = drafts[st, : n - num_new_sampled]

    pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.full((max_num_reqs,), 99, dtype=torch.int32, device=device)
    ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    logits_indices, expanded, local_pos = prepare_decode_inputs(
        idx_mapping.to(device),
        qsl.to(device),
        num_computed.to(device),
        pos,
        seq_lens,
        ids,
        last_sampled.to(device),
        prefill_len.to(device),
        drafts.to(device),
        cu.to(device),
        total_logits,
        num_new_sampled,
        max_expand_len=(num_spec + 1) if any(drafts_per_req) else None,
    )
    assert torch.equal(pos.cpu(), pos_ref)
    assert torch.equal(seq_lens.cpu(), seq_ref)
    assert torch.equal(ids.cpu(), ids_ref)
    assert logits_indices.cpu().tolist() == li_ref
    if any(drafts_per_req):
        assert expanded.cpu().tolist() == ex_ref
        assert local_pos.cpu().tolist() == lp_ref
    else:
        assert expanded is None and local_pos is None
