# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import numpy as np
import pytest
import torch

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
