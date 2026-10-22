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
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.gpu import cp_utils
from vllm.v1.worker.gpu.input_batch import (
    BatchIndexUploader,
    InputBatch,
    InputBuffers,
)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_interleave", [1, 16])
def test_prepare_dcp_local_seq_lens_matches_reference(
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


requires_cuda_uva = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_uva_available()),
    reason="BatchIndexUploader needs CUDA and UVA",
)


@requires_cuda_uva
@pytest.mark.parametrize(
    ("num_reqs", "num_tokens", "num_tokens_after_padding", "mark_padding"),
    [
        (4, 16, 16, True),  # uniform spec decode, no padding
        (5, 20, 32, True),  # padded to a captured size
        (3, 9, 9, False),  # VLLM_MOE_SKIP_PADDING off
        (1, 1, 1, True),
        (8, 1500, 1536, True),  # padding tokens span several kernel blocks
    ],
)
def test_batch_index_uploader_matches_h2d_uploads(
    num_reqs: int, num_tokens: int, num_tokens_after_padding: int, mark_padding: bool
):
    """One kernel must leave the same device state as the separate uploads."""
    device = torch.device("cuda:0")
    max_num_reqs, max_num_tokens = 16, 2048
    rng = np.random.default_rng(num_reqs)
    idx_mapping_np = rng.permutation(max_num_reqs)[:num_reqs].astype(np.int64)
    cu_num_logits_np = np.concatenate(
        [[0], np.cumsum(rng.integers(1, 5, num_reqs))]
    ).astype(np.int32)
    query_start_loc_np = np.full(max_num_reqs + 1, num_tokens, dtype=np.int32)
    query_start_loc_np[: num_reqs + 1] = np.linspace(0, num_tokens, num_reqs + 1)
    stale_qsl = torch.randint(-9, 99, (max_num_reqs + 1,), dtype=torch.int32)
    stale_padding = torch.rand(max_num_tokens) < 0.5

    ref_qsl = stale_qsl.to(device)
    ref_padding = stale_padding.to(device)
    ref_idx_mapping = async_tensor_h2d(idx_mapping_np, device=device, dtype=torch.int64)
    ref_cu_num_logits = async_tensor_h2d(cu_num_logits_np, device=device)
    async_tensor_h2d(query_start_loc_np, out=ref_qsl)
    if mark_padding:
        ref_padding[:num_tokens].fill_(False)
        ref_padding[num_tokens:num_tokens_after_padding].fill_(True)

    new_qsl = stale_qsl.to(device)
    new_padding = stale_padding.to(device)
    uploader = BatchIndexUploader(max_num_reqs, device)
    idx_mapping, cu_num_logits = uploader.upload(
        idx_mapping_np,
        cu_num_logits_np,
        query_start_loc_np,
        num_tokens,
        new_qsl,
        new_padding[:num_tokens_after_padding] if mark_padding else None,
        num_tokens_after_padding,
    )
    torch.accelerator.synchronize()

    for ref, new in (
        (ref_idx_mapping, idx_mapping),
        (ref_cu_num_logits, cu_num_logits),
        (ref_qsl, new_qsl),
        (ref_padding, new_padding),
    ):
        assert ref.dtype == new.dtype
        assert torch.equal(ref, new)


@requires_cuda_uva
def test_batch_index_uploader_keeps_staged_inputs_until_read():
    """Wrapping the slot ring must wait for the kernel still reading a slot."""
    device = torch.device("cuda:0")
    uploader = BatchIndexUploader(8, device, num_slots=2)
    steps = [
        (np.array([3, 1, 2], dtype=np.int64), np.array([0, 4, 8, 12], dtype=np.int32)),
        (np.array([7, 0], dtype=np.int64), np.array([0, 1, 5], dtype=np.int32)),
        (np.array([5], dtype=np.int64), np.array([0, 4], dtype=np.int32)),
    ]
    # Compile the kernel first so that the uploads below outpace the sleep.
    qsl = torch.empty(9, dtype=torch.int32, device=device)
    uploader.upload(*steps[0], steps[0][1], 99, qsl, None, 0)
    torch.accelerator.synchronize()
    stream = torch.cuda.Stream(device=device)
    outputs = []
    with torch.cuda.stream(stream):
        # Keep both slots' kernels pending while the third upload wraps around.
        torch.cuda._sleep(200_000_000)
        for idx_mapping_np, cu_np in steps:
            qsl = torch.empty(9, dtype=torch.int32, device=device)
            outputs.append(
                (*uploader.upload(idx_mapping_np, cu_np, cu_np, 99, qsl, None, 0), qsl)
            )
    stream.synchronize()

    for (idx_mapping, cu_num_logits, qsl), (idx_np, cu_np) in zip(outputs, steps):
        assert idx_mapping.tolist() == idx_np.tolist()
        assert cu_num_logits.tolist() == cu_np.tolist()
        assert qsl.tolist() == cu_np.tolist() + [99] * (9 - len(cu_np))
