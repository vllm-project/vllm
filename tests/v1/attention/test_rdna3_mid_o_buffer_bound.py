# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDNA3 decode fast paths must not allocate an unbounded split-K buffer.

``_pth_mid_o_buf`` is sized for the cudagraph capture size and is deliberately
never reassigned. Eager batches above it get a transient buffer, and sizing that
one with the full ``max_num_kv_splits`` makes it scale linearly with the batch:
a continuation-decode step admits 128 tokens per request, so a prefix-cache
resume of a few requests reaches hundreds of query tokens.

In production (head_size=256, 8 heads per rank, 256 splits) that came to 2 MiB
per query token: a 564-token batch asked for 1.11 GiB on a worker with ~1 GiB
of headroom. Two of four ranks raised OutOfMemoryError inside the forward, the
surviving two blocked in the next all-gather, and the engine died five minutes
later on an RPC timeout -- far from the actual fault.

CPU only: this is buffer arithmetic, no GPU required.
"""

import pytest
import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test

# ainode2 serving R128-coder at TP4: head_dim 256, 32 heads over 4 ranks,
# --attention-config '{"tq_max_kv_splits_for_cuda_graph": 256}', capture 48.
PROD = dict(num_heads=8, head_size=256, splits=256, capture=48)

# 6 requests resuming from prefix cache, ~94 tokens each: the batch that OOMed.
OOM_BATCH = 564


def _impl(num_heads, head_size, splits, capture):
    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl.num_heads = num_heads
    impl.head_size = head_size
    impl.max_num_kv_splits = splits
    impl._max_cudagraph_capture_size = capture
    return impl


def _persistent_nbytes(cfg):
    return (
        cfg["capture"] * cfg["num_heads"] * cfg["splits"] * (cfg["head_size"] + 2) * 4
    )


def _unbounded_nbytes(cfg, num_q):
    """What the buffer used to cost: splits held at max, rows follow the batch."""
    return num_q * cfg["num_heads"] * cfg["splits"] * (cfg["head_size"] + 2) * 4


def test_the_production_batch_no_longer_asks_for_a_gigabyte():
    assert _unbounded_nbytes(PROD, OOM_BATCH) > 1.1 * 1024**3  # the defect
    buf, _ = _impl(**PROD)._transient_mid_o(OOM_BATCH, torch.device("cpu"))
    assert buf.nbytes <= _persistent_nbytes(PROD)


# --max-num-batched-tokens on the node: the largest batch that can reach here.
MAX_BATCHED_TOKENS = 4160


@pytest.mark.parametrize("num_q", [49, 128, OOM_BATCH, 768, MAX_BATCHED_TOKENS])
def test_transient_never_exceeds_the_persistent_buffer(num_q):
    """Holds across the whole reachable range, up to max_num_batched_tokens."""
    buf, splits = _impl(**PROD)._transient_mid_o(num_q, torch.device("cpu"))
    assert buf.nbytes <= _persistent_nbytes(PROD)
    assert buf.shape == (num_q, PROD["num_heads"], splits, PROD["head_size"] + 2)


def test_splits_bottom_out_at_one_rather_than_zero():
    """Past the budget the split count must clamp, not divide away to an empty dim.

    One partial per query token per head is the floor of the layout, so beyond
    this point the buffer does grow again -- but only linearly in the batch, not
    multiplied by 256 splits. Sized small here so the assertion is about the
    arithmetic and not about allocating the result.
    """
    impl = _impl(num_heads=1, head_size=2, splits=256, capture=48)
    buf, splits = impl._transient_mid_o(10**6, torch.device("cpu"))
    assert splits == 1
    assert buf.shape[2] == 1


def test_a_batch_just_over_the_capture_size_keeps_its_parallelism():
    """Capping must not cost throughput where split-K is still what fills the GPU."""
    _, splits = _impl(**PROD)._transient_mid_o(PROD["capture"] + 1, torch.device("cpu"))
    assert splits >= PROD["splits"] // 2


def test_the_returned_count_matches_the_buffer_it_describes():
    """The kernels take the count and the strides separately; they must agree."""
    buf, splits = _impl(**PROD)._transient_mid_o(OOM_BATCH, torch.device("cpu"))
    assert buf.shape[2] == splits
