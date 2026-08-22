# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness for flash-decoding sequence partitioning in the Triton paged
decode fallback.

``kernel_paged_attention_2d`` launches ``(num_seqs, num_kv_heads)`` -- during
decode a single-digit number of programs, each walking the whole sequence. The
partitioned path splits the sequence across programs that accumulate an
unnormalised softmax numerator plus its running max and denominator, and a
reducer combines them.

Reference is the unpartitioned kernel on identical tensors: "correct" here means
"agrees with what it replaces".

Three things this is specifically shaped to catch:

* **Partitions past the end of a sequence.** They exit without storing, and the
  reducer is responsible for never reading them (it recomputes ``num_parts``
  from the true per-sequence length). If that contract breaks, uninitialised
  scratch enters the softmax. Ragged batches make surplus partitions the common
  case rather than an edge case.
* **The cudagraph bound.** The grid is sized from the block table's width, not
  from the runtime ``max_seq_len``, because
  ``build_for_cudagraph_capture()`` sets ``seq_lens`` to 1 and a grid sized from
  that would silently truncate long sequences at replay. The wide-block-table
  case pins the "bound much larger than the live sequence" shape that a captured
  graph actually replays.
* **A vacuous pass.** ``test_partitioning_is_actually_engaged`` asserts the path
  is on for the shapes under test. Without it, a disabled partitioner sends both
  arms down the same code and the comparison succeeds no matter what.

The hybrid interleaved KV layout is used throughout (``kv`` re-strided so K and V
alternate per block, exactly as
``GPUModelRunner._update_hybrid_attention_mamba_layout`` does), because that plus
block_size 784 is what a hybrid GDN+attention model gets in practice.
"""

import math

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform

NUM_KV_HEADS = 2
NUM_QUERIES_PER_KV = 8
NUM_HEADS = NUM_KV_HEADS * NUM_QUERIES_PER_KV
HEAD_SIZE = 256
PHYSICAL_BLOCK_SIZE = 784

# Both arms are the same bf16 kernel differing only in reduction order, so they
# agree far better than two independent kernels would. This bound still catches
# a dropped or double-counted partition, which is an order-1 error.
REL_TOL = 2e-2

SEQ_LEN_CASES = [
    pytest.param([1000], 0, id="short-single"),
    pytest.param([4096], 0, id="medium-single"),
    pytest.param([40960], 0, id="long-single"),
    pytest.param([200000], 0, id="very-long-single"),
    pytest.param([1000, 4096, 12345], 0, id="ragged-small"),
    pytest.param([1, 700, 33000, 120000], 0, id="ragged-wide-spread"),
    pytest.param([65536, 3, 65535], 0, id="ragged-long-short-long"),
    # Block table sized for a 256K model while the live sequences are tiny --
    # the shape a captured cudagraph replays.
    pytest.param([900, 2048], 335, id="wide-block-table-captured-graph"),
]


def _build(seq_lens, wide_blocks, dtype, device):
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    num_seqs = len(seq_lens)
    max_len = max(seq_lens)
    blocks_per_seq = max(
        (max_len + PHYSICAL_BLOCK_SIZE - 1) // PHYSICAL_BLOCK_SIZE, wide_blocks
    )
    num_blocks = num_seqs * blocks_per_seq + 4

    torch.manual_seed(0x5EED)
    kv = torch.randn(2, num_blocks, PHYSICAL_BLOCK_SIZE, NUM_KV_HEADS,
                     HEAD_SIZE, dtype=dtype, device=device) * 0.5
    # Hybrid interleaved layout: K and V alternate per block.
    hidden = kv.shape[2:].numel()
    kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv, NUM_KV_HEADS, HEAD_SIZE)

    block_table = torch.randperm(
        num_blocks, device=device)[: num_seqs * blocks_per_seq].reshape(
            num_seqs, blocks_per_seq).to(torch.int32)
    return {
        "query": torch.randn(num_seqs, NUM_HEADS, HEAD_SIZE, dtype=dtype,
                             device=device) * 0.5,
        "key": None,
        "value": None,
        "kv_cache_dtype": "auto",
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "query_start_loc": torch.arange(num_seqs + 1, dtype=torch.int32,
                                        device=device),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "max_seq_len": max_len,
        "max_query_len": 1,
        "k_scale": torch.tensor(1.0, device=device),
        "v_scale": torch.tensor(1.0, device=device),
        "sm_scale": 1.0 / math.sqrt(HEAD_SIZE),
    }


def _run(kwargs, enabled, monkeypatch):
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as mod

    monkeypatch.setattr(envs, "VLLM_TRITON_PA_SEQ_PARTITION", enabled)
    mod._seq_partition_enabled.cache_clear()
    out = torch.zeros_like(kwargs["query"])
    mod.chunked_prefill_paged_decode(output=out, **kwargs)
    mod._seq_partition_enabled.cache_clear()
    return out.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_lens,wide_blocks", SEQ_LEN_CASES)
def test_partitioned_matches_unpartitioned(seq_lens, wide_blocks, monkeypatch):
    device = torch.device("cuda")
    kwargs = _build(seq_lens, wide_blocks, torch.bfloat16, device)

    baseline = _run(kwargs, enabled=False, monkeypatch=monkeypatch)
    partitioned = _run(kwargs, enabled=True, monkeypatch=monkeypatch)

    denom = max(baseline.abs().max().item(), 1e-6)
    rel = (partitioned - baseline).abs().max().item() / denom
    assert rel < REL_TOL, (
        f"partitioned decode disagrees with the unpartitioned kernel for "
        f"seq_lens={seq_lens}: max_rel={rel:.3e}. An order-1 value means a "
        "partition was dropped, double-counted, or read before it was written."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_lens,wide_blocks", SEQ_LEN_CASES)
def test_partitioning_is_actually_engaged(seq_lens, wide_blocks, monkeypatch):
    """Guard against a vacuous comparison.

    If the partitioner declines, both arms of the test above run identical code
    and pass regardless of kernel correctness. Every case here is expected to
    partition -- including the wide-block-table one, whose bound comes from the
    block table rather than the (short) live sequences.
    """
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as mod

    monkeypatch.setattr(envs, "VLLM_TRITON_PA_SEQ_PARTITION", True)
    mod._seq_partition_enabled.cache_clear()
    kwargs = _build(seq_lens, wide_blocks, torch.bfloat16, torch.device("cuda"))
    bound = kwargs["block_table"].shape[1] * PHYSICAL_BLOCK_SIZE
    part = mod._choose_partition_size(
        len(seq_lens), NUM_KV_HEADS, bound, 32,
        num_query_heads=NUM_HEADS, head_size_padded=HEAD_SIZE,
    )
    mod._seq_partition_enabled.cache_clear()
    assert part > 0, (
        f"partitioning declined for seq_lens={seq_lens} (bound={bound}); the "
        "correctness comparison would be vacuous"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_saturated_batch_declines_partitioning(monkeypatch):
    """A batch that already fills the device should not pay for partitioning.

    This is the knob that keeps scratch bounded: partition count is chosen from
    a memory budget, and a large batch is refused outright rather than being
    given one partition each.
    """
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as mod

    monkeypatch.setattr(envs, "VLLM_TRITON_PA_SEQ_PARTITION", True)
    mod._seq_partition_enabled.cache_clear()
    part = mod._choose_partition_size(
        4096, NUM_KV_HEADS, 262_640, 32,
        num_query_heads=NUM_HEADS, head_size_padded=HEAD_SIZE,
    )
    mod._seq_partition_enabled.cache_clear()
    assert part == 0, "a saturating batch should decline partitioning"
