# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness for the partitioned speculative-decode VERIFY path.

With speculative decoding on, every step has ``query_len = num_spec + 1``, so
``max_query_len > 1`` and ``chunked_prefill_paged_decode`` routes to
``context_attention_fwd``. The paged-decode kernels are never reached. That
makes this -- not the decode path -- the hot kernel for a spec-decode
deployment, and ``_fwd_kernel``'s grid of
``(batch, head, cdiv(max_input_len, BLOCK_M))`` collapses to one program per head
for a short query, each scanning the entire cached context serially.

The specialised route splits the cached-context scan across programs and has a
reducer combine the partials and then apply the current chunk in a single tile.
Reference is ``_fwd_kernel`` on identical tensors.

Shaped to catch, specifically:

* **The single-tile assumption.** The reducer handles the new tokens in one
  ``BLOCK_M`` tile, valid only because the route is gated on
  ``query_len <= VERIFY_MAX_Q``. ``query_len`` 17 exercises a tile that is more
  than half empty; ``query_len`` 1 must be skipped entirely under
  ``skip_decode`` and left to the decode kernel.
* **Mixed batches.** Ragged query lengths in one batch, including a
  ``query_len == 1`` row that must be skipped while its neighbours are
  processed. Getting the skip wrong writes garbage into a row nothing else
  touches, which no aggregate check would notice.
* **Zero cached context.** ``ctx_len == 0`` means no partitions run at all and
  the reducer starts from ``-inf``/0. ``exp(-inf - -inf)`` is NaN, so this is
  the case that catches a missing mask on the empty-partition path.
* **Both causal modes.** DFlash-style drafters attend bidirectionally over the
  drafted tokens (``causal=False``), which is what production uses; ordinary
  speculative verify is causal.
* **A vacuous pass.** ``test_partitioning_is_actually_engaged`` asserts the
  route is taken, since a declined route sends both arms through
  ``_fwd_kernel`` and the comparison succeeds unconditionally.
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

# Same kernel family, different reduction order. An order-1 value means a
# partition was dropped or the current chunk was applied twice.
REL_TOL = 2e-2

CASES = [
    pytest.param([40960], [9], False, 0, id="41k-dflash-bidirectional"),
    pytest.param([102400], [9], False, 0, id="100k"),
    pytest.param([200000], [9], False, 0, id="200k"),
    pytest.param([40960], [9], True, 0, id="41k-causal"),
    pytest.param([40960], [1], False, 0, id="query-len-1-must-be-skipped"),
    pytest.param([4096], [17], False, 0, id="query-len-17-partial-tile"),
    pytest.param([102400, 3000, 60000], [9, 9, 9], False, 0, id="ragged-ctx"),
    pytest.param([102400, 3000, 60000], [9, 1, 5], False, 0,
                 id="ragged-query-with-decode-row"),
    pytest.param([0], [9], False, 0, id="no-cached-context"),
    pytest.param([8192], [9], False, 335, id="wide-block-table-captured-graph"),
]


def _build(ctx_lens, qlens, wide_blocks, dtype, device):
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    batch = len(ctx_lens)
    totals = [c + q for c, q in zip(ctx_lens, qlens)]
    blocks_per_seq = max(
        max((t + PHYSICAL_BLOCK_SIZE - 1) // PHYSICAL_BLOCK_SIZE
            for t in totals),
        wide_blocks,
    )
    num_blocks = batch * blocks_per_seq + 4

    torch.manual_seed(0x5EC0)
    kv = torch.randn(2, num_blocks, PHYSICAL_BLOCK_SIZE, NUM_KV_HEADS,
                     HEAD_SIZE, dtype=dtype, device=device) * 0.5
    hidden = kv.shape[2:].numel()
    kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
    k_cache, v_cache = PagedAttention.split_kv_cache(kv, NUM_KV_HEADS, HEAD_SIZE)

    b_loc = torch.randperm(
        num_blocks, device=device)[: batch * blocks_per_seq].reshape(
            batch, blocks_per_seq).to(torch.int32)
    tot_q = sum(qlens)
    starts = [0]
    for q in qlens:
        starts.append(starts[-1] + q)
    return {
        "q": torch.randn(tot_q, NUM_HEADS, HEAD_SIZE, dtype=dtype,
                         device=device) * 0.5,
        "k": torch.randn(tot_q, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype,
                         device=device) * 0.5,
        "v": torch.randn(tot_q, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype,
                         device=device) * 0.5,
        "kv_cache_dtype": "auto",
        "k_cache": k_cache,
        "v_cache": v_cache,
        "b_loc": b_loc,
        "b_start_loc": torch.tensor(starts, dtype=torch.int32, device=device),
        "b_seq_len": torch.tensor(totals, dtype=torch.int32, device=device),
        "max_seq_len": max(totals),
        "max_input_len": max(qlens),
        "k_scale": torch.tensor(1.0, device=device),
        "v_scale": torch.tensor(1.0, device=device),
        "sm_scale": 1.0 / math.sqrt(HEAD_SIZE),
        "skip_decode": True,
    }


def _run(kwargs, causal, enabled, monkeypatch):
    from vllm.v1.attention.ops import prefix_prefill as mod

    monkeypatch.setattr(envs, "VLLM_TRITON_VERIFY_CTX_PARTITION", enabled)
    mod._verify_partition_enabled.cache_clear()
    out = torch.zeros_like(kwargs["q"])
    mod.context_attention_fwd(o=out, causal=causal, **kwargs)
    mod._verify_partition_enabled.cache_clear()
    return out.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("ctx_lens,qlens,causal,wide_blocks", CASES)
def test_partitioned_verify_matches_fwd_kernel(
    ctx_lens, qlens, causal, wide_blocks, monkeypatch
):
    device = torch.device("cuda")
    kwargs = _build(ctx_lens, qlens, wide_blocks, torch.bfloat16, device)

    baseline = _run(kwargs, causal, enabled=False, monkeypatch=monkeypatch)
    partitioned = _run(kwargs, causal, enabled=True, monkeypatch=monkeypatch)

    denom = max(baseline.abs().max().item(), 1e-6)
    rel = (partitioned - baseline).abs().max().item() / denom
    assert rel < REL_TOL, (
        f"partitioned verify disagrees with _fwd_kernel for ctx={ctx_lens}, "
        f"q={qlens}, causal={causal}: max_rel={rel:.3e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_partitioning_is_actually_engaged(monkeypatch):
    """A declined route would make every comparison above vacuous."""
    from vllm.v1.attention.ops import prefix_prefill as mod

    monkeypatch.setattr(envs, "VLLM_TRITON_VERIFY_CTX_PARTITION", True)
    mod._verify_partition_enabled.cache_clear()
    part = mod._choose_verify_partition(1, NUM_HEADS, 262_640,
                                        block_m=16, head_dim=HEAD_SIZE)
    mod._verify_partition_enabled.cache_clear()
    assert part > 0, "verify-path partitioning declined at the production bound"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_long_query_falls_through_to_fwd_kernel(monkeypatch):
    """Ordinary prefill must not take the specialised route.

    The reducer applies the current chunk in a single ``BLOCK_M`` tile, so the
    route is only valid while ``query_len <= VERIFY_MAX_Q``. A real prefill has
    thousands of query tokens and must be left to ``_fwd_kernel``.
    """
    from vllm.v1.attention.ops import prefix_prefill as mod

    device = torch.device("cuda")
    monkeypatch.setattr(envs, "VLLM_TRITON_VERIFY_CTX_PARTITION", True)
    mod._verify_partition_enabled.cache_clear()
    kwargs = _build([4096], [mod.VERIFY_MAX_Q + 1], 0, torch.bfloat16, device)
    baseline = _run(kwargs, False, enabled=False, monkeypatch=monkeypatch)
    routed = _run(kwargs, False, enabled=True, monkeypatch=monkeypatch)
    mod._verify_partition_enabled.cache_clear()
    # Above the gate both calls run _fwd_kernel, so they must be bit-identical.
    assert torch.equal(baseline, routed), (
        "a query longer than VERIFY_MAX_Q took the specialised single-tile "
        "route; its current-chunk handling cannot cover more than one tile"
    )
