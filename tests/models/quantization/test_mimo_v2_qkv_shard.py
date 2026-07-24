# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for MiMo-V2 fused fp8 QKV sharding (_shard_fp8_qkv_proj).

The Pro checkpoint stores the fused QKV as ``num_kv_heads`` block-fp8 stripes,
one per KV head, each ordered ``[Q | K | V]``. This test builds such a
checkpoint with a distinct, identifiable weight per head and verifies that
``_shard_fp8_qkv_proj`` hands every TP rank exactly the heads it should own —
for tp < num_kv_heads (groups merged), tp == num_kv_heads (plain chunk), and
tp > num_kv_heads (KV heads replicated, Q heads sub-sharded).
"""

import pytest
import torch

from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj

# MiMo-V2.5-Pro attention shape.
NUM_HEADS = 128
NUM_KV = 8
HEAD_DIM = 192  # qk head_dim
V_HEAD_DIM = 128  # asymmetric v head_dim
HIDDEN = 6144
BLOCK = 128
FP8 = torch.float8_e4m3fn

Q_PER_GROUP = (NUM_HEADS // NUM_KV) * HEAD_DIM  # 3072
ROWS_PER_GROUP = Q_PER_GROUP + HEAD_DIM + V_HEAD_DIM  # 3392


def _head_block(seed: int, rows: int) -> torch.Tensor:
    # Distinct unit-scale random matrix per head so heads are well separated
    # under relative-L2 (~sqrt(2) apart), while a single fp8 round-trip is ~0.03.
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=g)


def _block_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Block (128x128) fp8 quant matching the checkpoint export: per-block
    # scale = amax / fp8_max (the dequant multiplier stored as weight_scale_inv);
    # the final row/col block may be partial (the stripe is 3392 rows = 26.5
    # blocks), which is exactly why scaled_quantize cannot build this.
    rows, cols = w.shape
    nr, nc = (rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK
    fp8_max = torch.finfo(FP8).max
    wq = torch.zeros(rows, cols, dtype=torch.float32)
    scale = torch.zeros(nr, nc, dtype=torch.float32)
    for i in range(nr):
        for j in range(nc):
            r0, r1 = i * BLOCK, min((i + 1) * BLOCK, rows)
            c0, c1 = j * BLOCK, min((j + 1) * BLOCK, cols)
            blk = w[r0:r1, c0:c1]
            s = blk.abs().max().clamp(min=1e-12) / fp8_max
            scale[i, j] = s
            wq[r0:r1, c0:c1] = blk / s
    return wq.to(FP8), scale


def _build_checkpoint():
    """Return (w_full fp8, s_full) plus the float ground-truth head blocks.

    Each stripe is quantized independently (as in the real checkpoint), so the
    scale has ``ceil(ROWS_PER_GROUP / BLOCK) * NUM_KV`` rows.
    """
    src_q, src_k, src_v = {}, {}, {}
    stripes_w, stripes_s = [], []
    for g in range(NUM_KV):
        rows = []
        for j in range(NUM_HEADS // NUM_KV):
            qh = g * (NUM_HEADS // NUM_KV) + j
            src_q[qh] = _head_block(qh, HEAD_DIM)
            rows.append(src_q[qh])
        src_k[g] = _head_block(1000 + g, HEAD_DIM)
        src_v[g] = _head_block(2000 + g, V_HEAD_DIM)
        rows.append(src_k[g])
        rows.append(src_v[g])
        stripe = torch.cat(rows, dim=0)
        w_g, s_g = _block_quant(stripe)
        stripes_w.append(w_g)
        stripes_s.append(s_g)
    w_full = torch.cat(stripes_w, dim=0)
    s_full = torch.cat(stripes_s, dim=0)
    return w_full, s_full, src_q, src_k, src_v


def _dequant(w_fp8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    se = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    se = se[: w_fp8.shape[0], : w_fp8.shape[1]]
    return w_fp8.to(torch.float32) * se


def _best_match(block: torch.Tensor, src: dict) -> tuple[int, float]:
    best_k, best_e = None, float("inf")
    for k, ref in src.items():
        e = (block - ref).norm().item() / ref.norm().item()
        if e < best_e:
            best_k, best_e = k, e
    return best_k, best_e


def _assert_heads(deq, off, rows, src, want_ids, kind, ctx):
    """Assert each head-block at ``off`` matches its expected source head."""
    tp_size, rank = ctx
    for want in want_ids:
        got, err = _best_match(deq[off : off + rows], src)
        assert got == want and err < 0.1, (
            f"tp{tp_size} r{rank} {kind} got {got} want {want} err {err:.3f}"
        )
        off += rows
    return off


def _expected_heads(tp_size: int, rank: int):
    """Return (q_head_ids, k_head_ids, v_head_ids) this rank should own, in order."""
    heads_per_group = NUM_HEADS // NUM_KV
    if tp_size <= NUM_KV:
        groups = range(rank * (NUM_KV // tp_size), (rank + 1) * (NUM_KV // tp_size))
        q = [g * heads_per_group + j for g in groups for j in range(heads_per_group)]
        k = list(groups)
        v = list(groups)
    else:
        replicas = tp_size // NUM_KV
        g = rank // replicas
        sub = rank % replicas
        q_per = heads_per_group // replicas
        q = [g * heads_per_group + sub * q_per + j for j in range(q_per)]
        k = [g]
        v = [g]
    return q, k, v


@pytest.mark.parametrize("tp_size", [2, 4, 8, 16])
def test_shard_fp8_qkv_proj_assigns_correct_heads(tp_size):
    w_full, s_full, src_q, src_k, src_v = _build_checkpoint()

    q_per_rank = NUM_HEADS // tp_size
    kv_per_rank = max(1, NUM_KV // tp_size)
    expected_rows = q_per_rank * HEAD_DIM + kv_per_rank * (HEAD_DIM + V_HEAD_DIM)

    for rank in range(tp_size):
        w_r, s_r = _shard_fp8_qkv_proj(
            w_full,
            s_full,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV,
            head_dim=HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            tp_rank=rank,
            tp_size=tp_size,
        )
        assert w_r.shape == (expected_rows, HIDDEN), (
            f"tp={tp_size} rank={rank}: weight rows {w_r.shape[0]} != {expected_rows}"
        )
        deq = _dequant(w_r, s_r)
        q_ids, k_ids, v_ids = _expected_heads(tp_size, rank)

        ctx = (tp_size, rank)
        off = _assert_heads(deq, 0, HEAD_DIM, src_q, q_ids, "Q", ctx)
        off = _assert_heads(deq, off, HEAD_DIM, src_k, k_ids, "K", ctx)
        off = _assert_heads(deq, off, V_HEAD_DIM, src_v, v_ids, "V", ctx)


def test_shard_fp8_qkv_proj_replicates_kv_across_ranks():
    """tp > num_kv_heads: ranks sharing a KV head get identical K/V but disjoint Q."""
    tp_size = 16
    w_full, s_full, *_ = _build_checkpoint()
    # ranks 0 and 1 both own KV head 0.
    outs = []
    for rank in (0, 1):
        w_r, s_r = _shard_fp8_qkv_proj(
            w_full,
            s_full,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV,
            head_dim=HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            tp_rank=rank,
            tp_size=tp_size,
        )
        outs.append(_dequant(w_r, s_r))
    q_rows = (NUM_HEADS // tp_size) * HEAD_DIM
    # K and V tail identical (replicated); Q heads disjoint (different slices).
    torch.testing.assert_close(outs[0][q_rows:], outs[1][q_rows:])
    assert not torch.allclose(outs[0][:q_rows], outs[1][:q_rows])
