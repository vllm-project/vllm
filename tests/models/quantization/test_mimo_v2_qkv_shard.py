# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for MiMo-V2 fused fp8 ``qkv_proj`` sharding.

The checkpoint stores the fused QKV pre-sharded at ``num_key_value_heads``
chunks (``ckpt_tp``), each holding that slice's ``[Q_c | K_c | V_c]`` rows with
its own block scale grid. ``ckpt_tp`` is not the layer's KV-head count: a
MiMo-V2.5 SWA layer has 8 KV heads but 4 chunks of 3712 rows (= 29 blocks), so
each chunk carries two KV heads. Reading the scales as one slice per KV head
expands 14 x 128 = 1792 rows for an 1856-row slice and fails.

These tests build such checkpoints for the GA, SWA and Pro geometries and check
that every rank gets exactly the weights it should own, for TP sizes below,
equal to and above ``ckpt_tp`` (the last case replicating KV heads).
"""

import pytest
import torch

from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.cpu_test

BLOCK = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
COLS = 128  # one column block is enough to exercise the row mapping
# fp8 e4m3 has 3 mantissa bits, so a single block-quantization round trip costs
# ~2-3% relative L2; a wrong row -> scale pairing costs >= 10%.
TOL = 0.06

# (num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp): MiMo-V2.5 GA, its
# SWA/MTP layers, and MiMo-V2.5-Pro.
GEOMETRIES = {
    "ga": (64, 4, 192, 128, 4),
    "swa": (64, 8, 192, 128, 4),
    "pro": (128, 8, 192, 128, 8),
}


def _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    return (
        num_heads // ckpt_tp * head_dim
        + num_kv_heads // ckpt_tp * head_dim
        + num_kv_heads // ckpt_tp * v_head_dim
    )


def _quantize_chunks(truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    """Block-quantize each checkpoint chunk on its own grid, like the export."""
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    weights, scales = [], []
    for chunk in range(ckpt_tp):
        rows = truth[chunk * rows_per_chunk : (chunk + 1) * rows_per_chunk]
        quantized = torch.zeros_like(rows)
        scale = torch.zeros(
            cdiv(rows_per_chunk, BLOCK), rows.shape[1] // BLOCK, dtype=torch.float32
        )
        for row in range(0, rows_per_chunk, BLOCK):
            for col in range(0, rows.shape[1], BLOCK):
                block = rows[row : row + BLOCK, col : col + BLOCK]
                amax = block.abs().max().clamp(min=1e-12) / FP8_MAX
                scale[row // BLOCK, col // BLOCK] = amax
                quantized[row : row + BLOCK, col : col + BLOCK] = block / amax
        weights.append(quantized.to(FP8_DTYPE))
        scales.append(scale)
    return torch.cat(weights, dim=0), torch.cat(scales, dim=0)


def _owned_rows(
    truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
):
    """Rows the rank must own, in the forward's `[Q | K | V]` order."""
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    q_per_chunk = num_heads // ckpt_tp * head_dim
    k_per_chunk = num_kv_heads // ckpt_tp * head_dim
    kv_heads: list[int] = []
    if tp_size <= num_kv_heads:
        kv_heads = list(
            range(
                tp_rank * (num_kv_heads // tp_size),
                (tp_rank + 1) * (num_kv_heads // tp_size),
            )
        )
    else:
        kv_heads = [tp_rank // (tp_size // num_kv_heads)]
    index = []
    for head in range(
        tp_rank * (num_heads // tp_size), (tp_rank + 1) * (num_heads // tp_size)
    ):
        chunk, offset = divmod(head, num_heads // ckpt_tp)
        start = chunk * rows_per_chunk + offset * head_dim
        index += list(range(start, start + head_dim))
    for head in kv_heads:
        chunk, offset = divmod(head, num_kv_heads // ckpt_tp)
        start = chunk * rows_per_chunk + q_per_chunk + offset * head_dim
        index += list(range(start, start + head_dim))
    for head in kv_heads:
        chunk, offset = divmod(head, num_kv_heads // ckpt_tp)
        start = chunk * rows_per_chunk + q_per_chunk + k_per_chunk + offset * v_head_dim
        index += list(range(start, start + v_head_dim))
    return truth[torch.tensor(index)]


def _dequantize(weight, scale):
    """What the kernel computes: scale[row // block] applied per weight row."""
    per_row = scale.repeat_interleave(BLOCK, dim=1)
    return weight.to(torch.float32) * per_row[torch.arange(weight.shape[0]) // BLOCK]


def _relative_l2(got, expected):
    return ((got - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_fused_qkv_proj_sharding(geometry, tp_size):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    if num_heads % tp_size or (tp_size > num_kv_heads and tp_size % num_kv_heads):
        pytest.skip("TP size does not divide the head counts")

    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    for chunk in range(ckpt_tp):  # per-chunk magnitude spread, as in the export
        truth[chunk * rows_per_chunk : (chunk + 1) * rows_per_chunk] *= 1.0 + chunk
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )

    kv_per_rank = (num_kv_heads // tp_size) if tp_size <= num_kv_heads else 1
    rows_rank = (num_heads // tp_size) * head_dim + kv_per_rank * (
        head_dim + v_head_dim
    )
    for tp_rank in range(tp_size):
        w_rank, s_rank = _shard_fp8_qkv_proj(
            weight,
            scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ckpt_tp=ckpt_tp,
        )
        assert w_rank.shape == (rows_rank, COLS)
        assert s_rank.shape == (cdiv(rows_rank, BLOCK), scale.shape[1])
        expected = _owned_rows(
            truth,
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            ckpt_tp,
            tp_rank,
            tp_size,
        )
        assert _relative_l2(_dequantize(w_rank, s_rank), expected) <= TOL


@pytest.mark.parametrize("geometry", ["ga", "swa"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_exact_chunk_matches_checkpoint(geometry, tp_size):
    """At tp_size <= ckpt_tp the chunk count is the only thing that matters."""
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    if tp_size == ckpt_tp:
        w_rank, s_rank = _shard_fp8_qkv_proj(
            weight,
            scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            tp_rank=0,
            tp_size=tp_size,
            ckpt_tp=ckpt_tp,
        )
        # The checkpoint chunk *is* the rank's slice: no re-quantization.
        # (torch.equal has no fp8 kernel, so compare the underlying bytes.)
        expected_w = weight.chunk(tp_size, dim=0)[0]
        assert torch.equal(w_rank.view(torch.uint8), expected_w.view(torch.uint8))
        assert torch.equal(s_rank, scale.chunk(tp_size, dim=0)[0])


def test_wrong_chunk_count_is_detected():
    """Reading SWA scales as one slice per KV head must not look correct.

    This is the regression the fix addresses: 116 scale rows are 4 x 29 (29 =
    ceil(3712 / 128)), not 8 x 15, so ``ckpt_tp = num_kv_heads`` pairs every
    weight row with the wrong scale.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES["swa"]
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    for chunk in range(ckpt_tp):
        truth[chunk * rows_per_chunk : (chunk + 1) * rows_per_chunk] *= 1.0 + chunk
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    assert scale.shape[0] == ckpt_tp * cdiv(rows_per_chunk, BLOCK)
    assert scale.shape[0] != num_kv_heads * cdiv(rows_per_chunk, BLOCK)

    kw = dict(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        tp_rank=0,
        tp_size=1,
    )
    correct = _shard_fp8_qkv_proj(weight, scale, ckpt_tp=ckpt_tp, **kw)
    wrong = _shard_fp8_qkv_proj(weight, scale, ckpt_tp=num_kv_heads, **kw)
    assert correct[0].shape == wrong[0].shape

    expected = _owned_rows(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, 0, 1
    )
    assert _relative_l2(_dequantize(*correct), expected) <= TOL
    assert _relative_l2(_dequantize(*wrong), expected) > 5 * TOL
