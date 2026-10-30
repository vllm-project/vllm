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

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.mimo_v2 import (
    _fused_qkv_kv_chunk_rows,
    _shard_fp8_qkv_proj,
    _split_qkv,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_num_threads

pytestmark = pytest.mark.cpu_test

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.bfloat16
)

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


@pytest.fixture(scope="module", autouse=True)
def single_threaded_cpu_ops():
    # Do not prime the parent's thread pool before later engine forks.
    with set_default_torch_num_threads(1):
        yield


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


def _owned_index(
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
):
    """Checkpoint row indices of the rank's slice, in the forward's `[Q | K | V]`."""
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
    return torch.tensor(index)


def _owned_rows(
    truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
):
    """Rows the rank must own, in the forward's `[Q | K | V]` order."""
    index = _owned_index(
        num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
    )
    return truth[index]


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


@pytest.mark.parametrize("tp_size", [1, 2])
def test_exact_gather_permutes_checkpoint_rows(tp_size):
    """The gather touches not a single bit.

    The SWA geometry divides every chunk segment into whole scale blocks, so
    the rank's slice must come back as a pure row permutation of the
    checkpoint weights and of its per-chunk scale grid.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES["swa"]
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    chunk_scale_rows = cdiv(rows_per_chunk, BLOCK)
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
        index = _owned_index(
            num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
        )
        assert torch.equal(w_rank.view(torch.uint8), weight[index].view(torch.uint8))
        scale_index = (
            (index // rows_per_chunk) * chunk_scale_rows
            + (index % rows_per_chunk) // BLOCK
        ).view(-1, BLOCK)
        assert (scale_index == scale_index[:, :1]).all()
        assert torch.equal(s_rank, scale[scale_index[:, 0]])


@pytest.mark.parametrize("geometry", ["ga", "pro"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_exact_padded_layout_matches_checkpoint(geometry, tp_size):
    """A straddling K segment gathers padded.

    The GA and Pro geometries give each chunk 192 K + 128 V rows, one and a
    half scale blocks, so the rank keeps each chunk's K and V rows together
    zero-padded to whole blocks: every weight byte and every scale row must
    come back exactly as the checkpoint wrote it, and the pad rows zero.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    if tp_size >= ckpt_tp or ckpt_tp % tp_size:
        pytest.skip("padded layout needs tp_size a proper divisor below ckpt_tp")
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    q_per_chunk = num_heads // ckpt_tp * head_dim
    k_per_chunk = num_kv_heads // ckpt_tp * head_dim
    v_per_chunk = num_kv_heads // ckpt_tp * v_head_dim
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    kv_chunk_rows = _fused_qkv_kv_chunk_rows(
        num_heads, num_kv_heads, head_dim, v_head_dim, tp_size, ckpt_tp
    )
    assert kv_chunk_rows == cdiv(k_per_chunk + v_per_chunk, BLOCK) * BLOCK
    chunks_per_rank = ckpt_tp // tp_size
    q_size = num_heads // tp_size * head_dim
    chunk_scale_rows = cdiv(rows_per_chunk, BLOCK)
    pad = kv_chunk_rows - (k_per_chunk + v_per_chunk)
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
            kv_chunk_rows=kv_chunk_rows,
        )
        index = _owned_index(
            num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
        )
        q_idx, kv_idx = index[:q_size], index[q_size:]
        k_idx, v_idx = kv_idx.split(
            [chunks_per_rank * k_per_chunk, chunks_per_rank * v_per_chunk]
        )
        pieces = [weight[q_idx].view(torch.uint8)]
        for c in range(chunks_per_rank):
            k_rows = k_idx[c * k_per_chunk : (c + 1) * k_per_chunk]
            v_rows = v_idx[c * v_per_chunk : (c + 1) * v_per_chunk]
            pieces += [
                weight[k_rows].view(torch.uint8),
                weight[v_rows].view(torch.uint8),
                torch.zeros(pad, COLS, dtype=torch.uint8),
            ]
        assert torch.equal(w_rank.view(torch.uint8), torch.cat(pieces))

        first_chunk = tp_rank * chunks_per_rank
        q_scales, kv_scales = [], []
        for c in range(first_chunk, first_chunk + chunks_per_rank):
            base = c * chunk_scale_rows
            q_end = base + q_per_chunk // BLOCK
            q_scales.append(scale[base:q_end])
            kv_scales.append(scale[q_end : (c + 1) * chunk_scale_rows])
        assert torch.equal(s_rank, torch.cat(q_scales + kv_scales))


@pytest.mark.parametrize("geometry", ["ga", "pro"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_split_qkv_round_trips_the_padded_layout(geometry, tp_size):
    """The forward split inverts the padded gather, byte for byte.

    The writer and the reader each encode the ``[Q | K V pad]`` chunk order,
    so a one-sided change would keep every shape intact while attention reads
    swapped K and V rows.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    if tp_size >= ckpt_tp or ckpt_tp % tp_size:
        pytest.skip("padded layout needs tp_size a proper divisor below ckpt_tp")
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    kv_chunk_rows = _fused_qkv_kv_chunk_rows(
        num_heads, num_kv_heads, head_dim, v_head_dim, tp_size, ckpt_tp
    )
    assert kv_chunk_rows
    q_size = num_heads // tp_size * head_dim
    k_size = num_kv_heads // tp_size * head_dim
    v_size = num_kv_heads // tp_size * v_head_dim
    for tp_rank in range(tp_size):
        w_rank, _ = _shard_fp8_qkv_proj(
            weight,
            scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ckpt_tp=ckpt_tp,
            kv_chunk_rows=kv_chunk_rows,
        )
        q, k, v = _split_qkv(w_rank.T, q_size, k_size, v_size, kv_chunk_rows)
        index = _owned_index(
            num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
        )
        parts = (
            index[:q_size],
            index[q_size : q_size + k_size],
            index[q_size + k_size :],
        )
        expected = [weight[part].T.contiguous().view(torch.uint8) for part in parts]
        for got, want in zip((q, k, v), expected, strict=True):
            assert torch.equal(got.contiguous().view(torch.uint8), want)


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


@pytest.mark.parametrize("tp_size", [4, 8])
def test_requantized_rows_match_the_checkpoint_rows(tp_size):
    """Rows that do not form whole checkpoint blocks are requantized from the
    checkpoint rows and their own scales, as before."""
    from vllm.model_executor.models.mimo_v2 import _requantize_fp8

    num_heads, num_kv_heads, head_dim, v_head_dim, _ = GEOMETRIES["ga"]
    ckpt_tp = 2  # tp_size above ckpt_tp: KV heads replicated
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS)
    weight, scale = _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    chunk_scale_rows = cdiv(rows_per_chunk, BLOCK)
    for tp_rank in range(tp_size):
        index = _owned_index(
            num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, tp_rank, tp_size
        )
        scale_index = (index // rows_per_chunk) * chunk_scale_rows + (
            index % rows_per_chunk
        ) // BLOCK
        grouped = weight[index].float() * scale[scale_index].repeat_interleave(
            BLOCK, dim=1
        )
        want_w, want_s = _requantize_fp8(grouped, index.numel(), BLOCK, FP8_DTYPE)
        got_w, got_s = _shard_fp8_qkv_proj(
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
        assert torch.equal(got_w.view(torch.uint8), want_w.view(torch.uint8))
        assert torch.equal(got_s, want_s)


class _RowScalesMethod(QuantizeMethodBase):
    """The row-scale step of `_Fp8RowScalesLinearMethod`, without the fp8
    kernel's own processing."""

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.models.mimo_v2 import _process_row_scales

        _process_row_scales(layer)


def _fused_qkv_model(monkeypatch, geometry, tp_rank, tp_size):
    """A MiMoV2Model with one attention layer whose fused fp8 qkv_proj loads
    as rows and row scales, and its checkpoint (weight, scale)."""
    from vllm.model_executor.models import mimo_v2

    monkeypatch.setattr(mimo_v2, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        mimo_v2, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    kv_chunk_rows = _fused_qkv_kv_chunk_rows(
        num_heads, num_kv_heads, head_dim, v_head_dim, tp_size, ckpt_tp
    )
    kv_per_rank = max(1, num_kv_heads // tp_size)
    rows = num_heads // tp_size * head_dim
    if kv_chunk_rows:
        rows += ckpt_tp // tp_size * kv_chunk_rows
    else:
        rows += kv_per_rank * (head_dim + v_head_dim)

    attn = torch.nn.Module()
    attn.total_num_heads, attn.total_num_kv_heads = num_heads, num_kv_heads
    attn.head_dim, attn.v_head_dim = head_dim, v_head_dim
    attn.kv_chunk_rows = kv_chunk_rows
    attn.qkv_proj = qkv = torch.nn.Module()
    qkv.weight = torch.nn.Parameter(
        torch.empty(rows, COLS, dtype=FP8_DTYPE), requires_grad=False
    )
    qkv.weight_scale_inv = torch.nn.Parameter(  # one scale per row
        torch.empty(rows, COLS // BLOCK), requires_grad=False
    )
    qkv.weight_block_size = [BLOCK, BLOCK]
    qkv.quant_method = _RowScalesMethod()
    layer = torch.nn.Module()
    layer.self_attn = attn

    model = mimo_v2.MiMoV2Model.__new__(mimo_v2.MiMoV2Model)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList([layer])
    model.config = SimpleNamespace(num_key_value_heads=ckpt_tp)
    monkeypatch.setattr(model, "get_expert_mapping", lambda: [])
    return model


def _fused_qkv_checkpoint(geometry, seed):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GEOMETRIES[geometry]
    rows_per_chunk = _chunk_rows(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp)
    gen = torch.Generator().manual_seed(seed)
    truth = torch.randn(ckpt_tp * rows_per_chunk, COLS, generator=gen)
    return _quantize_chunks(
        truth, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )


def _expected_shard(model, weight, scale, tp_rank, tp_size):
    attn = model.layers[0].self_attn
    return _shard_fp8_qkv_proj(
        weight,
        scale,
        num_heads=attn.total_num_heads,
        num_kv_heads=attn.total_num_kv_heads,
        head_dim=attn.head_dim,
        v_head_dim=attn.v_head_dim,
        tp_rank=tp_rank,
        tp_size=tp_size,
        ckpt_tp=model.config.num_key_value_heads,
        kv_chunk_rows=attn.kv_chunk_rows,
    )


def _assert_shard(model, expected):
    qkv = model.layers[0].self_attn.qkv_proj
    assert torch.equal(qkv.weight.view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(qkv.weight_scale_inv, expected[1])


# ga at 4: one chunk per rank; ga at 2: padded block permutation; ga at 8 and
# swa at 8: requantized
@pytest.mark.parametrize(
    "geometry,tp_size", [("ga", 4), ("ga", 2), ("ga", 8), ("swa", 8)]
)
@pytest.mark.parametrize("scale_first", [False, True])
def test_load_weights_split_fp8_qkv_proj_and_skips_absent_layers(
    monkeypatch, geometry, tp_size, scale_first
):
    """Through ``MiMoV2Model.load_weights``: a layer's fused fp8 ``qkv_proj``
    weight and scale arriving in separate calls, in either order, load on their
    own; post-load processing gives the rank's shard. The pair for a layer the
    instance does not have (checkpoint deeper than ``num_hidden_layers``) is
    skipped.
    """
    from vllm.model_executor.models import mimo_v2

    tp_rank = tp_size - 1
    model = _fused_qkv_model(monkeypatch, geometry, tp_rank, tp_size)
    weight, scale = _fused_qkv_checkpoint(geometry, 0)
    present, absent = "layers.0.self_attn.qkv_proj", "layers.7.self_attn.qkv_proj"
    calls = [
        [(f"{present}.weight", weight), (f"{absent}.weight", weight)],
        [(f"{absent}.weight_scale_inv", scale), (f"{present}.weight_scale_inv", scale)],
    ]
    loaded = set()
    for call in reversed(calls) if scale_first else calls:
        loaded |= mimo_v2.MiMoV2Model.load_weights(model, call)
    assert loaded == {f"{present}.weight", f"{present}.weight_scale_inv"}

    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))
    _assert_shard(model, _expected_shard(model, weight, scale, tp_rank, tp_size))


@pytest.mark.parametrize("geometry,tp_size", [("ga", 2), ("ga", 8)])
@pytest.mark.parametrize("left_out", [False, True])
def test_direct_reload_equals_cold_start(monkeypatch, geometry, tp_size, left_out):
    """A direct reload views the scale back to its row layout; the new rows
    load in separate calls and processing gives what a cold start gives. A
    qkv_proj left out of the update keeps its weights."""
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        finish_direct_reload,
    )
    from vllm.model_executor.models import mimo_v2

    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    tp_rank = tp_size - 1
    model = _fused_qkv_model(monkeypatch, geometry, tp_rank, tp_size)
    prefix = "layers.0.self_attn.qkv_proj"
    old = _fused_qkv_checkpoint(geometry, 1)
    mimo_v2.MiMoV2Model.load_weights(model, [(f"{prefix}.weight", old[0])])
    mimo_v2.MiMoV2Model.load_weights(model, [(f"{prefix}.weight_scale_inv", old[1])])
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    new = old if left_out else _fused_qkv_checkpoint(geometry, 2)
    begin_direct_reload(model)
    if not left_out:
        for kind, tensor in (("weight_scale_inv", new[1]), ("weight", new[0])):
            mimo_v2.MiMoV2Model.load_weights(model, [(f"{prefix}.{kind}", tensor)])
    finish_direct_reload(model, _MODEL_CONFIG, torch.device("cpu"))

    _assert_shard(model, _expected_shard(model, new[0], new[1], tp_rank, tp_size))
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs
