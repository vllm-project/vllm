# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the ROCm AITER fused DSA indexer QK path.

Compares ``torch.ops.vllm.rocm_aiter_indexer_qk_rope_quant_and_cache`` against
the unfused flow it replaces in ``Indexer.forward``
(vllm/model_executor/models/deepseek_v2.py), built from the same primitives:

    k = layer_norm(k, k_norm.weight, k_norm.bias, eps)
    ops.rotary_embedding(positions, q[..., :64], k[..., :64], 64, cache, neox)
    q_fp8, q_scale = per_token_group_quant_fp8(q, 128, use_ue8m0=True)
    weights_out = weights * q_scale * softmax_scale * n_head_scale
    indexer_k_quant_and_cache_triton(k, kv_cache, slot_mapping, 128, "ue8m0")

The fused kernel folds the q scale into ``weights_out``, so the comparison is
on the products the downstream MQA-logits kernels consume
(``q_fp8 * weights_out``) and on the dequantized indexer K cache.

The two paths cannot be bit-equal, and the difference is in the q scale, not in
the RoPE: the kernel rounds the roped q through bf16 exactly as the unfused flow
does, but derives a plain fp32 scale from it, while per_token_group_quant_fp8
rounds that scale to a power of two (use_ue8m0=True). A different scale moves
every element of the token by at most one fp8 code, in either direction, so
asserting closeness to the unfused path would penalise whichever path happens to
sit further from the truth. Both are instead scored against an fp64 golden of the
same math, and the fused path must be no less accurate than the unfused one, with
every element within one fp8 code of it.

Speed is not asserted here: see benchmarks/kernels/benchmark_indexer_qk_fusion.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform

_SKIP_NON_MI3XX = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_mi3xx

    _SKIP_NON_MI3XX = not on_mi3xx()

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(_SKIP_NON_MI3XX, reason="MI300/MI350 only (aiter CK kernel)"),
]

HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 32
MAX_POS = 65536
QUANT_BLOCK = 128
EPS = 1e-6
SCALE_FMT = "ue8m0"
WEIGHTS_SCALE = HEAD_DIM**-0.5 * N_HEAD**-0.5
PREFIX = "model.layers.0.self_attn.indexer.k_cache"

# fp8-e4m3 keeps 3 mantissa bits, so one code step is at most 12.5% of an
# element: a value sitting on a quantization boundary flips to the neighbouring
# code under any reordering of the RoPE/LayerNorm arithmetic.
ONE_CODE_REL = 0.13
# Slack on "no less accurate than unfused": both are fp8, so their errors
# against the golden are dominated by the same quantization step.
ACCURACY_SLACK = 1.10
MAX_FP8_REL_L2 = 0.05


def _require_aiter() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required for the fused indexer QK kernel")


def _indexer_metadata(slot_mapping: torch.Tensor):
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    num_tokens = int(slot_mapping.shape[0])
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([num_tokens], device=slot_mapping.device),
        max_seq_len=num_tokens,
        slot_mapping=slot_mapping,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=num_tokens,
    )


def _inputs(num_tokens: int, block_size: int, capacity: int | None = None):
    capacity = capacity or num_tokens
    num_blocks = (capacity + block_size - 1) // block_size + 2
    dev, dt = "cuda", torch.bfloat16
    return SimpleNamespace(
        q=torch.randn(capacity, N_HEAD, HEAD_DIM, device=dev, dtype=dt),
        kw=torch.randn(capacity, HEAD_DIM + N_HEAD, device=dev, dtype=dt),
        positions=torch.randint(0, MAX_POS, (capacity,), device=dev, dtype=torch.int64),
        slots=torch.randperm(num_blocks * block_size, device=dev, dtype=torch.int64)[
            :num_tokens
        ],
        norm_weight=torch.randn(HEAD_DIM, device=dev, dtype=dt),
        norm_bias=torch.randn(HEAD_DIM, device=dev, dtype=dt),
        cos_sin_cache=torch.randn(MAX_POS, ROPE_DIM, device=dev, dtype=dt),
        kv_cache=torch.zeros(
            num_blocks,
            block_size,
            HEAD_DIM + 4,
            dtype=current_platform.fp8_dtype(),
            device=dev,
        ),
    )


def _reference(t, kv_cache: torch.Tensor, is_neox: bool):
    """The five launches the fused kernel replaces, in the same order."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    num_tokens = t.q.shape[0]
    k = torch.nn.functional.layer_norm(
        t.kw[:, :HEAD_DIM].float(),
        (HEAD_DIM,),
        t.norm_weight.float(),
        t.norm_bias.float(),
        EPS,
    ).to(t.q.dtype)
    q = t.q.clone()
    ops.rotary_embedding(
        t.positions,
        q[..., :ROPE_DIM],
        k[..., :ROPE_DIM].unsqueeze(1),
        ROPE_DIM,
        t.cos_sin_cache,
        is_neox,
    )
    q_fp8, q_scale = per_token_group_quant_fp8(
        q.view(-1, HEAD_DIM), QUANT_BLOCK, column_major_scales=False, use_ue8m0=True
    )
    weights = (
        t.kw[:, HEAD_DIM:].float() * q_scale.view(num_tokens, N_HEAD) * WEIGHTS_SCALE
    )
    indexer_k_quant_and_cache_triton(k, kv_cache, t.slots, QUANT_BLOCK, SCALE_FMT)
    return q_fp8.view(num_tokens, N_HEAD, HEAD_DIM), weights


def _fused(t, kv_cache, is_neox, q_out=None, w_out=None, zero_outputs=False):
    half = ROPE_DIM // 2
    if q_out is None:
        q_out = torch.zeros(
            t.q.shape, dtype=current_platform.fp8_dtype(), device=t.q.device
        )
    if w_out is None:
        w_out = torch.zeros(t.q.shape[:2], dtype=torch.float32, device=t.q.device)
    torch.ops.vllm.rocm_aiter_indexer_qk_rope_quant_and_cache(
        PREFIX,
        kv_cache,
        t.q,
        t.kw[:, :HEAD_DIM],
        t.kw[:, HEAD_DIM:],
        t.positions,
        t.cos_sin_cache[:, :half],
        t.cos_sin_cache[:, half:],
        t.norm_weight,
        t.norm_bias,
        q_out,
        w_out,
        EPS,
        QUANT_BLOCK,
        SCALE_FMT,
        WEIGHTS_SCALE,
        zero_outputs,
        is_neox,
    )
    return q_out, w_out


def _patch_context(monkeypatch, slot_mapping: torch.Tensor) -> None:
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as mla_sparse

    monkeypatch.setattr(
        mla_sparse,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={PREFIX: _indexer_metadata(slot_mapping)}
        ),
    )


def _golden(t, is_neox: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 LayerNorm + RoPE: the same math with no intermediate rounding."""
    half = ROPE_DIM // 2
    k = torch.nn.functional.layer_norm(
        t.kw[:, :HEAD_DIM].double(),
        (HEAD_DIM,),
        t.norm_weight.double(),
        t.norm_bias.double(),
        EPS,
    )
    cos = t.cos_sin_cache[t.positions, :half].double()
    sin = t.cos_sin_cache[t.positions, half:].double()

    def rope(x: torch.Tensor) -> torch.Tensor:
        c, s_ = cos, sin
        while c.dim() < x.dim():
            c, s_ = c.unsqueeze(-2), s_.unsqueeze(-2)
        pe, rest = x[..., :ROPE_DIM], x[..., ROPE_DIM:]
        if is_neox:
            x1, x2 = pe[..., :half], pe[..., half:]
            roped = torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1)
        else:
            x1, x2 = pe[..., 0::2], pe[..., 1::2]
            roped = torch.stack([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1).flatten(
                -2
            )
        return torch.cat([roped, rest], dim=-1)

    return rope(t.q.double()), rope(k)


def _rel_l2(got: torch.Tensor, golden: torch.Tensor) -> float:
    return ((got.double() - golden).norm() / golden.norm().clamp(min=1e-12)).item()


def _assert_no_less_accurate(
    fused: torch.Tensor, unfused: torch.Tensor, golden: torch.Tensor, what: str
) -> None:
    err_fused, err_unfused = _rel_l2(fused, golden), _rel_l2(unfused, golden)
    assert err_fused <= MAX_FP8_REL_L2, (
        f"{what}: fused error vs fp64 golden {err_fused:.3e} exceeds fp8 granularity"
    )
    assert err_fused <= err_unfused * ACCURACY_SLACK, (
        f"{what}: fused is less accurate than unfused "
        f"({err_fused:.3e} vs {err_unfused:.3e})"
    )
    # And the two paths must still agree element-wise to within one fp8 code.
    scale = torch.maximum(fused.abs(), unfused.abs())
    live = scale > 1e-3 * unfused.abs().max()
    rel = (fused - unfused).abs()[live] / scale[live]
    worst = rel.max().item()
    assert worst <= ONE_CODE_REL, (
        f"{what}: {(rel > ONE_CODE_REL).sum().item()} of {rel.numel()} elements "
        f"differ by more than one fp8 code (worst rel {worst:.4f})"
    )


def _dequant_cache(kv_cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Dequantize the rows `slots` point at, for either in-block layout."""
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    flat = kv_cache.view(num_blocks, -1)
    values = flat[:, : block_size * HEAD_DIM]
    scales = flat[:, block_size * HEAD_DIM :].contiguous().view(torch.float32)
    tile = 16
    j = torch.arange(HEAD_DIM, device=kv_cache.device)
    out = torch.empty(
        (slots.shape[0], HEAD_DIM), dtype=torch.float32, device=kv_cache.device
    )
    for i, slot in enumerate(slots.tolist()):
        block_id, off = slot // block_size, slot % block_size
        if block_size == 1:
            idx = off * HEAD_DIM + j
        else:
            # 16x16-tiled in-block layout, mirroring the writer's SHUFFLE path.
            idx = (
                (off // tile) * tile * HEAD_DIM
                + (off % tile) * tile
                + (j // tile) * tile * tile
                + j % tile
            )
        out[i] = (
            values[block_id, idx].view(kv_cache.dtype).float() * scales[block_id, off]
        )
    return out


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 257, 1023])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize("is_neox", [True, False])
@torch.inference_mode()
def test_fused_matches_unfused(monkeypatch, num_tokens, block_size, is_neox):
    """One fused launch reproduces the five unfused launches it replaces."""
    _require_aiter()
    torch.manual_seed(0)
    t = _inputs(num_tokens, block_size)
    kv_ref = torch.zeros_like(t.kv_cache)

    q_fp8_ref, weights_ref = _reference(t, kv_ref, is_neox)
    _patch_context(monkeypatch, t.slots)
    q_fp8, weights_out = _fused(t, t.kv_cache, is_neox)
    q_golden, k_golden = _golden(t, is_neox)

    # Scale-fold invariant: the kernel may split the scale between q_fp8 and
    # weights_out differently, only the product reaches the logits kernels.
    weights_raw = t.kw[:, HEAD_DIM:].double() * WEIGHTS_SCALE
    _assert_no_less_accurate(
        q_fp8.float() * weights_out.unsqueeze(-1),
        q_fp8_ref.float() * weights_ref.unsqueeze(-1),
        q_golden * weights_raw.unsqueeze(-1),
        "q_fp8 * weights_out",
    )
    _assert_no_less_accurate(
        _dequant_cache(t.kv_cache, t.slots),
        _dequant_cache(kv_ref, t.slots),
        k_golden,
        "indexer K cache",
    )


@torch.inference_mode()
def test_unowned_and_padded_rows_stay_zero(monkeypatch):
    """Rows the kernel skips must read as zero, not as stale data.

    The kernel early-returns on ``slot_mapping < 0`` - PAD_SLOT_ID marks both
    CUDA-graph padding and, under context parallel, tokens this rank does not
    own - and never touches rows past ``slot_mapping``. Decode reads
    ``weights[:batch_size * next_n]``, which covers those rows, so they must be
    zero for the padded logits they feed to stay finite.
    """
    _require_aiter()
    torch.manual_seed(0)
    num_tokens, capacity = 64, 96
    t = _inputs(num_tokens, block_size=64, capacity=capacity)
    t.slots[::4] = -1
    before = t.kv_cache.view(torch.uint8).clone()

    _patch_context(monkeypatch, t.slots)
    q_fp8, weights_out = _fused(t, t.kv_cache, is_neox=True)

    skipped = t.slots < 0
    assert torch.all(q_fp8[:num_tokens][skipped].view(torch.uint8) == 0)
    assert torch.all(weights_out[:num_tokens][skipped] == 0)
    assert torch.all(q_fp8[num_tokens:].view(torch.uint8) == 0)
    assert torch.all(weights_out[num_tokens:] == 0)
    # Not vacuous: the owned rows were written, in both outputs and the cache.
    assert not torch.all(q_fp8[:num_tokens][~skipped].view(torch.uint8) == 0)
    assert not torch.equal(t.kv_cache.view(torch.uint8), before)


@torch.inference_mode()
def test_op_schema(monkeypatch):
    """opcheck the custom op: fake impl for tracing, declared mutates_args."""
    _require_aiter()
    from tests.kernels.utils import opcheck

    torch.manual_seed(0)
    t = _inputs(num_tokens=8, block_size=64)
    _patch_context(monkeypatch, t.slots)
    half = ROPE_DIM // 2
    opcheck(
        torch.ops.vllm.rocm_aiter_indexer_qk_rope_quant_and_cache,
        (
            PREFIX,
            t.kv_cache,
            t.q,
            t.kw[:, :HEAD_DIM],
            t.kw[:, HEAD_DIM:],
            t.positions,
            t.cos_sin_cache[:, :half],
            t.cos_sin_cache[:, half:],
            t.norm_weight,
            t.norm_bias,
            torch.zeros(
                t.q.shape, dtype=current_platform.fp8_dtype(), device=t.q.device
            ),
            torch.zeros(t.q.shape[:2], dtype=torch.float32, device=t.q.device),
            EPS,
            QUANT_BLOCK,
            SCALE_FMT,
            WEIGHTS_SCALE,
            False,
            True,
        ),
    )
