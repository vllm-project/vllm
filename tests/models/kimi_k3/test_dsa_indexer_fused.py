# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 fused DSA indexer prologues against the unfused reference path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAIndexer
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
    dsa_fused_prologue,
    indexer_k_norm_quant_insert,
    indexer_q_quant_fold_weights,
)
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
    mxfp4_fused_prologue,
    mxfp4_k_norm_quant_insert,
    mxfp4_q_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton indexer prologues need CUDA"
)

HEAD_DIM = 128
NUM_HEADS = 32
ENTRY = HEAD_DIM + 4  # fp8 values + one fp32 scale per token
MXFP4_ENTRY = HEAD_DIM // 2 + HEAD_DIM // 32  # e2m1 values + one ue8m0 per 32
BLOCK = 64  # K3.1 kernel block size
MLA_BYTES = 656  # fp8_ds_mla bytes per token ahead of the indexer page
ROW_STRIDE = 50512  # packed block row: 64 x (656 + 132), 656-aligned
TOPK = 2048


def _make_cache(
    packed: bool, num_blocks: int, entry: int = ENTRY
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (backing bytes, indexer cache view [num_blocks, BLOCK, entry])."""
    if not packed:
        raw = torch.zeros(num_blocks, BLOCK, entry, dtype=torch.uint8, device="cuda")
        return raw, raw
    # One DSA layer's indexer page inside a packed block row.
    raw = torch.zeros(num_blocks, ROW_STRIDE, dtype=torch.uint8, device="cuda")
    view = torch.as_strided(
        raw,
        size=(num_blocks, BLOCK, entry),
        stride=(ROW_STRIDE, entry, 1),
        storage_offset=BLOCK * MLA_BYTES,
    )
    return raw, view


def _dequant(view: torch.Tensor) -> torch.Tensor:
    """Dequantize every token of an indexer cache view to [blocks*BLOCK, D]."""
    num_blocks = view.shape[0]
    rows = []
    for b in range(num_blocks):
        flat = view[b].reshape(-1)  # row-major within the block
        data = flat[: BLOCK * HEAD_DIM].view(current_platform.fp8_dtype())
        data = data.view(BLOCK, HEAD_DIM).float()
        scale_bytes = flat[BLOCK * HEAD_DIM : BLOCK * ENTRY].contiguous()
        if mqa_logits_use_mx_scales():
            # Four equal UE8M0 bytes: the exponent of the power-of-two scale.
            exp = scale_bytes.view(BLOCK, 4)[:, 0].to(torch.int32)
            scale = torch.pow(2.0, (exp - 127).float())
        else:
            scale = scale_bytes.view(torch.float32)
        rows.append(data * scale[:, None])
    return torch.cat(rows)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_k_norm_quant_insert_matches_unfused(packed: bool, num_tokens: int):
    torch.manual_seed(0)
    num_blocks = (num_tokens + 2 * BLOCK - 1) // BLOCK + 1
    # The merged wkv GEMM output is [k | weights]; k is a strided view of it.
    kw = torch.randn(
        num_tokens, HEAD_DIM + NUM_HEADS, dtype=torch.bfloat16, device="cuda"
    )
    k = kw[:, :HEAD_DIM]
    weight = torch.rand(HEAD_DIM, dtype=torch.float32, device="cuda") + 0.5
    bias = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1
    eps = 1e-6
    # Fewer slots than rows (cudagraph padding), plus a padded (-1) slot.
    num_slots = max(num_tokens - 1, 1)
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_slots]
    if num_slots > 2:
        slots[1] = -1
    slots = slots.to(torch.int64)

    ref_raw, ref_view = _make_cache(packed, num_blocks)
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), weight, bias, eps).to(k.dtype)
    ops.indexer_k_quant_and_cache(
        ref_k[:num_slots].contiguous(),
        ref_view,
        slots,
        HEAD_DIM,
        "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0",
    )

    out_raw, out_view = _make_cache(packed, num_blocks)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    indexer_k_norm_quant_insert(k, weight, bias, eps, slots, out_view, topk, num_tokens)

    # Only LayerNorm reduction order differs, so bytes match except for rare
    # bf16 rounding ties; dequantized values agree to fp8 precision.
    mismatch = (ref_raw != out_raw).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of cache bytes differ"
    torch.testing.assert_close(
        _dequant(out_view), _dequant(ref_view), atol=0.07, rtol=0.07
    )
    if packed:
        # The MLA section of each unit is untouched.
        mla = out_raw[:, : BLOCK * MLA_BYTES]
        assert torch.count_nonzero(mla) == 0
    assert torch.all(topk[:num_tokens] == -1)
    assert torch.all(topk[num_tokens:] == 7)


@pytest.mark.parametrize("num_heads", [NUM_HEADS, 20])
@pytest.mark.parametrize("num_tokens", [1, 33, 1100])
@torch.inference_mode()
def test_q_quant_fold_weights_matches_unfused(num_heads: int, num_tokens: int):
    torch.manual_seed(0)
    padded = 32
    q = torch.randn(
        num_tokens, num_heads * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).view(num_tokens, num_heads, HEAD_DIM)
    q[0, 0].zero_()  # all-zero group: exercises the amax floor
    kw = torch.randn(
        num_tokens, HEAD_DIM + num_heads, dtype=torch.bfloat16, device="cuda"
    )
    w = kw[:, HEAD_DIM:]
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5

    q_fp8, weights = indexer_q_quant_fold_weights(q, w, weight_scale, padded)

    ref_fp8, ref_scale = per_token_group_quant_fp8(
        q.reshape(-1, HEAD_DIM).contiguous(),
        HEAD_DIM,
        column_major_scales=False,
        use_ue8m0=True,
    )
    ref_fp8 = ref_fp8.view(num_tokens, num_heads, HEAD_DIM)
    ref_weights = w.float() * ref_scale.view(num_tokens, num_heads) * weight_scale

    assert q_fp8.shape == (num_tokens, padded, HEAD_DIM)
    assert weights.shape == (num_tokens, padded)
    assert torch.equal(
        q_fp8[:, :num_heads].view(torch.uint8), ref_fp8.view(torch.uint8)
    )
    torch.testing.assert_close(weights[:, :num_heads], ref_weights)
    if padded != num_heads:
        assert torch.count_nonzero(q_fp8[:, num_heads:].view(torch.uint8)) == 0
        assert torch.count_nonzero(weights[:, num_heads:]) == 0


Q_LORA = 1536
KV_LORA = 512
ROPE = 64


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)


def _dsa_caches(packed: bool, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8_ds_mla cache, indexer cache) views, both pages of one block row
    when packed."""
    if not packed:
        mla = torch.zeros(
            num_blocks, BLOCK, MLA_BYTES, dtype=torch.uint8, device="cuda"
        )
        return mla, _make_cache(False, num_blocks)[1]
    raw, index = _make_cache(True, num_blocks)
    return torch.as_strided(
        raw, (num_blocks, BLOCK, MLA_BYTES), (ROW_STRIDE, MLA_BYTES, 1)
    ), index


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_fused_prologue_matches_unfused(packed: bool, num_tokens: int):
    """dsa_fused_prologue == q/kv RMSNorm + fp8_ds_mla insert + indexer key insert
    + top-k fill, on strided views of the projection outputs."""
    torch.manual_seed(0)
    num_blocks = (num_tokens + 2 * BLOCK - 1) // BLOCK + 1
    qkv = torch.randn(
        num_tokens, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda"
    )
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    k = torch.randn(
        num_tokens, HEAD_DIM + NUM_HEADS, dtype=torch.bfloat16, device="cuda"
    )[:, :HEAD_DIM]
    q_w = torch.rand(Q_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    kv_w = torch.rand(KV_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    k_w = torch.rand(HEAD_DIM, dtype=torch.float32, device="cuda") + 0.5
    k_b = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1
    eps = 1e-6
    # Fewer slots than rows (cudagraph padding), plus a padded (-1) slot.
    num_slots = max(num_tokens - 1, 1)
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_slots]
    if num_slots > 2:
        slots[1] = -1
    slots = slots.to(torch.int64)

    ref_mla, ref_index = _dsa_caches(packed, num_blocks)
    ref_kv = _rms_norm_ref(kv_c, kv_w, eps)
    ops.concat_and_cache_mla(
        ref_kv[:num_slots].contiguous(),
        k_pe[:num_slots].contiguous(),
        ref_mla,
        slots,
        "fp8_ds_mla",
        torch.ones(1, dtype=torch.float32, device="cuda"),
    )
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), k_w, k_b, eps).to(k.dtype)
    ops.indexer_k_quant_and_cache(
        ref_k[:num_slots].contiguous(),
        ref_index,
        slots,
        HEAD_DIM,
        "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0",
    )

    mla, index = _dsa_caches(packed, num_blocks)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    q_out, kv_out = dsa_fused_prologue(
        q_c, kv_c, k_pe, k, q_w, kv_w, k_w, k_b, eps, topk, mla, slots, index, slots
    )

    # Only reduction order differs, so bytes match except for rare bf16 ties.
    torch.testing.assert_close(
        q_out, _rms_norm_ref(q_c, q_w, eps), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(kv_out, ref_kv, atol=2e-2, rtol=2e-2)
    mismatch = (mla != ref_mla).float().mean().item()
    assert mismatch < 2e-3, f"{mismatch:.2e} of MLA cache bytes differ"
    torch.testing.assert_close(
        _dequant(index), _dequant(ref_index), atol=0.07, rtol=0.07
    )
    assert torch.all(topk[:num_tokens] == -1)
    assert torch.all(topk[num_tokens:] == 7)


@torch.inference_mode()
@pytest.mark.parametrize("use_fp4_cache", [False, True])
def test_fused_prologue_without_caches(use_fp4_cache: bool):
    """Profile / dummy runs: norms only, no cache access."""
    qkv = torch.randn(3, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda")
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    k = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(Q_LORA, dtype=torch.bfloat16, device="cuda")
    kn = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    topk = torch.zeros(3, TOPK, dtype=torch.int32, device="cuda")
    prologue = mxfp4_fused_prologue if use_fp4_cache else dsa_fused_prologue
    q_out, kv_out = prologue(
        q_c, kv_c, k_pe, k, w, w[:KV_LORA], kn, kn, 1e-6, topk,
        None, None, None, None,
    )  # fmt: skip
    torch.testing.assert_close(q_out, _rms_norm_ref(q_c, w, 1e-6))
    torch.testing.assert_close(kv_out, _rms_norm_ref(kv_c, w[:KV_LORA], 1e-6))


# ---------------------------------------------------------------------------
# MXFP4 indexer key cache (attention_config.indexer_kv_dtype="mxfp4")
# ---------------------------------------------------------------------------

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _quantize_mxfp4_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., D] -> (packed [..., D // 2] uint8 with the even element in the low
    nibble, ue8m0 [..., D // 32] uint8), as DeepSeek V4's MXFP4 indexer (see
    tests/kernels/test_fused_indexer_q_rope_quant.py)."""
    shape = x.shape
    xb = x.float().reshape(-1, shape[-1] // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
    log2_scale = (amax / 6.0).log2().ceil().clamp(-127.0, 127.0)
    xs = (xb / log2_scale.exp2()).clamp(-6.0, 6.0)
    a = xs.abs()
    code = torch.zeros_like(a, dtype=torch.int32)
    # Round to nearest e2m1, ties to the even code.
    for c, bound, inclusive in (
        (1, 0.25, False),
        (2, 0.75, True),
        (3, 1.25, False),
        (4, 1.75, True),
        (5, 2.5, False),
        (6, 3.5, True),
        (7, 5.0, False),
    ):
        code = torch.where(a >= bound if inclusive else a > bound, c, code)
    sign = (xs.view(torch.int32) >> 31) & 1
    nibble = (code | (sign << 3)).to(torch.uint8).reshape(-1, shape[-1])
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    ue8m0 = (log2_scale + 127).to(torch.uint8)
    return (
        packed.reshape(*shape[:-1], shape[-1] // 2),
        ue8m0.reshape(*shape[:-1], shape[-1] // 32),
    )


def _dequant_mxfp4(packed: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES], device=packed.device)
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1)
    vals = vals.reshape(*packed.shape[:-1], -1, 32)
    scale = torch.pow(2.0, ue8m0.float() - 127)[..., None]
    return (vals * scale).reshape(*packed.shape[:-1], -1)


def _mxfp4_cache_rows(view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed [blocks*BLOCK, D // 2], ue8m0 [blocks*BLOCK, D // 32]) of an
    MXFP4 cache view: each block is [BLOCK x D // 2 values | BLOCK x D // 32
    scales]."""
    flat = torch.stack([view[b].reshape(-1) for b in range(view.shape[0])])
    values = flat[:, : BLOCK * HEAD_DIM // 2].reshape(-1, HEAD_DIM // 2)
    scales = flat[:, BLOCK * HEAD_DIM // 2 : BLOCK * MXFP4_ENTRY]
    return values, scales.reshape(-1, HEAD_DIM // 32)


def _ref_mxfp4_cache(
    packed: bool, num_blocks: int, k: torch.Tensor, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, view = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    k_packed, k_scale = _quantize_mxfp4_ref(k)
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        b, off = divmod(slot, BLOCK)
        flat = view[b].reshape(-1)
        d = HEAD_DIM // 2
        flat[off * d : (off + 1) * d] = k_packed[i]
        scale_at = BLOCK * d + off * (HEAD_DIM // 32)
        flat[scale_at : scale_at + HEAD_DIM // 32] = k_scale[i]
    return raw, view


def _assert_mxfp4_cache_close(out_raw, out_view, ref_raw, ref_view) -> None:
    # LayerNorm reduction order may flip a rare bf16 tie; dequantized values
    # then differ by one e2m1 step at most.
    mismatch = (out_raw != ref_raw).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of cache bytes differ"
    out = _dequant_mxfp4(*_mxfp4_cache_rows(out_view))
    ref = _dequant_mxfp4(*_mxfp4_cache_rows(ref_view))
    torch.testing.assert_close(out, ref, atol=0.3, rtol=0.3)


def _index_k_inputs(num_tokens: int):
    num_blocks = (num_tokens + 2 * BLOCK - 1) // BLOCK + 1
    kw = torch.randn(
        num_tokens, HEAD_DIM + NUM_HEADS, dtype=torch.bfloat16, device="cuda"
    )
    k = kw[:, :HEAD_DIM]
    weight = torch.rand(HEAD_DIM, dtype=torch.float32, device="cuda") + 0.5
    bias = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1
    num_slots = max(num_tokens - 1, 1)
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_slots]
    if num_slots > 2:
        slots[1] = -1
    return num_blocks, k, weight, bias, 1e-6, slots.to(torch.int64)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_k_norm_quant_insert_mxfp4(packed: bool, num_tokens: int):
    """The MXFP4 cache (68 B/token) gets LayerNorm + MXFP4 of the key in the
    DeepSeek V4 block layout; the fp8 test above covers the 132 B cache."""
    torch.manual_seed(0)
    num_blocks, k, weight, bias, eps, slots = _index_k_inputs(num_tokens)
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), weight, bias, eps).to(k.dtype)
    ref_raw, ref_view = _ref_mxfp4_cache(
        packed, num_blocks, ref_k[: slots.shape[0]], slots
    )

    out_raw, out_view = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    mxfp4_k_norm_quant_insert(k, weight, bias, eps, slots, out_view, topk, num_tokens)

    _assert_mxfp4_cache_close(out_raw, out_view, ref_raw, ref_view)
    if packed:
        assert torch.count_nonzero(out_raw[:, : BLOCK * MLA_BYTES]) == 0
    assert torch.all(topk[:num_tokens] == -1)
    assert torch.all(topk[num_tokens:] == 7)


@pytest.mark.parametrize("num_heads", [NUM_HEADS, 20])
@pytest.mark.parametrize("num_tokens", [1, 33, 1100])
@torch.inference_mode()
def test_q_quant_mxfp4(num_heads: int, num_tokens: int):
    torch.manual_seed(0)
    padded = 32
    q = torch.randn(
        num_tokens, num_heads * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).view(num_tokens, num_heads, HEAD_DIM)
    q[0, 0].zero_()  # all-zero blocks: exercises the amax floor
    q[-1, -1, :32] *= 1e-3  # blocks with different scales in one head
    w = torch.randn(
        num_tokens, HEAD_DIM + num_heads, dtype=torch.bfloat16, device="cuda"
    )[:, HEAD_DIM:]
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    # Hand the outputs dirty memory, so the zero padded heads are written.
    dirty = [
        torch.full((num_tokens, padded, n), 0xFF, dtype=torch.uint8, device="cuda")
        for n in (HEAD_DIM // 2, 4, 4)
    ]
    del dirty

    q_packed, q_scale, weights = mxfp4_q_quant(q, w, weight_scale, padded)

    ref_packed, ref_scale = _quantize_mxfp4_ref(q)
    assert q_packed.shape == (num_tokens, padded, HEAD_DIM // 2)
    assert q_scale.shape == (num_tokens, padded) and q_scale.dtype == torch.int32
    assert torch.equal(q_packed[:, :num_heads], ref_packed)
    scale_bytes = q_scale.view(torch.uint8).view(num_tokens, padded, 4)
    assert torch.equal(scale_bytes[:, :num_heads], ref_scale)
    # MXFP4 keeps the Q scales out of the weights.
    torch.testing.assert_close(weights[:, :num_heads], w.float() * weight_scale)
    if padded != num_heads:
        assert torch.count_nonzero(q_packed[:, num_heads:]) == 0
        assert torch.count_nonzero(q_scale[:, num_heads:]) == 0
        assert torch.count_nonzero(weights[:, num_heads:]) == 0


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [7, 300])
@torch.inference_mode()
def test_fused_prologue_mxfp4_index_cache(packed: bool, num_tokens: int):
    """dsa_fused_prologue writes the MXFP4 indexer cache like the standalone
    insert, next to an unchanged fp8_ds_mla insert."""
    torch.manual_seed(0)
    num_blocks, k, k_w, k_b, eps, slots = _index_k_inputs(num_tokens)
    qkv = torch.randn(
        num_tokens, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda"
    )
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    q_w = torch.rand(Q_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    kv_w = torch.rand(KV_LORA, dtype=torch.bfloat16, device="cuda") + 0.5

    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), k_w, k_b, eps).to(k.dtype)
    ref_raw, ref_view = _ref_mxfp4_cache(
        packed, num_blocks, ref_k[: slots.shape[0]], slots
    )
    mla = torch.zeros(num_blocks, BLOCK, MLA_BYTES, dtype=torch.uint8, device="cuda")
    ref_mla = mla.clone()
    ops.concat_and_cache_mla(
        _rms_norm_ref(kv_c, kv_w, eps)[: slots.shape[0]].contiguous(),
        k_pe[: slots.shape[0]].contiguous(),
        ref_mla,
        slots,
        "fp8_ds_mla",
        torch.ones(1, dtype=torch.float32, device="cuda"),
    )

    out_raw, out_view = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    topk = torch.full((num_tokens, TOPK), 7, dtype=torch.int32, device="cuda")
    mxfp4_fused_prologue(
        q_c, kv_c, k_pe, k, q_w, kv_w, k_w, k_b, eps, topk, mla, slots, out_view,
        slots,
    )  # fmt: skip

    _assert_mxfp4_cache_close(out_raw, out_view, ref_raw, ref_view)
    mismatch = (mla != ref_mla).float().mean().item()
    assert mismatch < 2e-3, f"{mismatch:.2e} of MLA cache bytes differ"
    assert torch.all(topk == -1)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300, 4100])
@torch.inference_mode()
def test_mxfp4_fused_prologue_bit_exact(packed: bool, num_tokens: int):
    """mxfp4_fused_prologue (ops/dsa_indexer_mxfp4) runs the stock FP8
    prologue's code for everything but the indexer key: its normed rows, MLA
    cache bytes and top-k fill are bit-identical to dsa_fused_prologue's, and its
    MXFP4 key bytes to the standalone mxfp4_k_norm_quant_insert's."""
    torch.manual_seed(num_tokens)
    num_blocks, k, k_w, k_b, eps, slots = _index_k_inputs(num_tokens)
    qkv = torch.randn(
        num_tokens, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda"
    )
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    q_w = torch.rand(Q_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    kv_w = torch.rand(KV_LORA, dtype=torch.bfloat16, device="cuda") + 0.5

    def mla_cache() -> torch.Tensor:
        return torch.full(
            (num_blocks, BLOCK, MLA_BYTES), 0x5A, dtype=torch.uint8, device="cuda"
        )

    def topk_rows() -> torch.Tensor:
        return torch.full((num_tokens + 3, TOPK), 7, dtype=torch.int32, device="cuda")

    ref_mla, ref_topk, ref_index = (
        mla_cache(),
        topk_rows(),
        _make_cache(False, num_blocks)[1],
    )
    ref_q, ref_kv = dsa_fused_prologue(
        q_c, kv_c, k_pe, k, q_w, kv_w, k_w, k_b, eps, ref_topk, ref_mla, slots,
        ref_index, slots,
    )  # fmt: skip
    ins_raw, ins_view = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    mxfp4_k_norm_quant_insert(
        k, k_w, k_b, eps, slots, ins_view, topk_rows(), num_tokens
    )

    mla, topk = mla_cache(), topk_rows()
    out_raw, out_view = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    q_out, kv_out = mxfp4_fused_prologue(
        q_c, kv_c, k_pe, k, q_w, kv_w, k_w, k_b, eps, topk, mla, slots, out_view,
        slots,
    )  # fmt: skip

    assert torch.equal(q_out, ref_q) and torch.equal(kv_out, ref_kv)
    assert torch.equal(mla, ref_mla)
    assert torch.equal(topk, ref_topk)
    assert torch.equal(out_raw, ins_raw)
    live = slots[slots >= 0]
    assert torch.count_nonzero(_mxfp4_cache_rows(out_view)[1][live]) > 0


def _logits_ref(q_deq, w, k_deq):
    """sum_h relu(q_h . k) * w_h for every (row, key)."""
    return (torch.einsum("thd,nd->thn", q_deq, k_deq).relu() * w[..., None]).sum(1)


@pytest.mark.skipif(
    not (current_platform.is_device_capability_family(100) and has_deep_gemm()),
    reason="DeepGEMM MXFP4 MQA logits need SM100",
)
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_mxfp4_cache_feeds_deepgemm_logits(packed: bool):
    """The cache and Q written here are what DeepGEMM's MXFP4 paged (decode)
    and gathered (prefill) MQA logits read: their logits match fp32 math over
    the dequantized MXFP4 Q and K."""
    torch.manual_seed(0)
    num_tokens, num_rows = 300, 6
    num_blocks, k, k_w, k_b, eps, _ = _index_k_inputs(num_tokens)
    # Tokens 0..num_tokens-1 of one request, scattered over the blocks.
    perm = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    block_table = perm[: (num_tokens + BLOCK - 1) // BLOCK]
    pos = torch.arange(num_tokens, device="cuda")
    slots = (block_table[pos // BLOCK].long() * BLOCK + pos % BLOCK).contiguous()
    raw, cache = _make_cache(packed, num_blocks, MXFP4_ENTRY)
    topk = torch.empty((num_tokens, TOPK), dtype=torch.int32, device="cuda")
    mxfp4_k_norm_quant_insert(k, k_w, k_b, eps, slots, cache, topk, num_tokens)

    q = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_rows, NUM_HEADS, dtype=torch.bfloat16, device="cuda")
    q_packed, q_scale, weights = mxfp4_q_quant(q, w, 0.1, NUM_HEADS)
    q_deq = _dequant_mxfp4(q_packed, q_scale.view(torch.uint8).view(num_rows, -1, 4))
    k_rows = _mxfp4_cache_rows(cache)
    k_deq = _dequant_mxfp4(*k_rows)[slots]
    ref = _logits_ref(q_deq, weights, k_deq)
    # Row i sees the first ctx[i] keys.
    ctx = torch.linspace(1, num_tokens, num_rows, device="cuda").to(torch.int32)

    q_scale_mx, w_mx = mqa_logits_mx_inputs(q_packed, q_scale, weights, torch.float32)
    # Decode: varlen paged logits as the indexer metadata builds them (one
    # row per query token, per-token block table, request index per row).
    req = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    context_lens = ctx.view(-1, 1).contiguous()
    paged = fp8_fp4_paged_mqa_logits(
        (q_packed.view(torch.int8).unsqueeze(1), q_scale_mx.unsqueeze(1)),
        kv_cache_as_quant_view(cache, HEAD_DIM, use_fp4_cache=True),
        w_mx,
        context_lens,
        block_table.expand(num_rows, -1).contiguous(),
        get_paged_mqa_logits_metadata(context_lens, BLOCK, get_num_sms(), req),
        max_model_len=num_tokens,
        indices=req,
    )
    # Prefill: the cache gathered to contiguous K, then ragged logits.
    k_quant = torch.empty(num_tokens, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    k_scale = torch.empty(num_tokens, HEAD_DIM // 32, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(
        cache,
        k_quant,
        k_scale,
        block_table.view(1, -1).contiguous(),
        torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(k_quant, k_rows[0][slots])
    ragged = fp8_fp4_mqa_logits(
        (q_packed.view(torch.int8), q_scale_mx),
        (k_quant.view(torch.int8), k_scale.view(torch.int32).squeeze(-1)),
        w_mx,
        torch.zeros(num_rows, dtype=torch.int32, device="cuda"),
        ctx,
        num_tokens,
    )
    for i, n in enumerate(ctx.tolist()):
        for logits in (paged, ragged):
            got = logits[i, :n].float()
            err = (got - ref[i, :n]).norm() / ref[i, :n].norm().clamp_min(1e-6)
            # SM100 sums the heads in bf16; a layout error is O(1).
            assert err < 3e-2, (i, n, err.item())


@pytest.mark.skipif(
    not (current_platform.is_device_capability_family(100) and has_deep_gemm()),
    reason="DeepGEMM MXFP4 MQA logits need SM100",
)
@pytest.mark.parametrize("use_fp4_cache", [False, True])
@torch.inference_mode()
def test_indexer_score_reads_its_cache_format(
    monkeypatch: pytest.MonkeyPatch, workspace_init, use_fp4_cache: bool
):
    """KimiK3DSAIndexer.score quantizes Q to the indexer's cache format and runs
    sparse_attn_indexer with the matching flag: the decode top-k over a cache
    written by the fused insert is the top-k of fp32 math over the dequantized
    Q and K."""
    torch.manual_seed(0)
    entry = MXFP4_ENTRY if use_fp4_cache else ENTRY
    seq_lens = torch.tensor([1000, 3000, 6100, 4500], device="cuda").int()
    batch = seq_lens.shape[0]
    max_blocks = (int(seq_lens.max()) + BLOCK - 1) // BLOCK
    num_blocks = batch * max_blocks + 1
    perm = torch.randperm(num_blocks, device="cuda").int()
    block_table = perm[: batch * max_blocks].view(batch, max_blocks).contiguous()
    req = torch.repeat_interleave(torch.arange(batch, device="cuda"), seq_lens.long())
    pos = torch.cat([torch.arange(n, device="cuda") for n in seq_lens.tolist()])
    slots = block_table[req, pos // BLOCK].long() * BLOCK + pos % BLOCK
    _, cache = _make_cache(True, num_blocks, entry)
    k = torch.randn(slots.shape[0], HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_w = torch.rand(HEAD_DIM, device="cuda") + 0.5
    k_b = torch.randn(HEAD_DIM, device="cuda") * 0.1
    scratch = torch.empty(slots.shape[0], TOPK, dtype=torch.int32, device="cuda")
    insert = mxfp4_k_norm_quant_insert if use_fp4_cache else indexer_k_norm_quant_insert
    insert(k, k_w, k_b, 1e-6, slots, cache, scratch, slots.shape[0])

    prefix = "model.layers.3.self_attn.indexer.k_cache"
    topk = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
    indexer = SimpleNamespace(
        use_fp4_cache=use_fp4_cache,
        index_softmax_scale=HEAD_DIM**-0.5,
        indexer_weight_scale=NUM_HEADS**-0.5,
        padded_index_num_heads=NUM_HEADS,
        k_cache=SimpleNamespace(prefix=prefix, kv_cache=cache),
        quant_block_size=HEAD_DIM,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        index_head_dim=HEAD_DIM,
        max_model_len=max_blocks * BLOCK,
        max_total_seq_len=max_blocks * BLOCK,
        topk_indices_buffer=topk,
        use_fused_k_insert=True,
        use_pcp=False,
        _parallel_config=SimpleNamespace(pcp_shard_decode_requests=False),
        dense_mha_metadata_layer_name="",
        dcp_rank=0,
        dcp_world_size=1,
        cp_kv_cache_interleave_size=1,
        topk_backend="auto",
    )
    context_lens = seq_lens.view(-1, 1).contiguous()
    # Varlen decode, as the metadata builder sets it up: request index per row.
    rows = torch.arange(batch, dtype=torch.int32, device="cuda")
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max()),
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        num_decodes=batch,
        num_decode_tokens=batch,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=context_lens,
            decode_lens=torch.ones(batch, dtype=torch.int32, device="cuda"),
            requires_padding=False,
            schedule_metadata=get_paged_mqa_logits_metadata(
                context_lens, BLOCK, get_num_sms(), rows
            ),
            indices=rows,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={prefix: metadata},
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ),
    )
    q = torch.randn(batch, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(batch, NUM_HEADS, dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.empty(batch, 16, dtype=torch.bfloat16, device="cuda")
    KimiK3DSAIndexer.score(indexer, hidden_states, q, w, None)

    # Reference: the same Q quant, fp32 math over the dequantized Q and K.
    weight_scale = indexer.index_softmax_scale * indexer.indexer_weight_scale
    if use_fp4_cache:
        q_packed, q_scale, weights = mxfp4_q_quant(q, w, weight_scale, NUM_HEADS)
        q_deq = _dequant_mxfp4(q_packed, q_scale.view(torch.uint8).view(batch, -1, 4))
        k_deq = _dequant_mxfp4(*_mxfp4_cache_rows(cache))
    else:
        q_fp8, weights = indexer_q_quant_fold_weights(q, w, weight_scale, NUM_HEADS)
        q_deq = q_fp8.float()
        k_deq = _dequant(cache)
    for b, n in enumerate(seq_lens.tolist()):
        keys = k_deq[slots[req == b]]
        ref = _logits_ref(q_deq[b : b + 1], weights[b : b + 1], keys)[0]
        got = topk[b][topk[b] >= 0]
        assert got.numel() == min(n, TOPK) and int(got.max()) < n, b
        if n <= TOPK:
            assert torch.equal(got.sort().values, torch.arange(n, device="cuda"))
            continue
        ref_set = ref.topk(TOPK).indices
        # SM100 sums the heads in bf16: only near-ties at the cut may differ.
        recall = torch.isin(got, ref_set).float().mean().item()
        assert recall > 0.97, (b, recall)
