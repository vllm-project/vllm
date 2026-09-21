# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-kernel-vs-eager-reference parity tests for the DeepSeek-V4 CPU port's
sgl-kernels ops (csrc/cpu/sgl-kernels/). Each test carries its own reference
reimplementation of the kernel's math rather than depending on production
model code.
"""

from unittest import mock

import pytest
import torch

import vllm.model_executor.kernels.mhc as mhc_kernels
from tests.kernels.test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    HEAD_DIM,
    NOPE_DIM,
    QUANT_BLOCK,
    ROPE_DIM,
    make_cos_sin_cache,
)
from tests.kernels.utils import bf16_ulp_distance, fp8_ulp_distance
from vllm._custom_ops import (
    compress_norm_rope_store_cpu,
    compress_norm_rope_store_indexer_cpu,
    flash_mla_with_kvcache_cpu,
    fp8_paged_mqa_logits_cpu,
    fused_indexer_q_rope_quant_cpu,
    fused_qnorm_rope_kv_insert_cpu,
    inverse_gptj_rope_o_proj_cpu,
    save_partial_states_cpu,
)
from vllm.models.deepseek_v4.common.ops import fused_indexer_q_rope_quant
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

# Main attention cache (fp8_ds_mla): NoPE fp8 in QUANT_BLOCK-wide UE8M0 blocks,
# RoPE bf16, one scale byte per block (+1 pad byte) per token.
SCALE_BYTES_PER_TOKEN = NOPE_DIM // QUANT_BLOCK + 1
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2
FP8_MAX = 448.0

# Indexer K-cache: all 128 dims in one fp8 block, one raw fp32 scale per token.
INDEXER_HEAD_DIM = 128
INDEXER_NOPE_DIM = INDEXER_HEAD_DIM - ROPE_DIM
INDEXER_SCALE_BYTES = 4


def _op_available(*names: str) -> bool:
    return all(hasattr(torch.ops._C, name) for name in names)


pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or not _op_available(
        "save_partial_states_cpu",
        "compress_norm_rope_store_cpu",
        "compress_norm_rope_store_indexer_cpu",
        "flash_mla_with_kvcache_cpu",
        "fused_qnorm_rope_kv_insert_cpu",
        "inverse_gptj_rope_o_proj_cpu",
        "fp8_paged_mqa_logits_cpu",
        "fused_indexer_q_rope_quant_cpu",
    ),
    reason="CPU not available or DeepSeek-V4 CPU kernels not built in",
)


def _apply_gptj_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def _make_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    block_bytes = block_size * TOKEN_DATA_BYTES + block_size * SCALE_BYTES_PER_TOKEN
    return torch.zeros(num_blocks, block_bytes, dtype=torch.uint8)


def _decode_token_regions(
    cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int
):
    """Slice out, per valid token, the raw NoPE-fp8 / RoPE-bf16 / scale byte
    regions this layout defines."""
    valid = slot_mapping >= 0
    idx = valid.nonzero(as_tuple=True)[0]
    slots = slot_mapping[idx]
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    pos_in_block = slots % block_size

    nope, rope, scale = [], [], []
    for b, p in zip(block_idx.tolist(), pos_in_block.tolist()):
        token_off = p * TOKEN_DATA_BYTES
        nope.append(cache[b, token_off : token_off + NOPE_DIM])
        rope.append(cache[b, token_off + NOPE_DIM : token_off + TOKEN_DATA_BYTES])
        scale_off = block_size * TOKEN_DATA_BYTES + p * SCALE_BYTES_PER_TOKEN
        scale.append(cache[b, scale_off : scale_off + SCALE_BYTES_PER_TOKEN - 1])

    if not nope:
        empty = lambda n: torch.empty(0, n, dtype=torch.uint8)
        return empty(NOPE_DIM), empty(ROPE_DIM * 2), empty(SCALE_BYTES_PER_TOKEN - 1)
    return torch.stack(nope), torch.stack(rope), torch.stack(scale)


def _assert_cache_parity(cache_fused, cache_ref, slot_mapping, block_size):
    """NoPE fp8 + the UE8M0 scale byte are a deterministic quant of an
    un-rotated bf16 value, so they match tightly; the RoPE region is rotated
    in fp32 before its bf16 store and eager/kernel can round a tie to
    opposite sides, so allow <=1 ULP there."""
    nope_ref, rope_ref, scale_ref = _decode_token_regions(
        cache_ref, slot_mapping, block_size
    )
    nope_fused, rope_fused, scale_fused = _decode_token_regions(
        cache_fused, slot_mapping, block_size
    )
    if nope_ref.numel() == 0:
        return

    max_ulp = int(fp8_ulp_distance(nope_fused, nope_ref).max().item())
    assert max_ulp <= 1, f"NoPE fp8 differs by {max_ulp} ULP (>1)"

    rope_ref_bf16 = rope_ref.contiguous().view(torch.bfloat16)
    rope_fused_bf16 = rope_fused.contiguous().view(torch.bfloat16)
    max_ulp_rope = int(bf16_ulp_distance(rope_fused_bf16, rope_ref_bf16).max().item())
    assert max_ulp_rope <= 1, f"RoPE bf16 differs by {max_ulp_rope} ULP (>1)"

    scale_diff = (scale_fused.int() - scale_ref.int()).abs().max().item()
    assert scale_diff <= 1, f"scale byte differs by {scale_diff} (>1)"


def _make_indexer_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    block_bytes = block_size * INDEXER_HEAD_DIM + block_size * INDEXER_SCALE_BYTES
    return torch.zeros(num_blocks, block_bytes, dtype=torch.uint8)


def _assert_indexer_cache_parity(cache_fused, cache_ref, slot_mapping, block_size):
    valid = slot_mapping >= 0
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    slots = slot_mapping[idx]
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    pos_in_block = slots % block_size

    fp8_ref, scale_ref = [], []
    fp8_fused, scale_fused = [], []
    for b, p in zip(block_idx.tolist(), pos_in_block.tolist()):
        token_off = p * INDEXER_HEAD_DIM
        scale_off = block_size * INDEXER_HEAD_DIM + p * INDEXER_SCALE_BYTES
        fp8_ref.append(cache_ref[b, token_off : token_off + INDEXER_HEAD_DIM])
        fp8_fused.append(cache_fused[b, token_off : token_off + INDEXER_HEAD_DIM])
        scale_ref.append(cache_ref[b, scale_off : scale_off + INDEXER_SCALE_BYTES])
        scale_fused.append(cache_fused[b, scale_off : scale_off + INDEXER_SCALE_BYTES])

    fp8_ref_t = torch.stack(fp8_ref).view(torch.float8_e4m3fn)
    fp8_fused_t = torch.stack(fp8_fused).view(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8_fused_t.float(), fp8_ref_t.float())

    scale_ref_t = torch.stack(scale_ref).view(torch.float32)
    scale_fused_t = torch.stack(scale_fused).view(torch.float32)
    torch.testing.assert_close(scale_fused_t, scale_ref_t)


# ---------------------------------------------------------------------------
# save_partial_states_cpu / compress_norm_rope_store_cpu (compressor.cpp)
# ---------------------------------------------------------------------------
#
# The compressor's state cache is pooled with the main KV cache's physical
# pages and page-aligned, so it is genuinely 3D with a block-to-block stride
# that need not equal block_size * width -- built here via torch.as_strided
# over an over-sized flat buffer rather than a plain contiguous tensor, so a
# regression back to flattening via .view() would be caught.


def _make_padded_block_cache(
    num_blocks: int, block_size: int, width: int, pad_elems: int
) -> torch.Tensor:
    page_stride = block_size * width + pad_elems
    flat = torch.randn(num_blocks, page_stride, dtype=torch.float32)
    return torch.as_strided(
        flat, (num_blocks, block_size, width), (page_stride, width, 1)
    )


def _dup_padded_block_cache(
    num_blocks: int, block_size: int, width: int, pad_elems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independently-stored padded caches with identical values, for ops
    that mutate the cache in place."""
    page_stride = block_size * width + pad_elems
    template = torch.randn(num_blocks, page_stride, dtype=torch.float32)

    def _view(flat: torch.Tensor) -> torch.Tensor:
        return torch.as_strided(
            flat, (num_blocks, block_size, width), (page_stride, width, 1)
        )

    return _view(template.clone()), _view(template.clone())


def _run_save_partial_states_eager(
    state_cache, kv, score, ape, positions, slot_mapping, compress_ratio
):
    """Reference: state_cache[block_idx, pos_in_block, :W] = kv;
    state_cache[..., W:] = score + ape[position % compress_ratio]; skip
    slot < 0. Mutates state_cache in place, matching the op's contract."""
    state_width = kv.shape[-1]
    block_size = state_cache.shape[1]
    valid = slot_mapping >= 0
    idx = valid.nonzero(as_tuple=True)[0]
    slots = slot_mapping[idx]
    block_idx = slots // block_size
    pos_in_block = slots % block_size

    state_cache[block_idx, pos_in_block, :state_width] = kv[idx]
    ape_row = ape[positions[idx] % compress_ratio]
    state_cache[block_idx, pos_in_block, state_width:] = score[idx] + ape_row


@pytest.mark.parametrize("pad_elems", [0, 24])
@pytest.mark.parametrize("num_tokens", [1, 17])
def test_save_partial_states_cpu_matches_eager(num_tokens: int, pad_elems: int):
    torch.manual_seed(0)
    compress_ratio = 4
    block_size = 4
    state_width = 512
    width = 2 * state_width
    num_blocks = (num_tokens + 4 + block_size - 1) // block_size + 1
    num_slots = num_blocks * block_size

    kv = torch.randn(num_tokens, state_width, dtype=torch.float32)
    score = torch.randn(num_tokens, state_width, dtype=torch.float32)
    ape = torch.randn(compress_ratio, state_width, dtype=torch.float32)
    positions = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64)

    ref_cache, fused_cache = _dup_padded_block_cache(
        num_blocks, block_size, width, pad_elems
    )

    slot_mapping = torch.randperm(num_slots)[:num_tokens].to(torch.int64)
    if num_tokens > 1:
        slot_mapping[-1] = -1  # exercise the "skip invalid slot" branch

    _run_save_partial_states_eager(
        ref_cache, kv, score, ape, positions, slot_mapping, compress_ratio
    )
    save_partial_states_cpu(kv, score, ape, positions, fused_cache, slot_mapping)

    torch.testing.assert_close(fused_cache, ref_cache)


def _run_compress_eager(
    state_cache,
    gather_slots,
    positions,
    kv_slot_mapping,
    rms_norm_weight,
    rms_norm_eps,
    cos_sin_cache,
    kv_cache,
    kv_cache_block_size,
    compress_ratio,
):
    """Reference for the per-channel-softmax-pooling + RMSNorm + NoPE-UE8M0
    quant + GPT-J RoPE math writing the main (head_dim=512) fp8_ds_mla
    layout."""
    num_tokens, window = gather_slots.shape
    state_width = state_cache.shape[-1] // 2
    state_block_size = state_cache.shape[1]
    num_quant_blocks = NOPE_DIM // QUANT_BLOCK

    w_idx = torch.arange(window)
    head_offset = torch.where(w_idx >= compress_ratio, HEAD_DIM, 0)

    for t in range(num_tokens):
        position = int(positions[t])
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[t])
        if kv_slot < 0:
            continue

        slots = gather_slots[t]
        valid_w = slots >= 0
        safe_slots = slots.clamp(min=0)
        block_idx = safe_slots // state_block_size
        pos_in_block = safe_slots % state_block_size

        rows = state_cache[block_idx, pos_in_block]  # [window, 2*state_width]
        col = head_offset.unsqueeze(-1) + torch.arange(HEAD_DIM)
        kv_vals = torch.gather(rows, 1, col)
        score_col = state_width + col
        score_vals = torch.gather(rows, 1, score_col)
        score_vals = torch.where(
            valid_w.unsqueeze(-1), score_vals, torch.full_like(score_vals, -torch.inf)
        )

        weights = torch.softmax(score_vals, dim=0)
        compressed = (kv_vals * weights).sum(dim=0)  # [HEAD_DIM]

        variance = compressed.pow(2).mean()
        normed = compressed * torch.rsqrt(variance + rms_norm_eps) * rms_norm_weight

        quant_input = normed.to(torch.bfloat16).to(torch.float32)

        compressed_pos = (position // compress_ratio) * compress_ratio
        cos = cos_sin_cache[compressed_pos, : ROPE_DIM // 2]
        sin = cos_sin_cache[compressed_pos, ROPE_DIM // 2 :]
        rotated_rope = _apply_gptj_rope(normed[NOPE_DIM:].clone(), cos, sin)

        nope_vals = quant_input[:NOPE_DIM].view(num_quant_blocks, QUANT_BLOCK)
        absmax = nope_vals.abs().amax(dim=-1).clamp(min=1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        inv_scale = torch.exp2(-exponent)
        scaled = (nope_vals * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        nope_fp8_bytes = scaled.to(torch.float8_e4m3fn).reshape(-1).view(torch.uint8)
        scale_bytes = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
        rope_bf16_bytes = rotated_rope.to(torch.bfloat16).contiguous().view(torch.uint8)

        block_idx_kv = kv_slot // kv_cache_block_size
        pos_in_block_kv = kv_slot % kv_cache_block_size
        token_off = pos_in_block_kv * TOKEN_DATA_BYTES
        kv_cache[block_idx_kv, token_off : token_off + NOPE_DIM] = nope_fp8_bytes
        kv_cache[block_idx_kv, token_off + NOPE_DIM : token_off + TOKEN_DATA_BYTES] = (
            rope_bf16_bytes
        )
        scale_off = (
            kv_cache_block_size * TOKEN_DATA_BYTES
            + pos_in_block_kv * SCALE_BYTES_PER_TOKEN
        )
        kv_cache[block_idx_kv, scale_off : scale_off + num_quant_blocks] = scale_bytes
        kv_cache[block_idx_kv, scale_off + num_quant_blocks] = 0


def _make_compress_inputs(
    num_tokens: int,
    compress_ratio: int,
    overlap: bool,
    pad_elems: int = 24,
    max_pos: int = 4096,
    invalid_window_frac: float = 0.1,
    head_dim: int = HEAD_DIM,
):
    torch.manual_seed(0)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio
    state_block_size = 4 if compress_ratio == 4 else 8
    num_state_slots = num_tokens * window + 8
    num_state_blocks = (num_state_slots + state_block_size - 1) // state_block_size + 1

    state_cache = _make_padded_block_cache(
        num_state_blocks, state_block_size, 2 * state_width, pad_elems
    )
    gather_slots = torch.randint(
        0, num_state_blocks * state_block_size, (num_tokens, window)
    ).to(torch.int64)
    invalid_mask = torch.rand(num_tokens, window) < invalid_window_frac
    # Never invalidate every column of a row (would force max=-inf -> NaN).
    invalid_mask[:, -1] = False
    gather_slots = torch.where(
        invalid_mask, torch.full_like(gather_slots, -1), gather_slots
    )

    base = (torch.arange(num_tokens) + 1) * compress_ratio
    is_boundary = torch.arange(num_tokens) % 2 == 0
    positions = torch.where(is_boundary, base - 1, base + 1).to(torch.int64)

    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    if num_tokens > 1:
        kv_slot_mapping[1] = -1  # exercise the "skip invalid output slot" branch

    rms_norm_weight = torch.randn(head_dim, dtype=torch.float32)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, "cpu")

    return (
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        is_boundary,
    )


@pytest.mark.parametrize("compress_ratio,overlap", [(128, False), (4, True)])
@pytest.mark.parametrize("num_tokens", [1, 17])
def test_compress_norm_rope_store_cpu_matches_eager(
    compress_ratio: int, overlap: bool, num_tokens: int
):
    eps = 1e-6
    block_size = 16
    (
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        is_boundary,
    ) = _make_compress_inputs(num_tokens, compress_ratio, overlap)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_ref = _make_cache(num_blocks, block_size)
    kv_cache_fused = kv_cache_ref.clone()

    _run_compress_eager(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_ref,
        block_size,
        compress_ratio,
    )
    compress_norm_rope_store_cpu(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_fused,
        block_size,
        compress_ratio,
    )

    # Only tokens that hit the compression boundary (and have a valid output
    # slot) were written -- restrict parity checks to those.
    written_slot_mapping = torch.where(
        is_boundary, kv_slot_mapping, torch.full_like(kv_slot_mapping, -1)
    )
    _assert_cache_parity(kv_cache_fused, kv_cache_ref, written_slot_mapping, block_size)


def test_compress_norm_rope_store_cpu_non_contiguous_kv_cache_row_stride():
    """The main (output) kv_cache is a separate physical buffer from the
    state cache and can independently be a strided view."""
    eps = 1e-6
    compress_ratio, overlap = 128, False
    num_tokens = 5
    block_size = 16
    cache_pad = 32

    (
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        is_boundary,
    ) = _make_compress_inputs(num_tokens, compress_ratio, overlap)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_ref = _make_cache(num_blocks, block_size)
    block_bytes = kv_cache_ref.shape[1]
    padded_cache_ref = torch.zeros(
        num_blocks, block_bytes + cache_pad, dtype=torch.uint8
    )
    kv_cache_ref_view = padded_cache_ref[:, :block_bytes]
    kv_cache_ref_view.copy_(kv_cache_ref)
    padded_cache_fused = padded_cache_ref.clone()
    kv_cache_fused_view = padded_cache_fused[:, :block_bytes]

    _run_compress_eager(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_ref_view,
        block_size,
        compress_ratio,
    )
    compress_norm_rope_store_cpu(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_fused_view,
        block_size,
        compress_ratio,
    )

    written_slot_mapping = torch.where(
        is_boundary, kv_slot_mapping, torch.full_like(kv_slot_mapping, -1)
    )
    _assert_cache_parity(
        kv_cache_fused_view, kv_cache_ref_view, written_slot_mapping, block_size
    )


def _run_compress_indexer_eager(
    state_cache,
    gather_slots,
    positions,
    kv_slot_mapping,
    rms_norm_weight,
    rms_norm_eps,
    cos_sin_cache,
    kv_cache,
    kv_cache_block_size,
    compress_ratio,
):
    """Same softmax-pool + RMSNorm + GPT-J RoPE prologue as the head=512
    path, but the indexer's cache quantizes all 128 post-RoPE dims as one
    FP8 block with a single raw fp32 scale per token (matching
    _indexer_k_quant_and_cache_cpu's layout), not 7 separate UE8M0 blocks."""
    num_tokens, window = gather_slots.shape
    state_width = state_cache.shape[-1] // 2
    state_block_size = state_cache.shape[1]

    w_idx = torch.arange(window)
    head_offset = torch.where(w_idx >= compress_ratio, INDEXER_HEAD_DIM, 0)

    for t in range(num_tokens):
        position = int(positions[t])
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[t])
        if kv_slot < 0:
            continue

        slots = gather_slots[t]
        valid_w = slots >= 0
        safe_slots = slots.clamp(min=0)
        block_idx = safe_slots // state_block_size
        pos_in_block = safe_slots % state_block_size

        rows = state_cache[block_idx, pos_in_block]  # [window, 2*state_width]
        col = head_offset.unsqueeze(-1) + torch.arange(INDEXER_HEAD_DIM)
        kv_vals = torch.gather(rows, 1, col)
        score_col = state_width + col
        score_vals = torch.gather(rows, 1, score_col)
        score_vals = torch.where(
            valid_w.unsqueeze(-1), score_vals, torch.full_like(score_vals, -torch.inf)
        )

        weights = torch.softmax(score_vals, dim=0)
        compressed = (kv_vals * weights).sum(dim=0)  # [INDEXER_HEAD_DIM]

        variance = compressed.pow(2).mean()
        normed = compressed * torch.rsqrt(variance + rms_norm_eps) * rms_norm_weight

        compressed_pos = (position // compress_ratio) * compress_ratio
        cos = cos_sin_cache[compressed_pos, : ROPE_DIM // 2]
        sin = cos_sin_cache[compressed_pos, ROPE_DIM // 2 :]
        rotated_rope = _apply_gptj_rope(normed[INDEXER_NOPE_DIM:].clone(), cos, sin)
        full = torch.cat((normed[:INDEXER_NOPE_DIM], rotated_rope))

        quant_input = full.to(torch.bfloat16).to(torch.float32)
        absmax = quant_input.abs().amax().clamp(min=1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        scale = torch.exp2(exponent)
        scaled = (quant_input / scale).clamp(-FP8_MAX, FP8_MAX)
        fp8_bytes = scaled.to(torch.float8_e4m3fn).view(torch.uint8)

        block_idx_kv = kv_slot // kv_cache_block_size
        pos_in_block_kv = kv_slot % kv_cache_block_size
        token_off = pos_in_block_kv * INDEXER_HEAD_DIM
        kv_cache[block_idx_kv, token_off : token_off + INDEXER_HEAD_DIM] = fp8_bytes
        scale_off = (
            kv_cache_block_size * INDEXER_HEAD_DIM
            + pos_in_block_kv * INDEXER_SCALE_BYTES
        )
        kv_cache[block_idx_kv, scale_off : scale_off + INDEXER_SCALE_BYTES] = (
            scale.reshape(1).view(torch.uint8)
        )


@pytest.mark.parametrize("compress_ratio,overlap", [(128, False), (4, True)])
@pytest.mark.parametrize("num_tokens", [1, 17])
def test_compress_norm_rope_store_indexer_cpu_matches_eager(
    compress_ratio: int, overlap: bool, num_tokens: int
):
    eps = 1e-6
    block_size = 16
    (
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        is_boundary,
    ) = _make_compress_inputs(
        num_tokens, compress_ratio, overlap, head_dim=INDEXER_HEAD_DIM
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_ref = _make_indexer_cache(num_blocks, block_size)
    kv_cache_fused = kv_cache_ref.clone()

    _run_compress_indexer_eager(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_ref,
        block_size,
        compress_ratio,
    )
    compress_norm_rope_store_indexer_cpu(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        eps,
        cos_sin_cache,
        kv_cache_fused,
        block_size,
        compress_ratio,
    )

    written_slot_mapping = torch.where(
        is_boundary, kv_slot_mapping, torch.full_like(kv_slot_mapping, -1)
    )
    _assert_indexer_cache_parity(
        kv_cache_fused, kv_cache_ref, written_slot_mapping, block_size
    )


# ---------------------------------------------------------------------------
# flash_mla_with_kvcache_cpu (flash_mla.cpp): fused sparse MQA decode attn
# ---------------------------------------------------------------------------


def _pack_flash_mla_cache(kv_f32: torch.Tensor, block_size: int):
    """Quantize kv_f32 ([N, HEAD_DIM]) into the fp8_ds_mla byte layout and
    return (cache, dequant_ref) -- the exact post-quantization value the
    kernel's dequant should reproduce. Slot id == row index, packed
    contiguously from block 0."""
    n = kv_f32.shape[0]
    num_blocks = max(1, -(-n // block_size))
    cache = _make_cache(num_blocks, block_size)
    if n == 0:
        return cache, kv_f32.new_zeros(0, HEAD_DIM)

    num_quant_blocks = NOPE_DIM // QUANT_BLOCK
    nope = kv_f32[:, :NOPE_DIM].reshape(n, num_quant_blocks, QUANT_BLOCK)
    absmax = nope.abs().amax(dim=-1).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
    inv_scale = torch.exp2(-exponent)
    scaled = (nope * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    nope_fp8 = scaled.to(torch.float8_e4m3fn)
    scale = torch.exp2(exponent)
    nope_dequant = (nope_fp8.float() * scale.unsqueeze(-1)).reshape(n, NOPE_DIM)

    rope_bf16 = kv_f32[:, NOPE_DIM:].to(torch.bfloat16)
    dequant_ref = torch.cat([nope_dequant, rope_bf16.float()], dim=-1)

    slot_ids = torch.arange(n)
    block_idx = slot_ids // block_size
    pos_in_block = slot_ids % block_size

    nope_fp8_bytes = nope_fp8.reshape(n, NOPE_DIM).contiguous().view(torch.uint8)
    rope_bf16_bytes = rope_bf16.contiguous().view(torch.uint8)
    scale_bytes = (exponent + 127.0).clamp(0, 255).to(torch.uint8)

    arange_nope = torch.arange(NOPE_DIM)
    arange_rope = torch.arange(ROPE_DIM * 2)
    arange_scale = torch.arange(num_quant_blocks)
    token_base = pos_in_block.unsqueeze(-1) * TOKEN_DATA_BYTES
    row_idx = block_idx.unsqueeze(-1)

    cache[row_idx, token_base + arange_nope] = nope_fp8_bytes
    cache[row_idx, token_base + NOPE_DIM + arange_rope] = rope_bf16_bytes
    scale_base = (
        block_size * TOKEN_DATA_BYTES
        + pos_in_block.unsqueeze(-1) * SCALE_BYTES_PER_TOKEN
    )
    cache[row_idx, scale_base + arange_scale] = scale_bytes
    cache[block_idx, scale_base.squeeze(-1) + num_quant_blocks] = 0

    return cache, dequant_ref


def _flash_mla_eager(
    q: torch.Tensor,
    window_dequant: torch.Tensor,
    window_slots: torch.Tensor,
    compressed_dequant: torch.Tensor,
    compressed_slots: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """MQA attention: gather the (already-dequantized) window+compressed KV
    rows referenced by *_slots (-1 = invalid/masked), append a learned
    per-head sink logit with zero value contribution, softmax, weighted
    sum."""
    num_tokens, num_heads, _ = q.shape
    q_f32 = q.float()

    def gather(dequant, slots):
        valid = slots >= 0
        safe = slots.clamp(min=0)
        kv = dequant[safe]  # [T, K, D]
        return kv, valid

    win_kv, win_valid = gather(window_dequant, window_slots)
    if compressed_slots.shape[1] > 0:
        comp_kv, comp_valid = gather(compressed_dequant, compressed_slots)
        kv = torch.cat([win_kv, comp_kv], dim=1)
        valid = torch.cat([win_valid, comp_valid], dim=1)
    else:
        kv = win_kv
        valid = win_valid

    scores = torch.einsum("thd,tkd->thk", q_f32, kv) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sink = attn_sink.view(1, num_heads, 1).expand(num_tokens, num_heads, 1)
    scores_full = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(scores_full, dim=-1)
    out = torch.einsum("thk,tkd->thd", probs[..., :-1], kv)
    return out.to(q.dtype)


@pytest.mark.parametrize("num_heads", [1, 64])
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize("swa_only", [True, False])
def test_flash_mla_cpu_kernel_matches_eager(
    num_heads: int, num_tokens: int, swa_only: bool
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    window_block_size = 16
    compressed_block_size = 8
    num_window = 6
    num_compressed = 0 if swa_only else 5

    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype)
    scale = 1.0 / (HEAD_DIM**0.5)
    attn_sink = torch.randn(num_heads, dtype=torch.float32)

    num_window_slots = num_tokens * num_window + window_block_size
    window_kv_f32 = torch.randn(num_window_slots, HEAD_DIM)
    window_cache, window_dequant = _pack_flash_mla_cache(
        window_kv_f32, window_block_size
    )
    window_slots = torch.randint(0, num_window_slots, (num_tokens, num_window))
    window_slots[:, 0] = -1  # exercise invalid-slot masking

    if num_compressed > 0:
        num_compressed_slots = num_tokens * num_compressed + compressed_block_size
        compressed_kv_f32 = torch.randn(num_compressed_slots, HEAD_DIM)
        compressed_cache, compressed_dequant = _pack_flash_mla_cache(
            compressed_kv_f32, compressed_block_size
        )
        compressed_slots = torch.randint(
            0, num_compressed_slots, (num_tokens, num_compressed)
        )
        if num_compressed > 1:
            compressed_slots[:, -1] = -1
    else:
        compressed_cache = window_cache
        compressed_dequant = window_dequant.new_zeros(0, HEAD_DIM)
        compressed_slots = window_slots.new_full((num_tokens, 0), -1)

    out_ref = _flash_mla_eager(
        q,
        window_dequant,
        window_slots,
        compressed_dequant,
        compressed_slots,
        attn_sink,
        scale,
    )
    out_kernel = q.new_empty(num_tokens, num_heads, HEAD_DIM)
    flash_mla_with_kvcache_cpu(
        out_kernel,
        q,
        window_cache,
        window_slots,
        window_block_size,
        compressed_cache,
        compressed_slots,
        compressed_block_size,
        attn_sink,
        scale,
    )

    torch.testing.assert_close(
        out_kernel.float(), out_ref.float(), rtol=2e-2, atol=2e-2
    )


def test_flash_mla_cpu_kernel_all_invalid_falls_back_to_sink():
    """When every KV slot is masked out, softmax degenerates to the sink
    logit alone and the output must be all-zero (no valid value rows)."""
    torch.manual_seed(2)
    dtype = torch.bfloat16
    num_tokens, num_heads, num_window = 3, 4, 5
    window_block_size = 16

    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype)
    scale = 1.0 / (HEAD_DIM**0.5)
    attn_sink = torch.randn(num_heads, dtype=torch.float32)

    window_kv_f32 = torch.randn(window_block_size, HEAD_DIM)
    window_cache, _ = _pack_flash_mla_cache(window_kv_f32, window_block_size)
    window_slots = torch.full((num_tokens, num_window), -1, dtype=torch.int64)
    compressed_slots = window_slots.new_full((num_tokens, 0), -1)

    out = q.new_empty(num_tokens, num_heads, HEAD_DIM)
    flash_mla_with_kvcache_cpu(
        out,
        q,
        window_cache,
        window_slots,
        window_block_size,
        window_cache,
        compressed_slots,
        1,
        attn_sink,
        scale,
    )
    assert out.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# fused_qnorm_rope_kv_insert_cpu (store_cache.cpp)
# ---------------------------------------------------------------------------


def _qnorm_kv_insert_eager(
    q, kv, positions, cache, slot_mapping, cos_sin_cache, q_head_padded, eps, bs
):
    """Reference: per-head weight-free RMSNorm + GPT-J RoPE on Q; GPT-J RoPE
    + UE8M0 FP8 quant + paged insert on KV."""
    num_quant_blocks = NOPE_DIM // QUANT_BLOCK

    num_tokens_full, num_heads_q, _ = q.shape
    out_dtype = q.dtype

    cos = cos_sin_cache[positions, : ROPE_DIM // 2]
    sin = cos_sin_cache[positions, ROPE_DIM // 2 :]

    q_f32 = q.float()
    var = q_f32.square().mean(dim=-1, keepdim=True)
    q_f32 = q_f32 * torch.rsqrt(var + eps)
    q_f32[..., NOPE_DIM:] = _apply_gptj_rope(
        q_f32[..., NOPE_DIM:], cos.unsqueeze(1), sin.unsqueeze(1)
    )
    q_out = q.new_zeros(num_tokens_full, q_head_padded, HEAD_DIM)
    q_out[:, :num_heads_q, :] = q_f32.to(out_dtype)

    num_tokens_insert = slot_mapping.shape[0]
    kv_f32 = kv[:num_tokens_insert].float()
    kv_f32[..., NOPE_DIM:] = _apply_gptj_rope(
        kv_f32[..., NOPE_DIM:], cos[:num_tokens_insert], sin[:num_tokens_insert]
    )
    kv_bf16 = kv_f32.to(torch.bfloat16)
    kv_f32 = kv_bf16.float()

    valid = slot_mapping >= 0
    if bool(valid.any()):
        idx = valid.nonzero(as_tuple=True)[0]
        slot_id = slot_mapping[idx]
        block_idx = torch.div(slot_id, bs, rounding_mode="floor")
        pos_in_block = slot_id % bs

        nope = kv_f32[idx, :NOPE_DIM].view(-1, num_quant_blocks, QUANT_BLOCK)
        absmax = nope.abs().amax(dim=-1).clamp(min=1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        inv_scale = torch.exp2(-exponent)
        scaled = (nope * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        # Flush fp8-subnormal magnitudes to zero, matching the kernel's
        # AVX512 cvt_fp32x16_to_fp8x16, which doesn't preserve exact
        # subnormal encodings the way .to(torch.float8_e4m3fn) does.
        fp8_min_normal = 2.0**-6
        scaled = torch.where(scaled.abs() < fp8_min_normal, 0.0, scaled)
        nope_fp8_bytes = (
            scaled.to(torch.float8_e4m3fn).view(-1, NOPE_DIM).view(torch.uint8)
        )
        scale_bytes = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
        rope_bf16_bytes = kv_bf16[idx, NOPE_DIM:].contiguous().view(torch.uint8)

        arange_nope = torch.arange(NOPE_DIM)
        arange_rope = torch.arange(ROPE_DIM * 2)
        arange_scale = torch.arange(num_quant_blocks)
        row_idx = block_idx.unsqueeze(-1)
        token_base = pos_in_block.unsqueeze(-1) * TOKEN_DATA_BYTES

        cache[row_idx, token_base + arange_nope] = nope_fp8_bytes
        cache[row_idx, token_base + NOPE_DIM + arange_rope] = rope_bf16_bytes
        scale_base = (
            bs * TOKEN_DATA_BYTES + pos_in_block.unsqueeze(-1) * SCALE_BYTES_PER_TOKEN
        )
        cache[row_idx, scale_base + arange_scale] = scale_bytes
        cache[block_idx, scale_base.squeeze(-1) + num_quant_blocks] = 0

    return q_out


@pytest.mark.parametrize("num_tokens", [1, 64])
@pytest.mark.parametrize("n_heads,padded_heads", [(1, 8), (8, 8), (8, 16), (64, 128)])
def test_qnorm_rope_kv_insert_cpu_kernel_matches_eager(
    num_tokens: int, n_heads: int, padded_heads: int
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    block_size = 16

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, "cpu")

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    if num_tokens > 1:
        slot_mapping[-1] = -1  # exercise the "skip invalid slot" branch

    cache_ref = _make_cache(num_blocks, block_size)
    cache_fused = cache_ref.clone()

    q_ref = _qnorm_kv_insert_eager(
        q,
        kv,
        positions,
        cache_ref,
        slot_mapping,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )
    q_fused = fused_qnorm_rope_kv_insert_cpu(
        q,
        kv,
        positions,
        cache_fused,
        slot_mapping,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )

    torch.testing.assert_close(q_fused.float(), q_ref.float(), rtol=1e-2, atol=1e-2)
    if n_heads < padded_heads:
        pad_region = q_fused[:, n_heads:padded_heads]
        assert pad_region.abs().max().item() == 0.0, (
            "padded head slots must be exact zero"
        )

    _assert_cache_parity(cache_fused, cache_ref, slot_mapping, block_size)


@pytest.mark.parametrize("pad", [1, 5])
def test_qnorm_rope_kv_insert_cpu_kernel_dp_padding(pad: int):
    """slot_mapping shorter than q/kv rows: the KV branch must only touch
    the first num_tokens rows while Q-norm+RoPE still runs on all rows."""
    torch.manual_seed(1)
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    block_size = 16
    num_tokens = 17
    total = num_tokens + pad
    padded_heads = 8

    q = torch.randn(total, padded_heads, HEAD_DIM, dtype=dtype)
    kv = torch.randn(total, HEAD_DIM, dtype=dtype)
    positions = torch.randint(0, max_pos, (total,), dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, "cpu")

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)

    cache_ref = _make_cache(num_blocks, block_size)
    cache_fused = cache_ref.clone()

    q_ref = _qnorm_kv_insert_eager(
        q,
        kv,
        positions,
        cache_ref,
        slot_mapping,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )
    q_fused = fused_qnorm_rope_kv_insert_cpu(
        q,
        kv,
        positions,
        cache_fused,
        slot_mapping,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )

    torch.testing.assert_close(q_fused.float(), q_ref.float(), rtol=1e-2, atol=1e-2)
    _assert_cache_parity(cache_fused, cache_ref, slot_mapping, block_size)


# ---------------------------------------------------------------------------
# inverse_gptj_rope_o_proj_cpu (store_cache.cpp): DeepseekV4CPUAttention's
# _o_proj de-rotation kernel
# ---------------------------------------------------------------------------


def _inverse_gptj_rope_o_proj_eager(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """NoPE dims pass through as fp32; the RoPE segment is de-rotated by
    R(-pos) in GPT-J (interleaved even/odd) convention."""
    nope_dim = o.shape[-1] - rope_dim
    half_dim = rope_dim // 2

    o_f32 = o.float()
    out = torch.empty_like(o_f32)
    out[..., :nope_dim] = o_f32[..., :nope_dim]

    cos = cos_sin_cache[positions, :half_dim].unsqueeze(1)
    sin = cos_sin_cache[positions, half_dim:].unsqueeze(1)

    e = o_f32[..., nope_dim::2]
    od = o_f32[..., nope_dim + 1 :: 2]
    out[..., nope_dim::2] = e * cos + od * sin
    out[..., nope_dim + 1 :: 2] = od * cos - e * sin
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize(
    "head_dim,rope_dim,num_heads",
    [
        (512, 64, 8),  # DeepseekV4CPUAttention._o_proj's actual shape
        (64, 64, 1),  # nope_dim=0: no NoPE region at all
    ],
)
def test_inverse_gptj_rope_o_proj_cpu_kernel_matches_eager(
    dtype: torch.dtype,
    num_tokens: int,
    head_dim: int,
    rope_dim: int,
    num_heads: int,
):
    torch.manual_seed(0)
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(max_pos, rope_dim, torch.float32, "cpu")

    out_ref = _inverse_gptj_rope_o_proj_eager(o, positions, cos_sin_cache, rope_dim)
    out_fused = inverse_gptj_rope_o_proj_cpu(o, positions, cos_sin_cache, rope_dim)

    torch.testing.assert_close(out_fused, out_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# fp8_paged_mqa_logits_cpu (paged_mqa_logits.cpp): sparse indexer's
# decode-path MQA-logits kernel
# ---------------------------------------------------------------------------


def _rand_fp8(*shape: int) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32).clamp(-3, 3).to(torch.float8_e4m3fn)


def _pages_per_request(block_size: int, seq_lens: list[int]) -> list[int]:
    return [-(-seq_len // block_size) for seq_len in seq_lens]


def _write_page(
    page_row: torch.Tensor, block_size: int, k_fp8: torch.Tensor, scale: torch.Tensor
) -> None:
    """Write k_fp8 ([block_size, INDEXER_HEAD_DIM]) and scale ([block_size]
    fp32) into one page row: K-region then scale-region, matching
    compress_norm_rope_store_indexer_cpu's layout."""
    k_region = page_row[: block_size * INDEXER_HEAD_DIM]
    k_region.copy_(k_fp8.view(torch.uint8).flatten())
    scale_region = page_row[block_size * INDEXER_HEAD_DIM :].view(torch.float32)
    scale_region[:].copy_(scale)


def _reference_paged_mqa_logits(
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    ref_k: list[torch.Tensor],
    ref_scale: list[torch.Tensor],
    max_seq_len: int,
) -> torch.Tensor:
    batch_size, _num_heads, _ = q_fp8.shape
    out = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    for b in range(batch_size):
        k = ref_k[b].float()  # [seq_len, head_dim]
        scale = ref_scale[b]  # [seq_len]
        q = q_fp8[b].float()  # [heads, head_dim]
        w = weight[b]  # [heads]
        dot = torch.einsum("jd,hd->jh", k, q).clamp(min=0.0)
        score = (dot * w.unsqueeze(0)).sum(dim=1)
        out[b, : k.shape[0]] = score * scale
    return out


def _build_paged_mqa_logits_batch(
    block_size: int,
    seq_lens: list[int],
    num_heads: int,
    pages: torch.Tensor,
    seed: int,
):
    """Fill pages (a pre-sized [n_pages, buf_width] uint8 tensor, possibly
    non-contiguous in dim 0) with deterministic-per-seed random K/scale
    data."""
    torch.manual_seed(seed)
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    per_request = _pages_per_request(block_size, seq_lens)
    pages_per_batch = max(per_request)
    page_table = torch.full((batch_size, pages_per_batch), -1, dtype=torch.int32)

    ref_k: list[torch.Tensor] = []
    ref_scale: list[torch.Tensor] = []
    next_page = 0
    for b, seq_len in enumerate(seq_lens):
        k_chunks = []
        scale_chunks = []
        for p in range(per_request[b]):
            page_table[b, p] = next_page
            k = _rand_fp8(block_size, INDEXER_HEAD_DIM)
            scale = torch.rand(block_size, dtype=torch.float32) * 0.05 + 0.01
            _write_page(pages[next_page], block_size, k, scale)
            tokens_here = min(block_size, seq_len - p * block_size)
            k_chunks.append(k[:tokens_here])
            scale_chunks.append(scale[:tokens_here])
            next_page += 1
        ref_k.append(torch.cat(k_chunks, dim=0))
        ref_scale.append(torch.cat(scale_chunks, dim=0))

    q_fp8 = _rand_fp8(batch_size, num_heads, INDEXER_HEAD_DIM)
    weight = torch.rand(batch_size, num_heads, dtype=torch.float32) * 0.1
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return q_fp8, weight, seq_lens_t, page_table, ref_k, ref_scale, max_seq_len


def _paged_mqa_logits_page_layout(
    block_size: int, seq_lens: list[int]
) -> tuple[int, int]:
    n_pages = sum(_pages_per_request(block_size, seq_lens))
    buf_width = block_size * INDEXER_HEAD_DIM + block_size * INDEXER_SCALE_BYTES
    return n_pages, buf_width


@pytest.mark.parametrize("block_size", [64, 256])
def test_fp8_paged_mqa_logits_cpu_matches_eager(block_size: int):
    num_heads = 4
    seq_lens = [1, block_size + 7, 3 * block_size]
    n_pages, buf_width = _paged_mqa_logits_page_layout(block_size, seq_lens)
    pages = torch.zeros((n_pages, buf_width), dtype=torch.uint8)

    q_fp8, weight, seq_lens_t, page_table, ref_k, ref_scale, max_seq_len = (
        _build_paged_mqa_logits_batch(block_size, seq_lens, num_heads, pages, seed=0)
    )

    logits = fp8_paged_mqa_logits_cpu(
        q_fp8, pages, weight, seq_lens_t, page_table, block_size, max_seq_len
    )
    ref = _reference_paged_mqa_logits(q_fp8, weight, ref_k, ref_scale, max_seq_len)

    for b, seq_len in enumerate(seq_lens):
        torch.testing.assert_close(
            logits[b, :seq_len], ref[b, :seq_len], atol=1e-2, rtol=1e-2
        )


def test_fp8_paged_mqa_logits_cpu_non_contiguous_kv_cache_row_stride():
    """The paged K-cache in production is a per-layer view into a shared
    multi-layer allocation, so its page stride generally exceeds the page
    byte width -- check the kernel reads a padded/strided buffer identically
    to an equivalent tightly-packed one."""
    block_size = 64
    num_heads = 4
    seq_lens = [1, block_size + 7, 3 * block_size]
    n_pages, buf_width = _paged_mqa_logits_page_layout(block_size, seq_lens)
    cache_pad = 48

    tight_pages = torch.zeros((n_pages, buf_width), dtype=torch.uint8)
    q_fp8, weight, seq_lens_t, page_table, _, _, max_seq_len = (
        _build_paged_mqa_logits_batch(
            block_size, seq_lens, num_heads, tight_pages, seed=0
        )
    )

    padded = torch.zeros((n_pages, buf_width + cache_pad), dtype=torch.uint8)
    strided_pages = padded[:, :buf_width]
    _build_paged_mqa_logits_batch(
        block_size, seq_lens, num_heads, strided_pages, seed=0
    )
    assert strided_pages.stride(0) != strided_pages.size(1)

    logits_tight = fp8_paged_mqa_logits_cpu(
        q_fp8, tight_pages, weight, seq_lens_t, page_table, block_size, max_seq_len
    )
    logits_strided = fp8_paged_mqa_logits_cpu(
        q_fp8, strided_pages, weight, seq_lens_t, page_table, block_size, max_seq_len
    )
    # Only [:seq_len] is ever written (and ever read downstream by
    # topk_transform_512_cpu) -- positions beyond that are uninitialized, so
    # comparing the full row would be comparing unrelated garbage.
    for b, seq_len in enumerate(seq_lens):
        torch.testing.assert_close(
            logits_tight[b, :seq_len], logits_strided[b, :seq_len]
        )


# ---------------------------------------------------------------------------
# fused_indexer_q_rope_quant_cpu (indexer.cpp): indexer Q-side RoPE + FP8
# quant, and the dispatcher wiring that routes to it on CPU
# ---------------------------------------------------------------------------
#
# Independent eager reference: this kernel's absmax reduction is over an
# unusual mix (raw NoPE values plus a bf16-round-tripped RoPE half), which
# per_token_group_quant_fp8 does not replicate. Allow <=1 fp8 ULP: the RoPE
# halves round a value sitting exactly on a tie to either side.

INDEXER_Q_N_HEAD = 64
INDEXER_Q_MAX_POS = 4096


def _indexer_q_rope_quant_reference(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _num_tokens, _num_heads, head_dim = q.shape
    half_rot_dim = cos_sin_cache.shape[-1] // 2
    rot_dim = 2 * half_rot_dim
    nope_dim = head_dim - rot_dim

    q_f32 = q.float()
    cos_sin_f32 = cos_sin_cache.float()
    pos_cos_sin = cos_sin_f32[positions]  # [T, rot_dim]
    cos = pos_cos_sin[:, :half_rot_dim].unsqueeze(1)
    sin = pos_cos_sin[:, half_rot_dim:].unsqueeze(1)

    nope = q_f32[..., :nope_dim]
    rot = q_f32[..., nope_dim:]
    x_even = rot[..., 0::2]
    x_odd = rot[..., 1::2]
    r_even = (x_even * cos - x_odd * sin).to(torch.bfloat16).to(torch.float32)
    r_odd = (x_odd * cos + x_even * sin).to(torch.bfloat16).to(torch.float32)
    rope_out = torch.stack((r_even, r_odd), dim=-1).flatten(-2)

    combined = torch.cat((nope, rope_out), dim=-1)  # [T, H, head_dim]
    amax = combined.abs().amax(dim=-1).clamp(min=1e-4)  # [T, H]
    exponent = torch.ceil(torch.log2(amax / FP8_MAX))
    scale = torch.exp2(exponent)
    inv_scale = scale.reciprocal().unsqueeze(-1)
    q_fp8 = (combined * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    weights_out = weights.to(torch.float32) * scale * softmax_scale * head_scale
    return q_fp8, weights_out


@pytest.mark.parametrize("num_tokens", [1, 257])
def test_fused_indexer_q_rope_quant_cpu_matches_eager(num_tokens: int):
    torch.manual_seed(0)

    q = torch.randn(num_tokens, INDEXER_Q_N_HEAD, INDEXER_HEAD_DIM, dtype=torch.float32)
    positions = torch.randint(0, INDEXER_Q_MAX_POS, (num_tokens,), dtype=torch.int64)
    cos_sin_cache = torch.randn(INDEXER_Q_MAX_POS, ROPE_DIM, dtype=torch.float32)
    weights = torch.randn(num_tokens, INDEXER_Q_N_HEAD, dtype=torch.float32)
    softmax_scale = INDEXER_HEAD_DIM**-0.5
    head_scale = INDEXER_Q_N_HEAD**-0.5

    q_fp8_ref, weights_ref = _indexer_q_rope_quant_reference(
        positions, q, cos_sin_cache, weights, softmax_scale, head_scale
    )

    q_fp8_fused = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights_fused = torch.empty_like(weights, dtype=torch.float32)
    fused_indexer_q_rope_quant_cpu(
        positions,
        q,
        cos_sin_cache,
        q_fp8_fused,
        weights,
        softmax_scale,
        head_scale,
        weights_fused,
    )

    max_ulp = int(fp8_ulp_distance(q_fp8_fused, q_fp8_ref).max().item())
    assert max_ulp <= 1, f"q_fp8 differs by {max_ulp} ULP (>1)"
    torch.testing.assert_close(weights_fused, weights_ref, rtol=1e-5, atol=1e-6)


def test_fused_indexer_q_rope_quant_dispatches_to_cpu_kernel():
    """End-to-end wiring check: the production dispatcher must route to the
    CPU op on a CPU tensor, not silently fall through to the Triton kernel
    (which would also "work" under triton-cpu, masking a dispatch
    regression). The dispatcher does its CPU import inside the CPU branch,
    so the patch target is the source module."""
    torch.manual_seed(0)
    num_tokens = 1

    q = torch.randn(num_tokens, INDEXER_Q_N_HEAD, INDEXER_HEAD_DIM, dtype=torch.float32)
    positions = torch.randint(0, INDEXER_Q_MAX_POS, (num_tokens,), dtype=torch.int64)
    cos_sin_cache = torch.randn(INDEXER_Q_MAX_POS, ROPE_DIM, dtype=torch.float32)
    weights = torch.randn(num_tokens, INDEXER_Q_N_HEAD, dtype=torch.float32)
    softmax_scale = INDEXER_HEAD_DIM**-0.5
    head_scale = INDEXER_Q_N_HEAD**-0.5

    q_fp8_ref, weights_ref = _indexer_q_rope_quant_reference(
        positions, q, cos_sin_cache, weights, softmax_scale, head_scale
    )

    with mock.patch("vllm._custom_ops.fused_indexer_q_rope_quant_cpu") as mocked:
        fused_indexer_q_rope_quant(
            positions,
            q.clone(),
            cos_sin_cache,
            weights,
            softmax_scale,
            head_scale,
            use_fp4=False,
        )
        assert mocked.called, (
            "fused_indexer_q_rope_quant did not dispatch to the CPU kernel "
            "on a CPU tensor"
        )

    q_fp8_fused, weights_fused = fused_indexer_q_rope_quant(
        positions,
        q.clone(),
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        use_fp4=False,
    )
    max_ulp = int(fp8_ulp_distance(q_fp8_fused, q_fp8_ref).max().item())
    assert max_ulp <= 1, f"q_fp8 differs by {max_ulp} ULP (>1)"
    torch.testing.assert_close(weights_fused, weights_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# mhc_pre_cpu / mhc_post_cpu / hc_head_fused_cpu (mhc.cpp): CPU ports of the
# mHC gating kernels
# ---------------------------------------------------------------------------


def _rmsnorm_nw_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weight-free RMSNorm over the last dim, matching hc_head_fused_cpu."""
    x_f32 = x.float()
    var = x_f32.square().mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(var + eps)).to(x.dtype)


def _hc_head_torch(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Eager reference for hc_head_fused_cpu: round-trips the RMSNorm
    output through bf16 before the linear, matching the kernel's bf16
    arithmetic."""
    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.reshape(-1, hc_mult, hidden_size)

    x_flat = hs_flat.flatten(-2)
    x_normed = _rmsnorm_nw_torch(x_flat, rms_norm_eps)
    mixes = torch.nn.functional.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    out = torch.sum(pre.unsqueeze(-1) * hs_flat.float(), dim=1).to(hidden_states.dtype)
    return out.view(*outer_shape, hidden_size)


@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_pre_cpu(num_tokens: int, hidden_size: int):
    """mhc_pre_cpu vs. the mhc_pre_torch eager reference it replaces."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    hc_mult = 4

    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((3,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float32) * 0.1

    rms_eps = hc_eps = 1e-6
    hc_post_mult_value = 2.0
    sinkhorn_repeat = 8

    ref = mhc_kernels.mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    out = mhc_kernels.mhc_pre_cpu(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    for actual, expected in zip(out, ref, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_post_cpu(num_tokens: int, hidden_size: int):
    """mhc_post_cpu vs. the mhc_post_torch eager reference it replaces."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    hc_mult = 4

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    ref = mhc_kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    out = mhc_kernels.mhc_post_cpu(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_hc_head_cpu(num_tokens: int, hidden_size: int):
    """hc_head_fused_cpu vs. the _hc_head_torch eager reference it
    replaces."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    hc_mult = 4

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    ref = _hc_head_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    out = mhc_kernels.hc_head_fused_cpu(
        residual, fn, hc_scale, hc_base, rms_eps, hc_eps
    )

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)
