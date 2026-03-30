"""
Register pure-Python fallback XPU implementations for all vLLM _C ops.

This module is imported by vllm/platforms/xpu.py:
    import vllm_xpu_kernels._C

It uses torch.library to:
1. Define op schemas (normally done by compiled csrc/torch_bindings.cpp)
2. Register Python XPU implementations for each op

Covers: activation ops, layernorm, rotary embedding, weak_ref_tensor.
"""
import math

import torch
from torch.library import Library

# ============================================================================
# Namespace: _C  (activation, normalization, rotary, misc)
# ============================================================================
# First, define the op schemas (same as csrc/torch_bindings.cpp)
# Then register XPU implementations.

_C_lib_def = Library("_C", "DEF")
_C_lib_impl = Library("_C", "IMPL")

# --- Activation ops ---

_C_lib_def.define("silu_and_mul(Tensor! result, Tensor input) -> ()")
_C_lib_def.define("mul_and_silu(Tensor! out, Tensor input) -> ()")
_C_lib_def.define("gelu_and_mul(Tensor! out, Tensor input) -> ()")
_C_lib_def.define("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()")
_C_lib_def.define("gelu_new(Tensor! out, Tensor input) -> ()")
_C_lib_def.define("gelu_fast(Tensor! out, Tensor input) -> ()")
_C_lib_def.define("gelu_quick(Tensor! out, Tensor input) -> ()")
_C_lib_def.define(
    "fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()"
)


def _silu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def _mul_and_silu(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(x[..., :d] * torch.nn.functional.silu(x[..., d:]))


def _gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.gelu(x[..., :d]) * x[..., d:])


def _gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(
        torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    )


def _gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = math.sqrt(2.0 / math.pi)
    out.copy_(0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3)))))


def _gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    # FastGELU approximation
    out.copy_(
        0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    )


def _gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    # QuickGELU: x * sigmoid(1.702 * x)
    out.copy_(x * torch.sigmoid(1.702 * x))


def _fatrelu_and_mul(
    out: torch.Tensor, x: torch.Tensor, threshold: float
) -> None:
    d = x.shape[-1] // 2
    relu_part = torch.where(
        x[..., :d] > threshold, x[..., :d], torch.zeros_like(x[..., :d])
    )
    out.copy_(relu_part * x[..., d:])


_C_lib_impl.impl("silu_and_mul", _silu_and_mul, "XPU")
_C_lib_impl.impl("mul_and_silu", _mul_and_silu, "XPU")
_C_lib_impl.impl("gelu_and_mul", _gelu_and_mul, "XPU")
_C_lib_impl.impl("gelu_tanh_and_mul", _gelu_tanh_and_mul, "XPU")
_C_lib_impl.impl("gelu_new", _gelu_new, "XPU")
_C_lib_impl.impl("gelu_fast", _gelu_fast, "XPU")
_C_lib_impl.impl("gelu_quick", _gelu_quick, "XPU")
_C_lib_impl.impl("fatrelu_and_mul", _fatrelu_and_mul, "XPU")

# --- LayerNorm ops ---

_C_lib_def.define(
    "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()"
)
_C_lib_def.define(
    "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
    "float epsilon) -> ()"
)


def _rms_norm(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normed = input * torch.rsqrt(variance + epsilon)
    result.copy_(normed * weight)


def _fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    # In-place: residual += input, then rms_norm(input, residual, ...)
    residual.add_(input)
    variance = residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normed = residual * torch.rsqrt(variance + epsilon)
    input.copy_(normed * weight)


_C_lib_impl.impl("rms_norm", _rms_norm, "XPU")
_C_lib_impl.impl("fused_add_rms_norm", _fused_add_rms_norm, "XPU")

# --- Rotary embedding ---

_C_lib_def.define(
    "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
    "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()"
)


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Pure-Python rotary embedding (NeoX + GPT-J styles)."""
    # cos_sin_cache shape: [max_position, rot_dim]
    # where rot_dim = head_size (for full rotation)
    rot_dim = cos_sin_cache.shape[1]
    # Gather cos/sin for current positions
    cos_sin = cos_sin_cache[positions.long()]  # [num_tokens, rot_dim]
    cos = cos_sin[:, : rot_dim // 2]  # [num_tokens, rot_dim/2]
    sin = cos_sin[:, rot_dim // 2 :]  # [num_tokens, rot_dim/2]

    def _apply_rotary(t: torch.Tensor) -> None:
        """Apply rotary embedding in-place to tensor of shape [num_tokens, num_heads * head_size]."""
        num_tokens = t.shape[0]
        t_view = t.view(num_tokens, -1, head_size)
        num_heads = t_view.shape[1]
        rot = t_view[:, :, :rot_dim]  # [tokens, heads, rot_dim]

        if is_neox:
            # NeoX style: split first/second half
            half = rot_dim // 2
            x1 = rot[:, :, :half].clone()
            x2 = rot[:, :, half:].clone()
            cos_b = cos[:, None, :]  # [tokens, 1, rot_dim/2]
            sin_b = sin[:, None, :]
            rot[:, :, :half] = x1 * cos_b - x2 * sin_b
            rot[:, :, half:] = x2 * cos_b + x1 * sin_b
        else:
            # GPT-J style: interleaved pairs
            half = rot_dim // 2
            cos_b = cos[:, None, :]
            sin_b = sin[:, None, :]
            x1 = rot[:, :, 0::2].clone()
            x2 = rot[:, :, 1::2].clone()
            rot[:, :, 0::2] = x1 * cos_b - x2 * sin_b
            rot[:, :, 1::2] = x2 * cos_b + x1 * sin_b

    _apply_rotary(query)
    if key is not None:
        _apply_rotary(key)


_C_lib_impl.impl("rotary_embedding", _rotary_embedding, "XPU")

# --- Misc ops ---

_C_lib_def.define("weak_ref_tensor(Tensor input) -> Tensor")


def _weak_ref_tensor(input: torch.Tensor) -> torch.Tensor:
    return input  # identity / alias on simulator


_C_lib_impl.impl("weak_ref_tensor", _weak_ref_tensor, "XPU")

# --- Sampling ops (apply_repetition_penalties_) ---

_C_lib_def.define(
    "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
    "Tensor output_mask, Tensor repetition_penalties) -> ()"
)


def _apply_repetition_penalties_(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    # Combined mask of tokens that appeared
    mask = prompt_mask | output_mask  # [batch, vocab]
    penalties = repetition_penalties.unsqueeze(1)  # [batch, 1]
    # Penalize: positive logits are divided by penalty, negative are multiplied
    logits_pos = torch.where(
        mask & (logits > 0), logits / penalties, logits
    )
    logits.copy_(
        torch.where(mask & (logits_pos <= 0), logits * penalties, logits_pos)
    )


_C_lib_impl.impl("apply_repetition_penalties_", _apply_repetition_penalties_, "XPU")


# ============================================================================
# Namespace: _C_cache_ops  (KV cache operations)
# ============================================================================

_cache_lib_def = Library("_C_cache_ops", "DEF")
_cache_lib_impl = Library("_C_cache_ops", "IMPL")

_cache_lib_def.define(
    "reshape_and_cache_flash(Tensor key, Tensor value, "
    "Tensor! key_cache, Tensor! value_cache, "
    "Tensor slot_mapping, str kv_cache_dtype, "
    "Tensor k_scale, Tensor v_scale) -> ()"
)

_cache_lib_def.define(
    "swap_blocks(Tensor src, Tensor! dst, "
    "int block_size_in_bytes, Tensor block_mapping) -> ()"
)


def _reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Scatter K/V into paged cache at slot positions."""
    num_blocks, block_size, num_heads, head_dim = key_cache.shape
    flat_key = key.reshape(-1, num_heads, head_dim)
    flat_val = value.reshape(-1, num_heads, head_dim)
    flat_slots = slot_mapping.reshape(-1)
    for i in range(flat_slots.numel()):
        slot = flat_slots[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        offset = slot % block_size
        key_cache[block_idx, offset, :, :] = flat_key[i]
        value_cache[block_idx, offset, :, :] = flat_val[i]


def _swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None:
    """Copy blocks from src to dst according to block_mapping."""
    for i in range(block_mapping.shape[0]):
        src_idx = block_mapping[i, 0].item()
        dst_idx = block_mapping[i, 1].item()
        dst[dst_idx].copy_(src[src_idx])


_cache_lib_impl.impl("reshape_and_cache_flash", _reshape_and_cache_flash, "XPU")
_cache_lib_impl.impl("swap_blocks", _swap_blocks, "XPU")

print("[vllm_xpu_kernels._C] Registered Python XPU fallbacks for all _C and _C_cache_ops")
