# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the fused MoE LoRA FP8 kernels with dynamic quantization
split across shrink (absmax) and expand (div + FP8 cast).

Strategy:
  1. Compare the fused-quant op against the *original* (already-validated)
     ``_fused_moe_lora_fp8`` op — they should produce near-identical results
     since the math is equivalent, just reorganised across kernels.
  2. Compare against the PyTorch reference implementation for absolute
     correctness.
  3. Cover the key code paths:
     - block-wise quantization (the fused path)
     - tensor-wise / per-channel quantization (fallback path)
     - non-FP8 (no quantization at all)
     - naive block assignment (small batch)
     - sorted token assignment (large batch)
"""

import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_fused_quant_op import (
    _fused_moe_lora_fp8_fused_quant,
)
from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_op import _fused_moe_lora_fp8
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


# ---------------------------------------------------------------------------
# Helpers (shared with the base test file)
# ---------------------------------------------------------------------------


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences
    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences
    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)
    start = 0
    for seq_idx in range(num_sequences):
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)
        lora_id = random.randint(0, max_loras - 1)
        token_lora_mapping[start:end] = lora_id
        start = end
    return token_lora_mapping


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int):
    assert top_k_num <= num_experts
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected
    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
    return expert_indices, expert_weights


def sample_data(num_tokens, num_sequences, max_loras, num_experts, top_k_num):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num
    )
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    active_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    lora_ids = torch.unique(token_lora_mapping, sorted=True)
    active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
    return topk_ids, topk_weights, token_lora_mapping, active_lora_ids


def quantize_to_fp8(tensor, per_channel=False, block_shape=None):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if block_shape is not None:
        block_n, block_k = block_shape
        orig_shape = tensor.shape
        dim0, dim1 = orig_shape[-2], orig_shape[-1]
        n_blocks = CEILDIV(dim0, block_n)
        k_blocks = CEILDIV(dim1, block_k)
        padded_dim0 = n_blocks * block_n
        padded_dim1 = k_blocks * block_k
        if padded_dim0 != dim0 or padded_dim1 != dim1:
            padded = torch.zeros(
                *orig_shape[:-2],
                padded_dim0,
                padded_dim1,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded[..., :dim0, :dim1] = tensor
        else:
            padded = tensor
        reshaped = padded.reshape(
            *orig_shape[:-2], n_blocks, block_n, k_blocks, block_k
        )
        reshaped = reshaped.permute(*range(len(orig_shape) - 2), -4, -2, -3, -1)
        amax = reshaped.abs().amax(dim=(-2, -1)).float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        reshaped_float = reshaped.float()
        quantized_blocks = (reshaped_float / scale_expanded).clamp(-fp8_max, fp8_max)
        quantized_blocks = quantized_blocks.permute(
            *range(len(orig_shape) - 2), -4, -2, -3, -1
        )
        quantized = quantized_blocks.reshape(*orig_shape[:-2], padded_dim0, padded_dim1)
        quantized = quantized[..., :dim0, :dim1]
        return quantized.to(torch.float8_e4m3fn), scale
    elif per_channel:
        amax = tensor.abs().amax(dim=-1, keepdim=True).float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
        quantized = (tensor.float() / scale).clamp(-fp8_max, fp8_max)
        return quantized.to(torch.float8_e4m3fn), scale.squeeze(-1)
    else:
        amax = tensor.abs().amax().float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
        quantized = (tensor.float() / scale).clamp(-fp8_max, fp8_max)
        return quantized.to(torch.float8_e4m3fn), scale


def _dequant_fp8_weights(weight_stacked, scale_stacked, per_channel_quant, block_shape):
    """Dequantize a list of FP8 weight tensors using the matching scales."""
    result = []
    for idx, w in enumerate(weight_stacked):
        w_float = w.float()
        if scale_stacked is not None:
            scale = scale_stacked[idx]
            if block_shape is not None:
                block_n, block_k = block_shape
                orig_shape = w_float.shape
                dim0, dim1 = orig_shape[-2], orig_shape[-1]
                n_blocks = CEILDIV(dim0, block_n)
                k_blocks = CEILDIV(dim1, block_k)
                padded_dim0 = n_blocks * block_n
                padded_dim1 = k_blocks * block_k
                padded = torch.zeros(
                    *orig_shape[:-2],
                    padded_dim0,
                    padded_dim1,
                    dtype=w_float.dtype,
                    device=w_float.device,
                )
                padded[..., :dim0, :dim1] = w_float
                reshaped = padded.reshape(
                    *orig_shape[:-2], n_blocks, block_n, k_blocks, block_k
                )
                reshaped = reshaped.permute(*range(len(orig_shape) - 2), -4, -2, -3, -1)
                scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
                dequantized = reshaped * scale_expanded
                dequantized = dequantized.permute(
                    *range(len(orig_shape) - 2), -4, -2, -3, -1
                )
                dequantized = dequantized.reshape(
                    *orig_shape[:-2], padded_dim0, padded_dim1
                )
                w_float = dequantized[..., :dim0, :dim1]
            elif per_channel_quant:
                if scale.ndim == w_float.ndim:
                    scale = scale.squeeze(-1)
                w_float = w_float * scale.unsqueeze(-1)
            else:
                w_float = w_float * scale.unsqueeze(-1).unsqueeze(-1)
        result.append(w_float)
    return result


def use_torch_ref(
    hidden_states,
    token_lora_mapping,
    topk_ids,
    lora_a_stacked,
    lora_b_stacked,
    top_k_num,
    lora_a_scale_stacked=None,
    lora_b_scale_stacked=None,
    act_scale=None,
    use_fp8_w8a8=False,
    per_channel_quant=False,
    block_shape=None,
):
    """PyTorch reference: dequantize everything and compute in float."""
    if use_fp8_w8a8:
        lora_a_list = _dequant_fp8_weights(
            lora_a_stacked, lora_a_scale_stacked, per_channel_quant, block_shape
        )
        lora_b_list = _dequant_fp8_weights(
            lora_b_stacked, lora_b_scale_stacked, per_channel_quant, block_shape
        )
    else:
        lora_a_list = [la.float() for la in lora_a_stacked]
        lora_b_list = [lb.float() for lb in lora_b_stacked]

    hidden_float = hidden_states.float()
    if act_scale is not None:
        hidden_float = hidden_float * act_scale.float()

    outputs = []
    for i in range(hidden_float.shape[0]):
        lora_idx = token_lora_mapping[i]
        expert_ids = topk_ids[i]
        lora_a = lora_a_list[0][lora_idx][expert_ids]
        lora_b = lora_b_list[0][lora_idx][expert_ids]
        tensors = [
            hidden_float[i] @ lora_a[x].T @ lora_b[x].T for x in range(top_k_num)
        ]
        outputs.append(torch.stack(tensors, dim=0))
    return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Kernel invocation helpers
# ---------------------------------------------------------------------------


def _prepare_sorted_tokens(
    topk_ids, token_lora_mapping, num_experts, block_size, max_loras, lora_ids
):
    """Run moe_lora_align_block_size and return sorted metadata."""
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.empty(
        (max_loras * max_num_tokens_padded,),
        dtype=torch.int32,
    )
    expert_ids = torch.empty(
        (max_loras * max_num_m_blocks,),
        dtype=torch.int32,
    )
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        adapter_enabled,
        lora_ids,
    )
    return (
        sorted_token_ids.view(max_loras, -1),
        expert_ids.view(max_loras, -1),
        num_tokens_post_padded,
        adapter_enabled,
    )


def _make_config():
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }


def _call_fused_quant_op(
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    num_experts,
    block_size,
    lora_a_scale_stacked=None,
    lora_b_scale_stacked=None,
    shrink_act_scale=None,
    expand_act_scale=None,
    use_fp8_w8a8=False,
    per_channel_quant=False,
    block_shape=None,
    naive=False,
):
    """Invoke _fused_moe_lora_fp8_fused_quant with sorted or naive path."""
    config = _make_config()

    if naive:
        sorted_token_ids = None
        expert_ids = topk_ids.reshape(-1)
        num_tokens_post_padded = None
        adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded, adapter_enabled = (
            _prepare_sorted_tokens(
                topk_ids,
                token_lora_mapping,
                num_experts,
                block_size,
                max_loras,
                lora_ids,
            )
        )

    num_active_loras = max_loras + 1
    _a_scale = lora_a_scale_stacked if lora_a_scale_stacked is not None else []
    _b_scale = lora_b_scale_stacked if lora_b_scale_stacked is not None else []

    _fused_moe_lora_fp8_fused_quant(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        _a_scale,
        _b_scale,
        shrink_act_scale=shrink_act_scale,
        expand_act_scale=expand_act_scale,
        mul_routed_weight=False,
        fully_sharded=False,
        offset=0,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )


def _call_original_op(
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    num_experts,
    block_size,
    lora_a_scale_stacked=None,
    lora_b_scale_stacked=None,
    shrink_act_scale=None,
    expand_act_scale=None,
    use_fp8_w8a8=False,
    per_channel_quant=False,
    block_shape=None,
    naive=False,
):
    """Invoke the original _fused_moe_lora_fp8 for comparison."""
    config = _make_config()

    if naive:
        sorted_token_ids = None
        expert_ids = topk_ids.reshape(-1)
        num_tokens_post_padded = None
        adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded, adapter_enabled = (
            _prepare_sorted_tokens(
                topk_ids,
                token_lora_mapping,
                num_experts,
                block_size,
                max_loras,
                lora_ids,
            )
        )

    num_active_loras = max_loras + 1
    _a_scale = lora_a_scale_stacked if lora_a_scale_stacked is not None else []
    _b_scale = lora_b_scale_stacked if lora_b_scale_stacked is not None else []

    _fused_moe_lora_fp8(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        _a_scale,
        _b_scale,
        shrink_act_scale=shrink_act_scale,
        expand_act_scale=expand_act_scale,
        mul_routed_weight=False,
        fully_sharded=False,
        offset=0,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:{0}"]
SEED = [42]


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Non-FP8 baseline (no quantization — fused op should behave
#          identically to the original since no quant fusion kicks in)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_no_fp8(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    """Non-FP8: fused-quant op vs torch reference."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output_fused,
        max_loras,
        num_experts,
        block_size,
    )

    output_ref = use_torch_ref(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=1e-2, rtol=1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: FP8 block-wise quantization — the FUSED path
#          This is the main test: absmax fused in shrink, div+cast in expand.
#          Compare against original op AND torch reference.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_block_wise_vs_original(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    block_shape,
    dtype,
    device,
    seed,
):
    """Block-wise FP8: fused-quant op vs original op (both should match)."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, block_shape=block_shape)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, block_shape=block_shape)

    lora_a_scale_stacked = [lora_a_scale.float()]
    lora_b_scale_stacked = [lora_b_scale.float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp)
    k_blocks = CEILDIV(K, block_shape[1])
    act_scale = torch.full(
        (num_tokens, k_blocks), act_scale_raw.item(), dtype=torch.float32
    )

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    # --- fused-quant op ---
    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    # --- original op ---
    output_orig = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_original_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_orig,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    # The fused path does the quant differently (absmax in shrink, div in
    # expand) so there will be small numerical differences due to the
    # intermediate representation.  Use a slightly wider tolerance.
    torch.testing.assert_close(output_fused, output_orig, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_block_wise_vs_torch(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    block_shape,
    dtype,
    device,
    seed,
):
    """Block-wise FP8: fused-quant op vs PyTorch reference."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, block_shape=block_shape)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, block_shape=block_shape)

    lora_a_scale_stacked = [lora_a_scale.float()]
    lora_b_scale_stacked = [lora_b_scale.float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp)
    k_blocks = CEILDIV(K, block_shape[1])
    act_scale = torch.full(
        (num_tokens, k_blocks), act_scale_raw.item(), dtype=torch.float32
    )

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    output_ref = use_torch_ref(
        hidden_fp8,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        act_scale=act_scale_raw,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: FP8 tensor-wise quantization (fallback path — no fusion)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_tensor_wise_fallback(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    """Tensor-wise FP8: fused-quant op falls back to original path."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp)

    lora_a_scale_stacked = [
        lora_a_scale.expand(max_loras, num_experts).contiguous().float()
    ]
    lora_b_scale_stacked = [
        lora_b_scale.expand(max_loras, num_experts).contiguous().float()
    ]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_scalar = quantize_to_fp8(hidden_states_fp)
    act_scale = act_scale_scalar.expand(num_tokens, 1).contiguous().float()

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
    )

    output_ref = use_torch_ref(
        hidden_fp8,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        act_scale=act_scale,
        use_fp8_w8a8=True,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: FP8 per-channel quantization (fallback path — no fusion)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_per_channel_fallback(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    """Per-channel FP8: fused-quant op falls back to original path."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, per_channel=True)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, per_channel=True)

    lora_a_scale_stacked = [lora_a_scale.unsqueeze(3).float()]
    lora_b_scale_stacked = [lora_b_scale.unsqueeze(3).float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp, per_channel=True)
    act_scale = act_scale_raw.unsqueeze(-1).float()

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        per_channel_quant=True,
    )

    output_ref = use_torch_ref(
        hidden_fp8,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        act_scale=act_scale,
        use_fp8_w8a8=True,
        per_channel_quant=True,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Naive block assignment — non-FP8
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [1, 4, 8])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_naive_no_fp8(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    """Naive block assignment, non-FP8: fused-quant op vs torch ref."""
    torch.set_default_device(device)
    set_random_seed(seed)

    SPARSITY_FACTOR = 8
    assert num_tokens * top_k_num * SPARSITY_FACTOR <= num_experts * max_loras

    num_sequences = min(num_tokens, 4)
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        naive=True,
    )

    output_ref = use_torch_ref(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=1e-2, rtol=1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Naive block assignment — FP8 block-wise (fused path)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_naive_block_wise(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    block_shape,
    dtype,
    device,
    seed,
):
    """Naive block assignment + block-wise FP8: fused-quant vs original."""
    torch.set_default_device(device)
    set_random_seed(seed)

    SPARSITY_FACTOR = 8
    assert num_tokens * top_k_num * SPARSITY_FACTOR <= num_experts * max_loras

    num_sequences = min(num_tokens, 4)
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, block_shape=block_shape)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, block_shape=block_shape)

    lora_a_scale_stacked = [lora_a_scale.float()]
    lora_b_scale_stacked = [lora_b_scale.float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp)
    k_blocks = CEILDIV(K, block_shape[1])
    act_scale = torch.full(
        (num_tokens, k_blocks), act_scale_raw.item(), dtype=torch.float32
    )

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    # --- fused-quant op (naive) ---
    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
        naive=True,
    )

    # --- original op (naive) ---
    output_orig = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_original_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_orig,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
        naive=True,
    )

    torch.testing.assert_close(output_fused, output_orig, atol=5e-2, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Naive block assignment — FP8 tensor-wise (fallback)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_naive_tensor_wise(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    """Naive block assignment + tensor-wise FP8: fused-quant vs torch ref."""
    torch.set_default_device(device)
    set_random_seed(seed)

    SPARSITY_FACTOR = 8
    assert num_tokens * top_k_num * SPARSITY_FACTOR <= num_experts * max_loras

    num_sequences = min(num_tokens, 4)
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp)

    lora_a_scale_stacked = [
        lora_a_scale.expand(max_loras, num_experts).contiguous().float()
    ]
    lora_b_scale_stacked = [
        lora_b_scale.expand(max_loras, num_experts).contiguous().float()
    ]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_scalar = quantize_to_fp8(hidden_states_fp)
    act_scale = act_scale_scalar.expand(num_tokens, 1).contiguous().float()

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        naive=True,
    )

    output_ref = use_torch_ref(
        hidden_fp8,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        act_scale=act_scale,
        use_fp8_w8a8=True,
    )

    torch.testing.assert_close(output_fused, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Block-wise FP8 with different lora ranks (64, 128)
#          to exercise the case where rank == or > BLOCK_SIZE_N
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [64, 128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_quant_block_wise_large_rank(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    block_shape,
    dtype,
    device,
    seed,
):
    """Block-wise FP8 with larger lora ranks: fused-quant vs original."""
    torch.set_default_device(device)
    set_random_seed(seed)

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 10, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, block_shape=block_shape)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, block_shape=block_shape)

    lora_a_scale_stacked = [lora_a_scale.float()]
    lora_b_scale_stacked = [lora_b_scale.float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp)
    k_blocks = CEILDIV(K, block_shape[1])
    act_scale = torch.full(
        (num_tokens, k_blocks), act_scale_raw.item(), dtype=torch.float32
    )

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output_fused = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_fused_quant_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_fused,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    output_orig = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    _call_original_op(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output_orig,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        block_shape=block_shape,
    )

    torch.testing.assert_close(output_fused, output_orig, atol=5e-2, rtol=5e-2)
