# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm import _custom_ops as ops
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.lora.ops.triton_ops import fused_moe_lora_fp8
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    """
    Split `num_tokens` into `num_sequences` sequences.
    Each sequence randomly selects 1 LoRA index from [0, max_loras),
    and all tokens in that sequence are assigned this LoRA index.

    Args:
        num_tokens (int): Total number of tokens.
        num_sequences (int): Number of sequences to split the tokens into.
        max_loras (int): Total number of available LoRA modules.

    Returns:
        torch.Tensor: 1D tensor of shape [num_tokens], where each value
                      is the LoRA index assigned to that token.
    """
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences, "num_tokens must be >= num_sequences"

    # Compute token distribution per sequence (distribute remainder evenly)
    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)

    start = 0
    for seq_idx in range(num_sequences):
        # Determine the token range for this sequence
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)

        # Randomly select one LoRA ID for this sequence
        lora_id = random.randint(0, max_loras - 1)

        # Assign the same LoRA ID to all tokens in this sequence
        token_lora_mapping[start:end] = lora_id

        start = end

    return token_lora_mapping


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int):
    """
    For each token, randomly select `top_k_num` distinct experts out of
    `num_experts`, and assign normalized random weights that sum to 1.
    """
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(
    num_tokens: int,
    num_sequences: int,
    max_loras: int,
    num_experts: int,
    top_k_num: int,
):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num
    )
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    active_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    lora_ids = torch.unique(token_lora_mapping, sorted=True)
    active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
    return topk_ids, topk_weights, token_lora_mapping, active_lora_ids


def quantize_to_fp8(
    tensor: torch.Tensor,
    per_channel: bool = False,
    block_shape: list[int] | None = None,
):
    """
    Quantize a tensor to FP8 (e4m3) and return the quantized tensor + scale.

    Supports tensor-wise, per-channel, and block-wise quantization.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    if block_shape is not None:
        # Block-wise quantization
        # tensor shape: (max_loras, num_experts, dim0, dim1)
        block_n, block_k = block_shape
        orig_shape = tensor.shape
        dim0, dim1 = orig_shape[-2], orig_shape[-1]
        n_blocks = CEILDIV(dim0, block_n)
        k_blocks = CEILDIV(dim1, block_k)

        # Pad if needed
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

        # Reshape to blocks
        reshaped = padded.reshape(
            *orig_shape[:-2], n_blocks, block_n, k_blocks, block_k
        )
        reshaped = reshaped.permute(
            *range(len(orig_shape) - 2), -4, -2, -3, -1
        )  # (..., n_blocks, k_blocks, block_n, block_k)

        amax = reshaped.abs().amax(dim=(-2, -1)).float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)

        # Scale and quantize
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        reshaped_float = reshaped.float()
        quantized_blocks = (reshaped_float / scale_expanded).clamp(-fp8_max, fp8_max)

        # Reshape back
        quantized_blocks = quantized_blocks.permute(
            *range(len(orig_shape) - 2), -4, -2, -3, -1
        )
        quantized = quantized_blocks.reshape(*orig_shape[:-2], padded_dim0, padded_dim1)
        quantized = quantized[..., :dim0, :dim1]

        return quantized.to(torch.float8_e4m3fn), scale

    elif per_channel:
        # Per-channel: scale per last dim
        amax = tensor.abs().amax(dim=-1, keepdim=True).float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
        quantized = (tensor.float() / scale).clamp(-fp8_max, fp8_max)
        return quantized.to(torch.float8_e4m3fn), scale.squeeze(-1)

    else:
        # Tensor-wise
        amax = tensor.abs().amax().float()
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
        quantized = (tensor.float() / scale).clamp(-fp8_max, fp8_max)
        return quantized.to(torch.float8_e4m3fn), scale


def use_fused_moe_lora_fp8_kernel(
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
    fully_sharded=False,
    offset=0,
):
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.empty(
        (max_loras * max_num_tokens_padded,),
        dtype=torch.int32,
    )
    expert_ids = torch.empty((max_loras * max_num_m_blocks,), dtype=torch.int32)
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)

    # call kernel
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

    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }

    mul_routed_weight = False
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1)

    num_active_loras = max_loras + 1

    # The custom op requires List[Tensor], not None
    _a_scale = lora_a_scale_stacked if lora_a_scale_stacked is not None else []
    _b_scale = lora_b_scale_stacked if lora_b_scale_stacked is not None else []

    fused_moe_lora_fp8(
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
        mul_routed_weight=mul_routed_weight,
        fully_sharded=fully_sharded,
        offset=offset,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )


def use_torch_fp8(
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
    """
    PyTorch reference implementation for FP8 fused MoE LoRA.
    Dequantizes FP8 weights before computing to get reference output.
    """
    # Dequantize lora_a and lora_b if FP8
    if use_fp8_w8a8:
        lora_a_list = []
        for idx, la in enumerate(lora_a_stacked):
            la_float = la.float()
            if lora_a_scale_stacked is not None:
                la_scale = lora_a_scale_stacked[idx]
                if block_shape is not None:
                    # Block-wise dequant for lora_a
                    block_n, block_k = block_shape
                    orig_shape = la_float.shape
                    dim0, dim1 = orig_shape[-2], orig_shape[-1]
                    n_blocks = CEILDIV(dim0, block_n)
                    k_blocks = CEILDIV(dim1, block_k)
                    padded_dim0 = n_blocks * block_n
                    padded_dim1 = k_blocks * block_k
                    padded = torch.zeros(
                        *orig_shape[:-2],
                        padded_dim0,
                        padded_dim1,
                        dtype=la_float.dtype,
                        device=la_float.device,
                    )
                    padded[..., :dim0, :dim1] = la_float
                    reshaped = padded.reshape(
                        *orig_shape[:-2], n_blocks, block_n, k_blocks, block_k
                    )
                    reshaped = reshaped.permute(
                        *range(len(orig_shape) - 2), -4, -2, -3, -1
                    )
                    scale_expanded = la_scale.unsqueeze(-1).unsqueeze(-1)
                    dequantized = reshaped * scale_expanded
                    dequantized = dequantized.permute(
                        *range(len(orig_shape) - 2), -4, -2, -3, -1
                    )
                    dequantized = dequantized.reshape(
                        *orig_shape[:-2], padded_dim0, padded_dim1
                    )
                    la_float = dequantized[..., :dim0, :dim1]
                elif per_channel_quant:
                    # Per-channel: scale shape (..., dim0, 1) from unsqueeze(3)
                    if la_scale.ndim == la_float.ndim:
                        la_scale = la_scale.squeeze(-1)
                    la_float = la_float * la_scale.unsqueeze(-1)
                else:
                    # Tensor-wise: scale shape (max_loras, num_experts)
                    # needs to broadcast to (max_loras, num_experts, rank, K)
                    la_float = la_float * la_scale.unsqueeze(-1).unsqueeze(-1)
            lora_a_list.append(la_float)

        lora_b_list = []
        for idx, lb in enumerate(lora_b_stacked):
            lb_float = lb.float()
            if lora_b_scale_stacked is not None:
                lb_scale = lora_b_scale_stacked[idx]
                if block_shape is not None:
                    block_n, block_k = block_shape
                    orig_shape = lb_float.shape
                    dim0, dim1 = orig_shape[-2], orig_shape[-1]
                    n_blocks = CEILDIV(dim0, block_n)
                    k_blocks = CEILDIV(dim1, block_k)
                    padded_dim0 = n_blocks * block_n
                    padded_dim1 = k_blocks * block_k
                    padded = torch.zeros(
                        *orig_shape[:-2],
                        padded_dim0,
                        padded_dim1,
                        dtype=lb_float.dtype,
                        device=lb_float.device,
                    )
                    padded[..., :dim0, :dim1] = lb_float
                    reshaped = padded.reshape(
                        *orig_shape[:-2], n_blocks, block_n, k_blocks, block_k
                    )
                    reshaped = reshaped.permute(
                        *range(len(orig_shape) - 2), -4, -2, -3, -1
                    )
                    scale_expanded = lb_scale.unsqueeze(-1).unsqueeze(-1)
                    dequantized = reshaped * scale_expanded
                    dequantized = dequantized.permute(
                        *range(len(orig_shape) - 2), -4, -2, -3, -1
                    )
                    dequantized = dequantized.reshape(
                        *orig_shape[:-2], padded_dim0, padded_dim1
                    )
                    lb_float = dequantized[..., :dim0, :dim1]
                elif per_channel_quant:
                    # Per-channel: scale shape (..., dim0, 1) from unsqueeze(3)
                    if lb_scale.ndim == lb_float.ndim:
                        lb_scale = lb_scale.squeeze(-1)
                    lb_float = lb_float * lb_scale.unsqueeze(-1)
                else:
                    # Tensor-wise: scale shape (max_loras, num_experts)
                    lb_float = lb_float * lb_scale.unsqueeze(-1).unsqueeze(-1)
            lora_b_list.append(lb_float)
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


DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:{0}"]
SEED = [42]


# ─── Non-FP8 baseline (same logic, just through the FP8 op with fp8 disabled) ──
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6, 12])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 16])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_fp8_kernel_no_quant(
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
    """Test the FP8 kernel path with use_fp8_w8a8=False (non-quantized)."""
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = 10

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

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel(
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
        use_fp8_w8a8=False,
    )

    output_ref = use_torch_fp8(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
    )

    torch.testing.assert_close(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)


# ─── FP8 tensor-wise quantization ───────────────────────────────────────────
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_fp8_kernel_tensor_wise(
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
    """Test FP8 kernel with tensor-wise quantization."""
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = 10

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    # Create full-precision weights, then quantize to FP8
    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp)

    # Scale is a scalar for tensor-wise; wrap in per-expert shape for the kernel
    # The kernel expects scale indexed by [lora_id, expert_id]
    lora_a_scale_stacked = [
        lora_a_scale.expand(max_loras, num_experts).contiguous().float()
    ]
    lora_b_scale_stacked = [
        lora_b_scale.expand(max_loras, num_experts).contiguous().float()
    ]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)

    # Quantize activations to FP8 (kernel expects FP8 inputs for dot product)
    hidden_fp8, act_scale_scalar = quantize_to_fp8(hidden_states_fp)
    # Per-token activation scale for the kernel
    act_scale = act_scale_scalar.expand(num_tokens, 1).contiguous().float()

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        expand_act_scale=None,
        use_fp8_w8a8=True,
        per_channel_quant=False,
        block_shape=None,
    )

    # Reference: dequantize everything and compute in float
    output_ref = use_torch_fp8(
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
        per_channel_quant=False,
        block_shape=None,
    )

    # FP8 has lower precision, so use wider tolerance
    torch.testing.assert_close(output, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ─── FP8 per-channel quantization ───────────────────────────────────────────
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_fp8_kernel_per_channel(
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
    """Test FP8 kernel with per-channel quantization."""
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = 10

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_fp = torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    lora_b_fp = torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)

    lora_a_fp8, lora_a_scale = quantize_to_fp8(lora_a_fp, per_channel=True)
    lora_b_fp8, lora_b_scale = quantize_to_fp8(lora_b_fp, per_channel=True)

    # Kernel expects 4D scale: (max_loras, num_experts, output_channels, 1)
    # so stride(2) = 1 (stride_bsn) and stride(3) = 1 (stride_bsk)
    lora_a_scale_stacked = [lora_a_scale.unsqueeze(3).float()]
    lora_b_scale_stacked = [lora_b_scale.unsqueeze(3).float()]

    hidden_states_fp = torch.rand((num_tokens, K), dtype=dtype)

    # Quantize activations to FP8 with per-token scale
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp, per_channel=True)
    # act_scale_raw shape: (num_tokens,) — reshape to (num_tokens, 1) for kernel
    act_scale = act_scale_raw.unsqueeze(-1).float()

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        expand_act_scale=None,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        block_shape=None,
    )

    output_ref = use_torch_fp8(
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
        block_shape=None,
    )

    torch.testing.assert_close(output, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ─── FP8 block-wise quantization ────────────────────────────────────────────
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
def test_fused_moe_lora_fp8_kernel_block_wise(
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
    """Test FP8 kernel with block-wise quantization."""
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = 10

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

    # Quantize activations to FP8 with block-wise scale
    hidden_fp8, act_scale_raw = quantize_to_fp8(hidden_states_fp)
    # act_scale_raw is a scalar tensor-wise scale; broadcast to
    # (num_tokens, ceil(K / block_k)) so the kernel can index it per block.
    k_blocks = CEILDIV(K, block_shape[1])
    act_scale = torch.full(
        (num_tokens, k_blocks), act_scale_raw.item(), dtype=torch.float32
    )

    lora_a_stacked = [lora_a_fp8]
    lora_b_stacked = [lora_b_fp8]

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output,
        max_loras,
        num_experts,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        expand_act_scale=None,
        use_fp8_w8a8=True,
        per_channel_quant=False,
        block_shape=block_shape,
    )

    output_ref = use_torch_fp8(
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
        per_channel_quant=False,
        block_shape=block_shape,
    )

    torch.testing.assert_close(output, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ─── Naive block assignment path (small batch) ──────────────────────────────
def use_fused_moe_lora_fp8_kernel_naive(
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
    block_size,
    lora_a_scale_stacked=None,
    lora_b_scale_stacked=None,
    shrink_act_scale=None,
    expand_act_scale=None,
    use_fp8_w8a8=False,
    per_channel_quant=False,
    block_shape=None,
    fully_sharded=False,
    offset=0,
):
    """
    Test helper for naive_block_assignment path.
    Skips moe_lora_align_block_size and uses flattened topk_ids as expert_ids.
    """
    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }

    mul_routed_weight = False

    expert_ids = topk_ids.reshape(-1)
    sorted_token_ids = None
    num_tokens_post_padded = None

    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    num_active_loras = max_loras + 1

    # The custom op requires List[Tensor], not None
    _a_scale = lora_a_scale_stacked if lora_a_scale_stacked is not None else []
    _b_scale = lora_b_scale_stacked if lora_b_scale_stacked is not None else []

    fused_moe_lora_fp8(
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
        mul_routed_weight=mul_routed_weight,
        fully_sharded=fully_sharded,
        offset=offset,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_fp8_kernel_naive_block_assignment(
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
    """
    Test the naive_block_assignment path of the FP8 fused_moe_lora kernel.
    Non-quantized variant through the FP8 op.
    """
    torch.set_default_device(device)
    set_random_seed(seed)

    SPARSITY_FACTOR = 8
    assert num_tokens * top_k_num * SPARSITY_FACTOR <= num_experts * max_loras, (
        f"Test configuration doesn't meet naive_block_assignment condition: "
        f"{num_tokens} * {top_k_num} * {SPARSITY_FACTOR} > {num_experts} * {max_loras}"
    )

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

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel_naive(
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
        block_size,
        use_fp8_w8a8=False,
    )

    output_ref = use_torch_fp8(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
    )

    torch.testing.assert_close(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)


# ─── Naive block assignment + FP8 tensor-wise ───────────────────────────────
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_fp8_kernel_naive_tensor_wise(
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
    """
    Test naive_block_assignment path with FP8 tensor-wise quantization.
    """
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

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_fp8_kernel_naive(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_fp8,
        output,
        max_loras,
        block_size,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=act_scale,
        use_fp8_w8a8=True,
        per_channel_quant=False,
    )

    output_ref = use_torch_fp8(
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

    torch.testing.assert_close(output, output_ref.to(dtype), atol=5e-2, rtol=5e-2)


# ─── Fully sharded (multi-GPU) test ─────────────────────────────────────────
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("column_parallel", [True, False])
def test_fused_moe_lora_fp8_kernel_fully_sharded(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    seed,
    column_parallel,
):
    """Test fully sharded (tensor parallel) path of the FP8 kernel."""
    set_random_seed(seed)
    num_sequences = 10
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                nprocs,
                f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}",
                dtype,
                seed,
                N,
                K,
                num_tokens,
                topk_ids,
                topk_weights,
                token_lora_mapping,
                max_lora_rank,
                top_k_num,
                lora_ids,
                max_loras,
                num_experts,
                block_size,
                column_parallel,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(use_fused_moe_lora_fp8_kernel_tensor_parallel, nprocs=2)


def use_fused_moe_lora_fp8_kernel_tensor_parallel(
    local_rank,
    world_size,
    init_method,
    dtype,
    seed,
    N,
    K,
    num_tokens,
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    max_loras,
    num_experts,
    block_size,
    column_parallel,
):
    def _get_shard_slice(shard_size):
        return slice(local_rank * shard_size, (local_rank + 1) * shard_size)

    set_random_seed(seed)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(world_size, 1)
    tp_size = get_tensor_model_parallel_world_size()

    input_dim = K if column_parallel else N
    output_dim = N if column_parallel else K

    lora_a = torch.rand((max_loras, num_experts, max_lora_rank, input_dim), dtype=dtype)
    lora_b = torch.rand(
        (max_loras, num_experts, output_dim, max_lora_rank), dtype=dtype
    )

    hidden_states = torch.rand((num_tokens, input_dim), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, output_dim), dtype=dtype)
    topk_ids = topk_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    lora_ids = lora_ids.to(device)

    ref_output = use_torch_fp8(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        [lora_a],
        [lora_b],
        top_k_num,
    )

    if column_parallel:
        lora_a_shard_size = max_lora_rank // tp_size
        lora_a = lora_a[:, :, _get_shard_slice(lora_a_shard_size), :]
        max_lora_rank = lora_a_shard_size
        offset = 0

        lora_b_shard_size = output_dim // tp_size
        lora_b = lora_b[:, :, _get_shard_slice(lora_b_shard_size), :]
        output = output[:, :, _get_shard_slice(lora_b_shard_size)].contiguous()
    else:
        lora_a_shard_size = input_dim // tp_size
        lora_a = lora_a[:, :, :, _get_shard_slice(lora_a_shard_size)]
        hidden_states = hidden_states[:, _get_shard_slice(lora_a_shard_size)]

        lora_b_shard_size = output_dim // tp_size
        lora_b = lora_b[:, :, _get_shard_slice(lora_b_shard_size), :]
        offset = lora_b_shard_size * local_rank

    use_fused_moe_lora_fp8_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        [lora_a],
        [lora_b],
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
        use_fp8_w8a8=False,
        fully_sharded=True,
        offset=offset,
    )

    if column_parallel:
        output = tensor_model_parallel_all_gather(output)
    else:
        output = tensor_model_parallel_all_reduce(output)

    torch.testing.assert_close(output, ref_output.to(dtype), atol=1e-2, rtol=1e-2)
