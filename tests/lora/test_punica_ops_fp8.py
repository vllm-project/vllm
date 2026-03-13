# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 accuracy tests for LoRA shrink and expand kernels.

Tests the FP8 kernels by:
1. Quantizing bf16 inputs/weights to FP8
2. Dequantizing them back to bf16
3. Running the bf16 reference (sgmv_shrink/sgmv_expand) with dequantized values
4. Comparing FP8 kernel output against this dequantized reference

This isolates kernel correctness from quantization precision loss,
allowing much tighter tolerances than comparing against the original bf16.
"""

import math
from threading import Lock

import pytest
import torch

import vllm.lora.ops.torch_ops as torch_ops
import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta
from vllm.lora.ops.triton_ops.lora_expand_fp8_op import (
    _EXPAND_LORA_SCALE_PTR_DICT,
)
from vllm.lora.ops.triton_ops.lora_shrink_fp8_op import (
    _SHRINK_LORA_SCALE_PTR_DICT,
)
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.utils.torch_utils import set_random_seed

DEVICES = [f"cuda:{0}"]
SEED = [0]

_dict_lock = Lock()


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


# ============================================================================
# Reference implementations (bf16 baseline)
# ============================================================================


def sgmv_shrink_for_nslices(
    nslices,
    inputs_tensor,
    lora_weights_lst,
    out_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    prompt_lora_mapping,
    batches,
    max_seq_length,
    num_tokens,
    scaling,
):
    """Wrapper around torch_ops.sgmv_shrink that handles any nslices."""
    for index in range(nslices):
        torch_ops.sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def sgmv_expand_for_nslices(
    nslices,
    hidden_size,
    inputs_tensor,
    lora_weights_lst,
    out_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    prompt_lora_mapping,
    batches,
    max_seq_length,
    num_tokens,
    add_inputs,
):
    """Wrapper around torch_ops.sgmv_expand that handles any nslices."""
    if nslices == 1:
        torch_ops.sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            torch_ops.sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights_lst[index],
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


# ============================================================================
# FP8 Quantization Helpers
# ============================================================================

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min


def quantize_to_fp8_per_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with per-tensor scaling."""
    amax = tensor.abs().float().max().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    fp8_tensor = (tensor.float() / scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return fp8_tensor, scale.reshape(1)


def quantize_to_fp8_per_channel(
    tensor: torch.Tensor,
    channel_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with per-channel scaling.

    For shrink lora_a weights of shape (num_loras, rank, hidden_size):
        channel_dim=1 gives per-rank scaling -> scale shape (num_loras, rank)
    For expand lora_b weights of shape (num_loras, hidden_size, rank):
        channel_dim=1 gives per-hidden scaling -> scale shape (num_loras, hidden_size)
    """
    # Compute amax along all dims except the leading dims up to channel_dim+1
    reduce_dims = list(range(channel_dim + 1, tensor.ndim))
    if reduce_dims:
        amax = tensor.abs().float().amax(dim=reduce_dims).clamp(min=1e-12)
    else:
        amax = tensor.abs().float().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)

    # Expand scale for broadcasting
    for _ in reduce_dims:
        scale = scale.unsqueeze(-1)
    fp8_tensor = (tensor.float() / scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    scale = scale.squeeze()
    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    return fp8_tensor, scale


def quantize_to_fp8_per_token(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 with per-token (per-row) scaling.

    Input shape: (num_tokens, hidden_size)
    Returns: (fp8_tensor, scale) where scale shape is (num_tokens, 1)
    """
    assert tensor.ndim == 2
    amax = tensor.abs().float().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    fp8_tensor = (tensor.float() / scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return fp8_tensor, scale


def quantize_to_fp8_blockwise(
    tensor: torch.Tensor,
    group_n: int,
    group_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D or 3D tensor to FP8 with block-wise scaling.

    For a 2D tensor (num_tokens, hidden_size):
        Blocks of size (1, group_k) ->
            scale shape (num_tokens, ceil(hidden_size/group_k))

    For a 3D tensor (num_loras, N, K):
        Blocks of size (group_n, group_k) ->
            scale shape (num_loras, ceil(N/group_n), ceil(K/group_k))
    """
    if tensor.ndim == 2:
        M, K = tensor.shape
        n_blocks_k = math.ceil(K / group_k)
        scale = torch.zeros(M, n_blocks_k, dtype=torch.float32, device=tensor.device)
        fp8_tensor = torch.zeros_like(tensor, dtype=FP8_DTYPE)
        for m in range(M):
            for bk in range(n_blocks_k):
                k_start = bk * group_k
                k_end = min(k_start + group_k, K)
                block = tensor[m, k_start:k_end].float()
                amax = block.abs().max().clamp(min=1e-12)
                s = (amax / FP8_MAX).to(torch.float32)
                scale[m, bk] = s
                fp8_tensor[m, k_start:k_end] = (
                    (block / s).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
                )
        return fp8_tensor, scale
    elif tensor.ndim == 3:
        L, N, K = tensor.shape
        n_blocks_n = math.ceil(N / group_n)
        n_blocks_k = math.ceil(K / group_k)
        scale = torch.zeros(
            L, n_blocks_n, n_blocks_k, dtype=torch.float32, device=tensor.device
        )
        fp8_tensor = torch.zeros_like(tensor, dtype=FP8_DTYPE)
        for li in range(L):
            for bn in range(n_blocks_n):
                for bk in range(n_blocks_k):
                    n_start = bn * group_n
                    n_end = min(n_start + group_n, N)
                    k_start = bk * group_k
                    k_end = min(k_start + group_k, K)
                    block = tensor[li, n_start:n_end, k_start:k_end].float()
                    amax = block.abs().max().clamp(min=1e-12)
                    s = (amax / FP8_MAX).to(torch.float32)
                    scale[li, bn, bk] = s
                    fp8_tensor[li, n_start:n_end, k_start:k_end] = (
                        (block / s).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
                    )
        return fp8_tensor, scale
    else:
        raise ValueError(f"Unsupported tensor ndim: {tensor.ndim}")


# ============================================================================
# FP8 Dequantization Helpers
# ============================================================================


def dequantize_fp8_per_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor with per-tensor scale back to output_dtype."""
    return (fp8_tensor.float() * scale.float()).to(output_dtype)


def dequantize_fp8_per_channel(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    channel_dim: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor with per-channel scale back to output_dtype.

    For 3D tensor (num_loras, N, K) with channel_dim=1:
        scale shape is (num_loras, N), broadcast over K.
    """
    expand_scale = scale.float()
    # Add trailing dims for broadcasting
    for _ in range(channel_dim + 1, fp8_tensor.ndim):
        expand_scale = expand_scale.unsqueeze(-1)
    return (fp8_tensor.float() * expand_scale).to(output_dtype)


def dequantize_fp8_per_token(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 2D tensor with per-token scale back to output_dtype.

    fp8_tensor: (num_tokens, hidden_size), scale: (num_tokens, 1)
    """
    return (fp8_tensor.float() * scale.float()).to(output_dtype)


def dequantize_fp8_blockwise(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    group_n: int,
    group_k: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor with block-wise scale back to output_dtype."""
    if fp8_tensor.ndim == 2:
        M, K = fp8_tensor.shape
        out = torch.zeros(M, K, dtype=output_dtype, device=fp8_tensor.device)
        n_blocks_k = math.ceil(K / group_k)
        for m in range(M):
            for bk in range(n_blocks_k):
                k_start = bk * group_k
                k_end = min(k_start + group_k, K)
                out[m, k_start:k_end] = (
                    fp8_tensor[m, k_start:k_end].float() * scale[m, bk].float()
                ).to(output_dtype)
        return out
    elif fp8_tensor.ndim == 3:
        L, N, K = fp8_tensor.shape
        out = torch.zeros(L, N, K, dtype=output_dtype, device=fp8_tensor.device)
        n_blocks_n = math.ceil(N / group_n)
        n_blocks_k = math.ceil(K / group_k)
        for l_idx in range(L):
            for bn in range(n_blocks_n):
                for bk in range(n_blocks_k):
                    n_start = bn * group_n
                    n_end = min(n_start + group_n, N)
                    k_start = bk * group_k
                    k_end = min(k_start + group_k, K)
                    out[l_idx, n_start:n_end, k_start:k_end] = (
                        fp8_tensor[l_idx, n_start:n_end, k_start:k_end].float()
                        * scale[l_idx, bn, bk].float()
                    ).to(output_dtype)
        return out
    else:
        raise ValueError(f"Unsupported tensor ndim: {fp8_tensor.ndim}")


# ============================================================================
# FP8 Data Generation
# ============================================================================


def generate_fp8_shrink_data(
    batches: int,
    hidden_size: int,
    num_loras: int,
    rank: int,
    seq_length: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    quant_mode: str,  # "per_tensor", "per_channel", "blockwise"
    group_k: int = 128,
    group_n: int = 128,
):
    """Generate test data for FP8 shrink kernel.

    Shrink: output = input @ lora_a^T * scaling
    input: (num_tokens, hidden_size) -> quantized to FP8
    lora_a: (num_loras, rank, hidden_size) -> quantized to FP8

    Returns bf16 reference tensors, FP8 quantized tensors with scales,
    and dequantized bf16 tensors for accurate reference computation.
    """
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum().item()

    # Generate bf16 reference data
    inputs_bf16 = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    lora_a_weights_bf16 = []
    for _ in range(nslices):
        lora_a_weights_bf16.append(
            torch.randn(num_loras, rank, hidden_size, dtype=dtype, device=device)
        )

    # Quantize inputs to FP8 and dequantize back for reference
    if quant_mode == "blockwise":
        inputs_fp8, a_scale = quantize_to_fp8_blockwise(
            inputs_bf16, group_n=1, group_k=group_k
        )
        inputs_dequant = dequantize_fp8_blockwise(
            inputs_fp8,
            a_scale,
            group_n=1,
            group_k=group_k,
            output_dtype=dtype,
        )
    elif quant_mode == "per_tensor":
        # Per-tensor: kernel loads a single scalar from a_scale_ptr
        inputs_fp8, a_scale = quantize_to_fp8_per_tensor(inputs_bf16)
        inputs_dequant = dequantize_fp8_per_tensor(
            inputs_fp8,
            a_scale,
            output_dtype=dtype,
        )
    else:
        # per_channel: kernel loads per-token a_scale via ram indexing
        inputs_fp8, a_scale = quantize_to_fp8_per_token(inputs_bf16)
        inputs_dequant = dequantize_fp8_per_token(
            inputs_fp8,
            a_scale,
            output_dtype=dtype,
        )

    # Quantize lora_a weights to FP8 and dequantize back for reference
    b_scales = []
    lora_a_weights_fp8 = []
    lora_a_weights_dequant = []
    for w in lora_a_weights_bf16:
        if quant_mode == "per_tensor":
            w_fp8, w_scale = quantize_to_fp8_per_tensor(w)
            w_dequant = dequantize_fp8_per_tensor(w_fp8, w_scale, output_dtype=dtype)
            # Scale shape: (1,) -> need (num_loras,) for the kernel
            w_scale = w_scale.expand(num_loras).contiguous()
            lora_a_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_a_weights_dequant.append(w_dequant)
        elif quant_mode == "per_channel":
            # Per-channel along rank dim: scale shape (num_loras, rank)
            w_fp8, w_scale = quantize_to_fp8_per_channel(w, channel_dim=1)
            w_dequant = dequantize_fp8_per_channel(
                w_fp8,
                w_scale,
                channel_dim=1,
                output_dtype=dtype,
            )
            lora_a_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_a_weights_dequant.append(w_dequant)
        elif quant_mode == "blockwise":
            w_fp8, w_scale = quantize_to_fp8_blockwise(
                w, group_n=group_n, group_k=group_k
            )
            w_dequant = dequantize_fp8_blockwise(
                w_fp8,
                w_scale,
                group_n=group_n,
                group_k=group_k,
                output_dtype=dtype,
            )
            lora_a_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_a_weights_dequant.append(w_dequant)

    # Output tensor (float32 for shrink)
    out_tensor = torch.zeros(
        nslices, total_tokens, rank, dtype=torch.float32, device=device
    )
    ref_out_tensor = out_tensor.clone()

    # Token-to-lora mapping
    lora_indices_tensor = torch.randint(0, max(num_loras - 1, 1), (batches,)).to(device)
    token_lora_mapping = torch.zeros(total_tokens, dtype=torch.long, device=device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        sl = seq_len_tensor[b_id].item()
        token_lora_mapping[current_offset : current_offset + sl] = lora_index
        current_offset += sl

    return {
        "inputs_bf16": inputs_bf16,
        "inputs_fp8": inputs_fp8,
        "inputs_dequant": inputs_dequant,
        "lora_a_bf16": lora_a_weights_bf16,
        "lora_a_fp8": lora_a_weights_fp8,
        "lora_a_dequant": lora_a_weights_dequant,
        "a_scale": a_scale,
        "b_scales": b_scales,
        "out_tensor": out_tensor,
        "ref_out_tensor": ref_out_tensor,
        "token_lora_mapping": token_lora_mapping,
        "seq_len_tensor": seq_len_tensor,
        "b_seq_start_loc": b_seq_start_loc,
        "lora_indices_tensor": lora_indices_tensor,
        "total_tokens": total_tokens,
    }


def generate_fp8_expand_data(
    batches: int,
    hidden_size: int,
    num_loras: int,
    rank: int,
    seq_length: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    quant_mode: str,  # "per_tensor", "per_channel", "blockwise"
    group_k: int = 128,
    group_n: int = 128,
):
    """Generate test data for FP8 expand kernel (w8a8).

    Expand: output += input @ lora_b^T
    input: (nslices, num_tokens, rank) -> quantized to FP8 (activations)
    lora_b: (num_loras, hidden_size, rank) -> quantized to FP8 (weights)

    In w8a8 mode, both activations and weights are FP8.
    Returns bf16 reference tensors, FP8 quantized tensors with scales,
    and dequantized bf16 tensors for accurate reference computation.
    """
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum().item()

    # Generate bf16 input (shrink output) and quantize to FP8
    inputs_bf16 = torch.randn(nslices, total_tokens, rank, dtype=dtype, device=device)

    # Quantize input to FP8 and dequantize back for reference
    inputs_2d_all = inputs_bf16.reshape(-1, rank)
    if quant_mode == "blockwise":
        # For blockwise, the kernel indexes a_scale by token id (0..total_tokens-1)
        # shared across slices. Compute shared scale across slices, then quantize.
        # First compute per-token-per-block scale across all slices
        n_blocks_k = math.ceil(rank / group_k)
        a_scale = torch.zeros(
            total_tokens, n_blocks_k, dtype=torch.float32, device=device
        )
        for m in range(total_tokens):
            for bk in range(n_blocks_k):
                k_start = bk * group_k
                k_end = min(k_start + group_k, rank)
                # Max across all slices for this token and block
                block_amax = torch.tensor(0.0, device=device)
                for s in range(nslices):
                    block = inputs_bf16[s, m, k_start:k_end].float()
                    block_amax = torch.max(
                        block_amax, block.abs().max().clamp(min=1e-12)
                    )
                a_scale[m, bk] = (block_amax / FP8_MAX).to(torch.float32)

        # Quantize all slices with the shared scale
        inputs_fp8_list = []
        inputs_dequant_list = []
        for s in range(nslices):
            slice_2d = inputs_bf16[s]  # (total_tokens, rank)
            fp8_slice = torch.zeros_like(slice_2d, dtype=FP8_DTYPE)
            dequant_slice = torch.zeros_like(slice_2d)
            for m in range(total_tokens):
                for bk in range(n_blocks_k):
                    k_start = bk * group_k
                    k_end = min(k_start + group_k, rank)
                    block = slice_2d[m, k_start:k_end].float()
                    s_val = a_scale[m, bk]
                    fp8_slice[m, k_start:k_end] = (
                        (block / s_val).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
                    )
                    dequant_slice[m, k_start:k_end] = (
                        fp8_slice[m, k_start:k_end].float() * s_val.float()
                    ).to(dtype)
            inputs_fp8_list.append(fp8_slice)
            inputs_dequant_list.append(dequant_slice)
        inputs_fp8 = torch.stack(inputs_fp8_list, dim=0)
        inputs_dequant = torch.stack(inputs_dequant_list, dim=0)
    elif quant_mode == "per_tensor":
        # Per-tensor: kernel loads a single scalar from a_scale_ptr
        inputs_fp8_2d, a_scale = quantize_to_fp8_per_tensor(inputs_2d_all)
        inputs_dequant_2d = dequantize_fp8_per_tensor(
            inputs_fp8_2d,
            a_scale,
            output_dtype=dtype,
        )
        inputs_fp8 = inputs_fp8_2d.reshape(nslices, total_tokens, rank)
        inputs_dequant = inputs_dequant_2d.reshape(nslices, total_tokens, rank)
    else:
        # per_channel: kernel loads per-token a_scale via ram indexing.
        # The kernel uses the same a_scale for all slices (indexed by token
        # id 0..total_tokens-1), so we compute a shared per-token scale
        # across all slices, then quantize each slice with that shared scale.
        per_slice_views = [inputs_bf16[s] for s in range(nslices)]
        # (nslices, total_tokens, rank) -> max across slices per token
        stacked = torch.stack(per_slice_views, dim=0)  # (nslices, tokens, rank)
        amax = stacked.abs().float().amax(dim=(0, 2), keepdim=False).clamp(min=1e-12)
        # amax shape: (total_tokens,)
        a_scale = (amax / FP8_MAX).to(torch.float32).unsqueeze(1)  # (tokens, 1)
        # Quantize all slices with the shared scale
        inputs_fp8_2d = (
            (inputs_2d_all.float() / a_scale.repeat(nslices, 1))
            .clamp(FP8_MIN, FP8_MAX)
            .to(FP8_DTYPE)
        )
        inputs_dequant_2d = (
            inputs_fp8_2d.float() * a_scale.repeat(nslices, 1).float()
        ).to(dtype)
        inputs_fp8 = inputs_fp8_2d.reshape(nslices, total_tokens, rank)
        inputs_dequant = inputs_dequant_2d.reshape(nslices, total_tokens, rank)

    # Generate bf16 LoRA B weights
    lora_b_weights_bf16 = []
    for _ in range(nslices):
        lora_b_weights_bf16.append(
            torch.randn(num_loras, hidden_size, rank, dtype=dtype, device=device)
        )

    # Quantize LoRA B weights to FP8 and dequantize back for reference
    b_scales = []
    lora_b_weights_fp8 = []
    lora_b_weights_dequant = []
    for w in lora_b_weights_bf16:
        if quant_mode == "per_tensor":
            w_fp8, w_scale = quantize_to_fp8_per_tensor(w)
            w_dequant = dequantize_fp8_per_tensor(w_fp8, w_scale, output_dtype=dtype)
            w_scale = w_scale.expand(num_loras).contiguous()
            lora_b_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_b_weights_dequant.append(w_dequant)
        elif quant_mode == "per_channel":
            # Per-channel along hidden_size dim: scale (num_loras, hidden_size)
            w_fp8, w_scale = quantize_to_fp8_per_channel(w, channel_dim=1)
            w_dequant = dequantize_fp8_per_channel(
                w_fp8,
                w_scale,
                channel_dim=1,
                output_dtype=dtype,
            )
            lora_b_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_b_weights_dequant.append(w_dequant)
        elif quant_mode == "blockwise":
            w_fp8, w_scale = quantize_to_fp8_blockwise(
                w, group_n=group_n, group_k=group_k
            )
            w_dequant = dequantize_fp8_blockwise(
                w_fp8,
                w_scale,
                group_n=group_n,
                group_k=group_k,
                output_dtype=dtype,
            )
            lora_b_weights_fp8.append(w_fp8)
            b_scales.append(w_scale)
            lora_b_weights_dequant.append(w_dequant)

    # Output tensor (initialized randomly for add_inputs)
    out_tensor = torch.randn(
        total_tokens, hidden_size * nslices, dtype=dtype, device=device
    )
    ref_out_tensor = out_tensor.clone()

    # Token-to-lora mapping
    lora_indices_tensor = torch.randint(0, max(num_loras - 1, 1), (batches,)).to(device)
    token_lora_mapping = torch.zeros(total_tokens, dtype=torch.long, device=device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        sl = seq_len_tensor[b_id].item()
        token_lora_mapping[current_offset : current_offset + sl] = lora_index
        current_offset += sl

    return {
        "inputs_bf16": inputs_bf16,
        "inputs_fp8": inputs_fp8,
        "inputs_dequant": inputs_dequant,
        "a_scale": a_scale,
        "lora_b_bf16": lora_b_weights_bf16,
        "lora_b_fp8": lora_b_weights_fp8,
        "lora_b_dequant": lora_b_weights_dequant,
        "b_scales": b_scales,
        "out_tensor": out_tensor,
        "ref_out_tensor": ref_out_tensor,
        "token_lora_mapping": token_lora_mapping,
        "seq_len_tensor": seq_len_tensor,
        "b_seq_start_loc": b_seq_start_loc,
        "lora_indices_tensor": lora_indices_tensor,
        "total_tokens": total_tokens,
    }


# ============================================================================
# FP8 Shrink Kernel Check
# ============================================================================


def check_lora_shrink_fp8_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    scaling: float,
    quant_mode: str,
    group_k: int = 128,
    group_n: int = 128,
):
    """Test FP8 shrink kernel against dequantized bf16 reference.

    Instead of comparing FP8 kernel output against the original bf16 reference
    (which conflates quantization error with kernel error), we:
    1. Quantize bf16 inputs/weights to FP8
    2. Dequantize them back to bf16
    3. Run the bf16 reference (sgmv_shrink) with the dequantized values
    4. Compare FP8 kernel output against this dequantized reference

    This isolates kernel correctness from quantization precision loss,
    allowing much tighter tolerances.
    """
    data = generate_fp8_shrink_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        device,
        quant_mode,
        group_k,
        group_n,
    )

    total_tokens = data["total_tokens"]

    # Setup LoRA kernel metadata
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras, max_num_tokens=total_tokens, device=device
    )
    lora_meta.prepare_tensors(data["token_lora_mapping"])

    out_tensor = data["out_tensor"]

    # Determine quantization params for the kernel
    per_channel = quant_mode == "per_channel"
    gk = group_k if quant_mode == "blockwise" else 0
    gn = group_n if quant_mode == "blockwise" else 0

    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        _SHRINK_LORA_SCALE_PTR_DICT.clear()
        triton_ops.lora_shrink_fp8(
            data["inputs_fp8"],
            data["lora_a_fp8"],
            out_tensor,
            *lora_meta.meta_args(token_nums=total_tokens, specialize_active_lora=False),
            scaling,
            data["b_scales"],
            a_scale=data["a_scale"],
            group_k=gk,
            group_n=gn,
            use_fp8_w8a8=True,
            per_channel_quant=per_channel,
        )

    # Compute reference using dequantized (round-tripped) tensors.
    # This means the reference sees the same quantization error as the kernel,
    # so any difference is purely kernel error.
    ref_out_tensor = data["ref_out_tensor"]
    max_seq_length = data["seq_len_tensor"].max().item()
    sgmv_shrink_for_nslices(
        nslices,
        data["inputs_dequant"],
        data["lora_a_dequant"],
        ref_out_tensor,
        data["b_seq_start_loc"],
        data["seq_len_tensor"],
        data["lora_indices_tensor"],
        batches,
        max_seq_length,
        total_tokens,
        scaling,
    )

    # With dequantized reference, we can use much tighter tolerances
    # since we're only measuring kernel error, not quantization error.
    # Blockwise accumulation order differs from the bf16 reference, so
    # allow a slightly larger margin for sporadic rounding outliers.
    rtol, atol = 0.1, 0.25
    torch.testing.assert_close(
        out_tensor.to(dtype), ref_out_tensor.to(dtype), rtol=rtol, atol=atol
    )


# ============================================================================
# FP8 Expand Kernel Check
# ============================================================================


def check_lora_expand_fp8_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    add_inputs: bool,
    quant_mode: str,
    group_k: int = 128,
    group_n: int = 128,
):
    """Test FP8 expand kernel (w8a8) against dequantized bf16 reference.

    Instead of comparing FP8 kernel output against the original bf16 reference
    (which conflates quantization error with kernel error), we:
    1. Quantize bf16 inputs/weights to FP8
    2. Dequantize them back to bf16
    3. Run the bf16 reference (sgmv_expand) with the dequantized values
    4. Compare FP8 kernel output against this dequantized reference

    This isolates kernel correctness from quantization precision loss,
    allowing much tighter tolerances.
    """
    data = generate_fp8_expand_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        device,
        quant_mode,
        group_k,
        group_n,
    )

    total_tokens = data["total_tokens"]

    # Setup LoRA kernel metadata
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras, max_num_tokens=total_tokens, device=device
    )
    lora_meta.prepare_tensors(data["token_lora_mapping"])

    out_tensor = data["out_tensor"]

    # Determine quantization params for the kernel
    per_channel = quant_mode == "per_channel"
    gk = group_k if quant_mode == "blockwise" else 0
    gn = group_n if quant_mode == "blockwise" else 0

    with _dict_lock:
        _LORA_B_PTR_DICT.clear()
        _EXPAND_LORA_SCALE_PTR_DICT.clear()
        triton_ops.lora_expand_fp8(
            data["inputs_fp8"],
            data["lora_b_fp8"],
            out_tensor,
            *lora_meta.meta_args(token_nums=total_tokens, specialize_active_lora=False),
            data["b_scales"],
            a_scale=data["a_scale"],
            offset_start=0,
            add_inputs=add_inputs,
            group_k=gk,
            group_n=gn,
            use_fp8_w8a8=True,
            per_channel_quant=per_channel,
        )

    # Compute reference using dequantized (round-tripped) tensors.
    ref_out_tensor = data["ref_out_tensor"]
    max_seq_length = data["seq_len_tensor"].max().item()
    sgmv_expand_for_nslices(
        nslices,
        hidden_size,
        data["inputs_dequant"],
        data["lora_b_dequant"],
        ref_out_tensor,
        data["b_seq_start_loc"],
        data["seq_len_tensor"],
        data["lora_indices_tensor"],
        batches,
        max_seq_length,
        total_tokens,
        add_inputs=add_inputs,
    )

    # With dequantized reference, we can use much tighter tolerances
    # since we're only measuring kernel error, not quantization error.
    rtol, atol = 0.1, 0.15
    torch.testing.assert_close(out_tensor, ref_out_tensor, rtol=rtol, atol=atol)


# ============================================================================
# FP8 Test Parameters
# ============================================================================

fp8_test_params = {
    "hidden_sizes": [512, 1024, 2048],
    "batches": [1, 4, 16],
    "num_loras": [1, 4, 8],
    "max_ranks": [8, 16, 32, 64],
}


# ============================================================================
# FP8 Shrink Tests
# ============================================================================


@pytest.mark.parametrize("batches", fp8_test_params["batches"])
@pytest.mark.parametrize("num_loras", fp8_test_params["num_loras"])
@pytest.mark.parametrize("rank", fp8_test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", fp8_test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("quant_mode", ["per_tensor", "per_channel", "blockwise"])
def test_lora_shrink_fp8(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    quant_mode: str,
):
    """Test FP8 shrink kernel with per-tensor, per-channel, and block-wise
    quantization, comparing against the bf16 baseline."""
    torch.set_default_device(device)
    set_random_seed(seed)

    # For blockwise, group sizes must divide evenly or be handled by the kernel
    group_k = 128
    group_n = 128

    # Adjust group sizes if they're larger than the dimensions
    if quant_mode == "blockwise":
        group_k = min(group_k, hidden_size)
        group_n = min(group_n, rank)

    check_lora_shrink_fp8_kernel(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        seq_length=128,
        scaling=0.5,
        quant_mode=quant_mode,
        group_k=group_k,
        group_n=group_n,
    )


# ============================================================================
# FP8 Expand Tests
# ============================================================================


@pytest.mark.parametrize("batches", fp8_test_params["batches"])
@pytest.mark.parametrize("num_loras", fp8_test_params["num_loras"])
@pytest.mark.parametrize("rank", fp8_test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", fp8_test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("quant_mode", ["per_tensor", "per_channel", "blockwise"])
def test_lora_expand_fp8(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    quant_mode: str,
):
    """Test FP8 expand kernel with per-tensor, per-channel, and block-wise
    quantization, comparing against the bf16 baseline."""
    torch.set_default_device(device)
    set_random_seed(seed)

    group_k = 128
    group_n = 128

    # Adjust group sizes if they're larger than the dimensions
    if quant_mode == "blockwise":
        group_k = min(group_k, rank)
        group_n = min(group_n, hidden_size)

    check_lora_expand_fp8_kernel(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        seq_length=128,
        add_inputs=True,
        quant_mode=quant_mode,
        group_k=group_k,
        group_n=group_n,
    )
