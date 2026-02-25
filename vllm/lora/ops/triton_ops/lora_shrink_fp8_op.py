# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

import torch

from vllm.lora.ops.triton_ops.fp8_kernel_utils import do_shrink_kernel_fp8
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr, get_lora_op_configs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_SHRINK_LORA_SCALE_PTR_DICT: dict[tuple[int, ...], tuple] = {}


def _get_shrink_lora_scale_ptr(
    lora_scale_weights: list[torch.Tensor], device: torch.device
):
    """
    `_SHRINK_LORA_SCALE_PTR_DICT` collects the required information during
    `profile_run`. After this, it remains constant and subsequent usage is
    through LUT.

    Returns a tuple of (scale_ptr_tensor, l_stride, n_stride, k_stride).

    Supports scale tensors of varying dimensionality:
    - 1D: (lora_num,) — tensor-wise quantization
    - 2D: (lora_num, N) — per-channel quantization
    - 3D: (lora_num, N, K) — block-wise quantization
    - 4D: (lora_num, 1, N, K) — block-wise with extra dim (squeezed to 3D)

    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_scale_weights)

    if values := _SHRINK_LORA_SCALE_PTR_DICT.get(key):
        return values

    tensor_ptrs = []
    scale_l_strides = []
    scale_n_strides = []
    scale_k_strides = []
    for lora_scale_weight in lora_scale_weights:
        if lora_scale_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_scale_weight.size(1) == 1
            lora_scale_weight = lora_scale_weight.squeeze(dim=1)
        assert 1 <= lora_scale_weight.ndim <= 3
        assert lora_scale_weight.is_contiguous()
        tensor_ptrs.append(lora_scale_weight.data_ptr())
        scale_l_strides.append(
            lora_scale_weight.stride(0) if lora_scale_weight.ndim > 0 else 0
        )
        scale_n_strides.append(
            lora_scale_weight.stride(-2)
            if lora_scale_weight.ndim > 2
            else (lora_scale_weight.stride(-1) if lora_scale_weight.ndim > 1 else 1)
        )
        scale_k_strides.append(
            lora_scale_weight.stride(-1) if lora_scale_weight.ndim > 2 else 0
        )
    if len(lora_scale_weights) > 1:
        scale_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        scale_ptr_tensor = lora_scale_weights[0]

    if (
        len(set(scale_l_strides)) > 1
        or len(set(scale_n_strides)) > 1
        or len(set(scale_k_strides)) > 1
    ):
        raise ValueError("All LoRA scale weights must have the same stride.")

    _SHRINK_LORA_SCALE_PTR_DICT[key] = (
        scale_ptr_tensor,
        scale_l_strides[0],
        scale_n_strides[0],
        scale_k_strides[0],
    )
    return _SHRINK_LORA_SCALE_PTR_DICT.get(key)


@triton.jit
def _lora_shrink_kernel_fp8(
    input_ptr,
    lora_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    a_scale_m_stride,
    a_scale_k_stride,
    b_scale_l_stride,
    b_scale_n_stride,
    b_scale_k_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,  ## should always be false in shrink kernel
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K

    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M

    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)

    # Column-major ordering within groups for better cache reuse
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )

    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel_fp8(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        a_scale_ptr,
        b_scale_ptr,
        N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        # input strides
        input_d0_stride,
        input_d1_stride,
        # lora strides
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        # scale strides
        a_scale_m_stride,
        a_scale_k_stride,
        b_scale_l_stride,
        b_scale_n_stride,
        b_scale_k_stride,
        # output strides
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        # block size for block-wise quantization
        group_n,
        group_k,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        per_channel_quant,
        launch_pdl,
    )


@torch.inference_mode()
def _lora_shrink_fp8(
    inputs: torch.Tensor,  # shape [num_tokens, hidden_size] - FP8 or FP16/BF16
    lora_a_weights: list[
        torch.Tensor
    ],  # shape [num_loras, lora_rank, hidden_size] - FP8 or FP16/BF16
    output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: int,  # number of active LoRAs (unused here, for API compat)
    scaling: float,
    b_scale: list[torch.Tensor],  # LoRA weight scale per slice
    a_scale: torch.Tensor | None = None,  # Activation scale - per-token or block-wise
    group_k: int = 0,  # Block size for K in block-wise quantization (0 = tensor-wise)
    group_n: int = 0,  # Block size for N in block-wise quantization
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    per_channel_quant: bool = False,
) -> None:
    """
    Args:
        inputs: FP8 or FP16/BF16 input tensor [num_tokens, hidden_size]
        lora_a_weights: List of FP8 or FP16/BF16 LoRA A weights per slice
        output_tensor: Output tensor (FP16/BF16/FP32)
        token_lora_mapping: Token to LoRA ID mapping
        token_indices_sorted_by_lora_ids: Sorted token indices
        num_tokens_per_lora: Number of tokens per LoRA
        lora_token_start_loc: Start location for each LoRA's tokens
        lora_ids: LoRA IDs to process
        scaling: LoRA scaling factor
        a_scale: Activation quantization scales
        b_scale: Weight quantization scales per slice
        group_k: Block size for K dimension quantization
        group_n: Block size for N dimension quantization
        use_fp8_w8a8: Whether to use FP8 weights and activations
        use_int8_w8a8: Whether to use INT8 weights and activations
        use_int8_w8a16: Whether to use INT8 weights and FP16 activations
        per_channel_quant: Whether to use per-channel quantization
    """
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        # None of the inputs require LoRA.
        return

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    # metadata sanity check
    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    output_tensor.zero_()

    # Get LoRA weight pointers
    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2) = (
        _get_lora_a_ptr(lora_a_weights, inputs.device)
    )

    # Get scale pointers if using FP8
    if use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16:
        assert a_scale is not None or use_int8_w8a16, (
            "a_scale required for FP8/INT8 w8a8"
        )
        assert b_scale is not None, "b_scale required for FP8/INT8"

        b_scale_ptr_tensor, b_scale_l_stride, b_scale_n_stride, b_scale_k_stride = (
            _get_shrink_lora_scale_ptr(b_scale, inputs.device)
        )
        # Get strides from the first scale tensor
        # b_scale_strides = (
        #     b_scale[0].stride(0),  # stride for lora dimension
        #     b_scale[0].stride(-1)
        #     if b_scale[0].ndim > 1
        #     else 1,  # stride for n dimension
        #     0,  # Not used for 2D scale tensors
        # )
        a_scale_ptr = (
            a_scale if a_scale is not None else torch.tensor(1.0, device=inputs.device)
        )
    else:
        b_scale_ptr_tensor = torch.tensor(0, device=inputs.device)
        b_scale_l_stride = 0
        b_scale_n_stride = 0
        b_scale_k_stride = 0
        a_scale_ptr = torch.tensor(0, device=inputs.device)
        # b_scale_strides = (0, 0, 0)

    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size, N=rank
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    # Triton kernel configs
    kernel_config = get_lora_op_configs(
        "shrink",
        max_loras=MAX_LORAS,
        batch=M,
        hidden_size=K,
        rank=N,
        num_slices=NUM_SLICES,
    )
    BLOCK_M = kernel_config["block_m"]
    BLOCK_N = kernel_config["block_n"]
    BLOCK_K = kernel_config["block_k"]
    SPLIT_K = kernel_config["split_k"]
    NUM_WARPS = kernel_config["num_warps"]
    NUM_STAGES = kernel_config["num_stages"]
    NUM_CTAS = kernel_config["num_ctas"]
    GROUP_SIZE_M = kernel_config.get("group_size_m", 8)
    assert BLOCK_K is not None and SPLIT_K is not None
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0

    # Grid configuration with column-major ordering support
    grid = (
        SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        num_active_loras,
    )

    # Determine scale strides
    if use_fp8_w8a8 or use_int8_w8a8:
        if a_scale is not None and a_scale.ndim == 2:
            a_scale_m_stride = a_scale.stride(0)
            a_scale_k_stride = a_scale.stride(1)
        else:
            a_scale_m_stride = 0
            a_scale_k_stride = 0
    else:
        a_scale_m_stride = 0
        a_scale_k_stride = 0

    # We disable PDL temporarily because LoRA kernels are not launching back-to-back,
    # making PDL invalid and affecting the kernel performance.
    use_gdc = False  # supports_pdl(inputs.device)
    _lora_shrink_kernel_fp8[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        a_scale_ptr,
        b_scale_ptr_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        a_scale_m_stride,
        a_scale_k_stride,
        b_scale_l_stride,
        b_scale_n_stride,
        b_scale_k_stride,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        group_n,
        group_k,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        GROUP_SIZE_M,
        NUM_SLICES,
        use_gdc,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        per_channel_quant,
        use_gdc,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
    )

    return


def _lora_shrink_fp8_fake(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: int,
    scaling: float,
    b_scale: list[torch.Tensor],  # LoRA weight scale per slice
    a_scale: torch.Tensor | None = None,  # Activation scale - per-token or block-wise
    group_k: int = 0,  # Block size for K in block-wise quantization (0 = tensor-wise)
    group_n: int = 0,  # Block size for N in block-wise quantization
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    per_channel_quant: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_shrink_fp8",
        op_func=_lora_shrink_fp8,
        mutates_args=["output_tensor"],
        fake_impl=_lora_shrink_fp8_fake,
    )
    lora_shrink_fp8 = torch.ops.vllm.lora_shrink_fp8

except AttributeError:
    lora_shrink_fp8 = _lora_shrink_fp8
