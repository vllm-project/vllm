# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused lora_shrink + lora_expand operation.

Launches both Triton kernels from a single custom op call to enable
PDL (Programmatic Dependent Launch) overlap on SM90+ GPUs. The shrink
kernel signals gdc_launch_dependents() when its output is ready, and
the expand kernel calls gdc_wait() before loading the shrink output,
allowing the expand kernel to pre-fetch LoRA B weights while shrink
is still computing.

This eliminates the inter-op scheduling gap (~5-10us) that occurs when
shrink and expand are launched as separate torch custom ops.
"""

import torch

from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand_kernel
from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink_kernel
from vllm.lora.ops.triton_ops.utils import (
    _get_lora_a_ptr,
    _get_lora_b_ptr,
    get_lora_op_configs,
    supports_pdl,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import direct_register_custom_op


@torch.inference_mode()
def _lora_shrink_expand(
    inputs: torch.Tensor,  # [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # [num_loras, lora_rank, hidden_size]
    shrink_buffer: torch.Tensor,  # [num_slices, num_tokens, lora_rank]
    lora_b_weights: list[torch.Tensor],  # [num_loras, hidden_size, lora_rank]
    output_tensor: torch.Tensor,  # [num_tokens, hidden_size * num_slices]
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,  # CPU tensor [1], kept as tensor for torch.compile
    scaling: float,
    offset_start: int = 0,
) -> None:
    """
    Fused shrink + expand: launches both Triton kernels back-to-back
    from a single custom op for PDL overlap.
    """
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return

    M = inputs.size(0)
    use_gdc = supports_pdl(inputs.device)

    # -- Shrink setup --
    shrink_buffer.zero_()

    (lora_a_ptr, la_s0, la_s1, la_s2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N_s, K_s = lora_a_weights[0].shape[-2:]  # rank, hidden_size
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    sc = get_lora_op_configs(
        "shrink",
        max_loras=MAX_LORAS,
        batch=M,
        hidden_size=K_s,
        rank=N_s,
        num_slices=NUM_SLICES,
    )
    S_BM = sc["block_m"]
    S_BN = sc["block_n"]
    S_BK = sc["block_k"]
    S_SK = sc["split_k"]
    S_GM = sc.get("group_size_m", 8)
    S_EK = K_s % (S_BK * S_SK) == 0

    shrink_grid = (
        S_SK * triton.cdiv(M, S_BM) * triton.cdiv(N_s, S_BN),
        NUM_SLICES,
        num_active_loras.item(),
    )

    # -- Expand setup --
    (
        slice_start,
        lora_b_ptr,
        lb_s0,
        lb_s1,
        lb_s2,
        hidden_sizes,
        same_stride,
        MAX_N,
    ) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)

    K_e = lora_b_weights[0].shape[-1]  # rank

    ec = get_lora_op_configs(
        op_type="expand",
        max_loras=MAX_LORAS,
        batch=M,
        hidden_size=MAX_N,
        rank=K_e,
        num_slices=NUM_SLICES,
        add_inputs=True,
    )
    E_BM = ec["block_m"]
    E_BN = ec["block_n"]
    E_BK = ec["block_k"]
    E_EK = K_e % E_BK == 0

    CAST_TYPE = shrink_buffer.dtype == torch.float32 and lora_b_weights[0].dtype in [
        torch.float16,
        torch.bfloat16,
    ]

    expand_grid = (
        triton.cdiv(M, E_BM) * triton.cdiv(MAX_N, E_BN),
        NUM_SLICES,
        num_active_loras.item(),
    )

    # -- Back-to-back kernel launch (enables PDL overlap) --
    _lora_shrink_kernel[shrink_grid](
        inputs,
        lora_a_ptr,
        shrink_buffer,
        M,
        N_s,
        K_s,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        la_s0,
        la_s1,
        la_s2,
        shrink_buffer.stride(0),
        shrink_buffer.stride(1),
        shrink_buffer.stride(2),
        S_BM,
        S_BN,
        S_BK,
        S_EK,
        S_SK,
        S_GM,
        NUM_SLICES,
        use_gdc,
        num_warps=sc["num_warps"],
        num_ctas=sc["num_ctas"],
        num_stages=sc["num_stages"],
        launch_pdl=use_gdc,
    )
    _lora_expand_kernel[expand_grid](
        shrink_buffer,
        lora_b_ptr,
        output_tensor,
        M,
        MAX_N,
        K_e,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start,
        shrink_buffer.stride(0),
        shrink_buffer.stride(1),
        shrink_buffer.stride(2),
        lb_s0,
        lb_s1,
        lb_s2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes,
        E_BM,
        E_BN,
        E_BK,
        E_EK,
        True,  # ADD_INPUTS
        CAST_TYPE,
        NUM_SLICES,
        same_stride,
        use_gdc,
        num_warps=ec["num_warps"],
        num_ctas=ec["num_ctas"],
        num_stages=ec["num_stages"],
        launch_pdl=use_gdc,
    )


def _lora_shrink_expand_fake(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    shrink_buffer: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    scaling: float,
    offset_start: int = 0,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_shrink_expand",
        op_func=_lora_shrink_expand,
        mutates_args=["shrink_buffer", "output_tensor"],
        fake_impl=_lora_shrink_expand_fake,
    )
    lora_shrink_expand = torch.ops.vllm.lora_shrink_expand

except AttributeError:
    lora_shrink_expand = _lora_shrink_expand
