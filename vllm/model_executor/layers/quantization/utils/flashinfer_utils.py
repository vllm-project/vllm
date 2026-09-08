# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from flashinfer.fused_moe.core import ActivationType

logger = init_logger(__name__)


def activation_to_flashinfer_int(activation: MoEActivation) -> int:
    return activation_to_flashinfer_type(activation).value


def has_flashinfer_situ_activation() -> bool:
    try:
        from flashinfer.fused_moe.core import ActivationType
    except ImportError:
        return False
    return hasattr(ActivationType, "Situ")


def activation_to_flashinfer_type(activation: MoEActivation) -> "ActivationType":
    from flashinfer.fused_moe.core import ActivationType

    if activation == MoEActivation.SITU:
        situ = getattr(ActivationType, "Situ", None)
        if situ is None:
            raise ValueError("The installed FlashInfer does not support SITU")
        return situ

    # silu and gelu are mapped to their gated versions SwiGLU and GeGLU respectively
    ACTIVATION_TO_FI_ACTIVATION = {
        MoEActivation.SILU_NO_MUL: ActivationType.Silu,
        MoEActivation.GELU_NO_MUL: ActivationType.Gelu,
        MoEActivation.SILU: ActivationType.Swiglu,
        # SwiGLU-OAI uses Swiglu; the OAI alpha/beta/clamp come from gemm1_* args.
        MoEActivation.SWIGLUOAI: ActivationType.Swiglu,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE: ActivationType.Swiglu,
        MoEActivation.GELU: ActivationType.Geglu,
        MoEActivation.GELU_TANH: ActivationType.Geglu,
        MoEActivation.RELU2_NO_MUL: ActivationType.Relu2,
    }
    return ACTIVATION_TO_FI_ACTIVATION[activation]


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor, is_gated_activation: bool
):
    """Shuffle weights for FI TRT-LLM Format"""
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i])
            if is_gated_activation
            else gemm1_weights[i]
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )


def convert_moe_weights_to_flashinfer_trtllm_block_layout(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert expert weights to FlashInfer's block layout.

    This reorders W13 and W2 into the expected epilogue-tiled block layout and
    returns the shuffled weight tensors.
    """
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError(
            "Unquantized Moe Backend FlashInfer TRTLLM requires bfloat16 weights"
        )

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128

    # Reorder rows of W13 and W2 for fused gated activation and convert to the
    # block layout expected by the FlashInfer kernel.
    num_experts = w13_weight.shape[0]

    def _copy_permuted_expert_to_block_layout(
        out: torch.Tensor,
        expert_uint8: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        expert_blocks = expert_uint8.view(
            expert_uint8.shape[0], out.shape[0], block_k
        ).permute(1, 0, 2)
        torch.index_select(
            expert_blocks,
            1,
            source_indices.to(expert_uint8.device),
            out=out,
        )

    w13_rows, w13_cols = w13_weight[0].view(torch.uint8).shape
    w2_rows, w2_cols = w2_weight[0].view(torch.uint8).shape
    w13_weights_shuffled_tensor = torch.empty(
        (num_experts, w13_cols // block_k, w13_rows, block_k),
        dtype=torch.uint8,
        device=w13_weight.device,
    )
    w2_weights_shuffled_tensor = torch.empty(
        (num_experts, w2_cols // block_k, w2_rows, block_k),
        dtype=torch.uint8,
        device=w2_weight.device,
    )

    for i in range(num_experts):
        w13_expert_uint8 = w13_weight[i].view(torch.uint8)

        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_expert_uint8,
            epilogue_tile_m,
            is_gated_act_gemm=is_gated_act_gemm,
        )
        if is_gated_act_gemm:
            rows = w13_expert_uint8.shape[0]
            permute_indices = (permute_indices + rows // 2) % rows
        _copy_permuted_expert_to_block_layout(
            w13_weights_shuffled_tensor[i],
            w13_expert_uint8,
            permute_indices,
        )

        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_weight[i].view(torch.uint8),
            epilogue_tile_m,
        )
        _copy_permuted_expert_to_block_layout(
            w2_weights_shuffled_tensor[i],
            w2_weight[i].view(torch.uint8),
            permute_indices,
        )

    return (
        w13_weights_shuffled_tensor.view(torch.bfloat16),
        w2_weights_shuffled_tensor.view(torch.bfloat16),
    )


def align_fp4_moe_weights_for_fi(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_act_and_mul: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer FP4 MoE kernels require the intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """

    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape
    intermediate *= 2  # because of packed FP4

    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size // 2))
    if is_act_and_mul:
        # Keep the two logical projections independently aligned. Copying the
        # fused [gate, up] tensor contiguously would move the up projection into
        # the padded tail of gate when intermediate is rounded up.
        padded_w13[:, :intermediate, :] = w13[:, :intermediate, :]
        padded_w13[:, padded_intermediate : padded_intermediate + intermediate, :] = (
            w13[:, intermediate:, :]
        )
    else:
        padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate // 2))
    padded_w2[:, :, : w2.shape[2]] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up_dim, hidden_size // 16)
    )
    if is_act_and_mul:
        padded_w13_scale[:, :intermediate, :] = w13_scale[:, :intermediate, :]
        padded_w13_scale[
            :, padded_intermediate : padded_intermediate + intermediate, :
        ] = w13_scale[:, intermediate:, :]
    else:
        padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // 16)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def align_trtllm_fp4_moe_hidden_dim_for_fi(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    min_alignment: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_experts, gate_up_dim, packed_hidden_size = w13.shape
    hidden_size = packed_hidden_size * 2
    padded_hidden_size = round_up(hidden_size, min_alignment)

    if padded_hidden_size == hidden_size:
        return w13, w13_scale, w2, w2_scale, hidden_size

    logger.warning_once(
        "Padding hidden size from %d to %d for TRTLLM NVFP4 MoE weights. "
        "This requires activation slicing at runtime and may cause "
        "performance degradation.",
        hidden_size,
        padded_hidden_size,
    )

    padded_w13 = w13.new_zeros((num_experts, gate_up_dim, padded_hidden_size // 2))
    padded_w13[:, :, :packed_hidden_size] = w13

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, gate_up_dim, padded_hidden_size // 16)
    )
    padded_w13_scale[:, :, : w13_scale.shape[2]] = w13_scale

    padded_w2 = w2.new_zeros((num_experts, padded_hidden_size, w2.shape[2]))
    padded_w2[:, : w2.shape[1], :] = w2

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, padded_hidden_size, w2_scale.shape[2])
    )
    padded_w2_scale[:, : w2_scale.shape[1], :] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_hidden_size


def align_moe_weights_for_fi(
    w13: torch.Tensor, w2: torch.Tensor, is_act_and_mul: bool, min_alignment: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer MoE kernels require the (gated) intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """

    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size))
    if is_act_and_mul:
        padded_w13[:, :intermediate, :] = w13[:, :intermediate, :]
        padded_w13[:, padded_intermediate : padded_intermediate + intermediate, :] = (
            w13[:, intermediate:, :]
        )
    else:
        padded_w13[:, :intermediate, :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


def _shuffle_deepseek_fp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess DeepSeek FP8 block-scale weights for the FlashInfer TRT-LLM
    kernel using the shuffle + BlockMajorK layout variant.

    Returns 4D weight tensors in BlockMajorK layout
    (E, K/block_k, Mn, block_k)
    """
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    epilogue_tile_m = 64
    block_k = 128

    def shuffle_to_block_major_k(w: torch.Tensor) -> torch.Tensor:
        # shuffle_matrix_a's row permutation depends only on (M,
        # epilogue_tile_m), so it is computed once and applied to every expert
        # in a single gather instead of once per expert. Gathering through the
        # BlockMajorK-permuted view also folds convert_to_block_layout into
        # that same kernel. Per-expert loops here cost minutes for a MoE this
        # wide (~24k tiny launches plus a host round-trip each).
        num_experts, m, k = w.shape
        rows = get_shuffle_matrix_a_row_indices(
            w[0].view(torch.uint8), epilogue_tile_m
        ).to(w.device)
        blocked = w.view(torch.uint8).view(num_experts, m, k // block_k, block_k)
        out = blocked.permute(0, 2, 1, 3)[:, :, rows, :].contiguous()
        return out.view(torch.float8_e4m3fn)

    return shuffle_to_block_major_k(w13), shuffle_to_block_major_k(w2)


def _shuffle_mxfp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess MXFP8 weights and scales for the FlashInfer TRT-LLM kernel.

    All three transforms (gate/up row reorder, ``shuffle_matrix_a`` weight
    shuffle, ``shuffle_matrix_sf_a`` scale shuffle) are fixed row/index
    permutations that depend only on the per-expert matrix shape, so the
    permutation is computed once and applied to every expert in a single
    gather instead of once per expert. ``block_scale_interleave`` accepts a
    batched ``(E, M, K)`` scale tensor directly. Output is bit-identical to
    the per-expert loop but ~20x faster (a 288-expert MoE otherwise costs
    seconds per layer at load).
    """
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    epilogue_tile_m = 128

    w13_u = w13.view(torch.uint8)
    w2_u = w2.view(torch.uint8)

    # 1. Interleave gate/up rows (gated activation GEMM layout).
    if is_gated:
        gate_idx = get_reorder_rows_for_gated_act_gemm_row_indices(
            w13_u[0].reshape(w13_u.shape[1], -1)
        )
        w13_u = w13_u[:, gate_idx]
        w13_scale = w13_scale[:, gate_idx]

    def shuffle_weights(t: torch.Tensor) -> torch.Tensor:
        # Row permutation depends only on (M, epilogue_tile_m).
        idx = get_shuffle_matrix_a_row_indices(t[0], epilogue_tile_m).to(t.device)
        return t[:, idx].view(torch.float8_e4m3fn)

    def shuffle_scales(s: torch.Tensor) -> torch.Tensor:
        # shuffle_matrix_sf_a == row-gather (same indices as the weight shuffle)
        # followed by the 128x4 block-scale interleave, which is batch-capable.
        idx = get_shuffle_matrix_a_row_indices(
            s[0].view(torch.uint8).reshape(s.shape[1], -1), epilogue_tile_m
        ).to(s.device)
        interleaved = block_scale_interleave(s[:, idx])
        return interleaved.reshape(s.shape)

    return (
        shuffle_weights(w13_u),
        shuffle_weights(w2_u),
        shuffle_scales(w13_scale),
        shuffle_scales(w2_scale),
    )


def prepare_fp8_moe_layer_for_fi(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor | None,
    is_trtllm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Fp8 MoE weights to flashinfer kernel format

    Note that for trtllm we update the model state dict
    with the scale format needed for these kernels.

    Note that for per-tensor, we update the layer's
    intermediate size if the weights needed padding.
    """

    assert hasattr(layer.moe_config, "is_act_and_mul")
    block_quant = (
        hasattr(layer, "weight_block_size") and layer.weight_block_size is not None
    )
    is_mxfp8 = block_quant and w13_scale.dtype == torch.uint8
    is_deepseek_fp8 = block_quant and not is_mxfp8
    is_gated = layer.activation.is_gated

    # MXFP8 TRT-LLM requires W31 swap + reorder + shuffle.
    if is_mxfp8 and is_trtllm:
        # FlashInfer TRT-LLM SwiGLU expects [up; gate] but vLLM stores
        # [gate; up].  Swap both weights and scales before interleaving.
        if layer.moe_config.is_act_and_mul:
            w13 = swap_w13_to_w31(w13)
            # Scales may be 2D [E, flat] from _quantize_mxfp8_moe_weight;
            # reshape to 3D so swap_w13_to_w31 can flip the two halves,
            # then flatten back.
            if w13_scale.ndim == 2:
                num_rows = w13.shape[1]  # 2 * intermediate_size
                w13_scale = w13_scale.reshape(w13_scale.shape[0], num_rows, -1)
                w13_scale = swap_w13_to_w31(w13_scale)
                w13_scale = w13_scale.reshape(w13_scale.shape[0], -1)
            else:
                w13_scale = swap_w13_to_w31(w13_scale)

        w13, w2, w13_scale, w2_scale = _shuffle_mxfp8_moe_weights(
            w13, w2, w13_scale, w2_scale, is_gated
        )
        return w13, w2, w13_scale, w2_scale

    # Some FI MoE kernels require internal alignment of 16
    # for the gate-up proj. Pad the weights to respect this.
    if not block_quant:
        min_alignment = 16 if is_gated else 128
        w13, w2, new_intermediate = align_moe_weights_for_fi(
            w13,
            w2,
            layer.moe_config.is_act_and_mul,
            min_alignment,
        )
        layer.moe_config.intermediate_size_per_partition = new_intermediate

    # FI kernels require W31 layout rather than W13.
    if layer.moe_config.is_act_and_mul:
        w13 = swap_w13_to_w31(w13)
        if block_quant:
            w13_scale = swap_w13_to_w31(w13_scale)

    # DeepSeekFp8 TRT-LLM: shuffle weights into BlockMajorK layout.
    if is_deepseek_fp8 and is_trtllm:
        w13, w2 = _shuffle_deepseek_fp8_moe_weights(w13, w2)

    # FI TRT-LLM FP8 per-tensor MoE kernel requires weight shuffle
    # and registration of alpha scales.
    if is_trtllm and not block_quant:
        assert w13_input_scale is not None
        assert w2_input_scale is not None

        rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(w13, w2, is_gated)

    # Clamp block scales to avoid NaN from the FlashInfer CUTLASS kernel.
    # Some FP8 models have near-zero block scales (~1e-23) for dead/unused
    # experts. The CUTLASS kernel doesn't handle these correctly on Hopper
    # (SM 9.0), producing NaN instead of near-zero output. Clamping to a
    # small minimum prevents this without affecting model accuracy since
    # these experts' effective weights are already zero.
    if block_quant:
        _FI_CUTLASS_MIN_BLOCK_SCALE = 1e-10
        w13_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)
        w2_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)

    return w13, w2, w13_scale, w2_scale
