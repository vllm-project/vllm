# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"
    CUTEDSL = "CUTEDSL"


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor
):
    """Shuffle weights for for FI TRT-LLM Format"""
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


def get_flashinfer_moe_backend() -> FlashinferMoeBackend:
    backend_map = {
        "throughput": FlashinferMoeBackend.CUTLASS,
        "latency": FlashinferMoeBackend.TENSORRT_LLM,
        "masked_gemm": FlashinferMoeBackend.CUTEDSL,
    }

    flashinfer_moe_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
    if flashinfer_moe_backend in backend_map:
        if (
            flashinfer_moe_backend == "latency"
            and not current_platform.is_device_capability_family(100)
        ):
            logger.info_once(
                "Flashinfer TRTLLM MOE backend is only supported on "
                "SM100 and later, using CUTLASS backend instead",
                scope="local",
            )
            return FlashinferMoeBackend.CUTLASS
        return backend_map[flashinfer_moe_backend]
    elif current_platform.is_device_capability(90):
        return FlashinferMoeBackend.CUTLASS

    raise ValueError(
        f"Unknown flashinfer moe backend: {flashinfer_moe_backend!r}. "
        f"Expected one of {list(backend_map.keys())}."
    )


def is_flashinfer_supporting_global_sf(backend: FlashinferMoeBackend | None) -> bool:
    # TODO(shuw@nvidia): Update when new backends are added.
    backends_supporting_global_sf = (
        FlashinferMoeBackend.CUTLASS,
        FlashinferMoeBackend.TENSORRT_LLM,
        FlashinferMoeBackend.CUTEDSL,
    )
    return backend in backends_supporting_global_sf


def convert_moe_weights_to_flashinfer_trtllm_block_layout(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
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
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128

    # Reorder rows of W13 and W2 for fused gated activation and convert to the
    # block layout expected by the FlashInfer kernel.
    num_experts = w13_weight.shape[0]
    device_w13 = w13_weight.device
    device_w2 = w2_weight.device

    w13_weights_shuffled: list[torch.Tensor] = []
    w2_weights_shuffled: list[torch.Tensor] = []

    for i in range(num_experts):
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_weight[i].view(torch.uint8),
            epilogue_tile_m,
        )
        tmp_weights1 = (
            w13_weight[i]
            .clone()
            .view(torch.uint8)[permute_indices.to(device_w13)]
            .contiguous()
        )

        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_weight[i].view(torch.uint8),
            epilogue_tile_m,
        )
        tmp_weights2 = (
            w2_weight[i]
            .clone()
            .view(torch.uint8)[permute_indices.to(device_w2)]
            .contiguous()
        )

        tmp_weights1 = convert_to_block_layout(tmp_weights1.view(torch.uint8), block_k)
        tmp_weights2 = convert_to_block_layout(tmp_weights2.view(torch.uint8), block_k)

        w13_weights_shuffled.append(tmp_weights1.view(torch.bfloat16))
        w2_weights_shuffled.append(tmp_weights2.view(torch.bfloat16))

    # Stack weights for all experts and return as BF16 tensors.
    w13_weights_shuffled_tensor = (
        torch.stack(w13_weights_shuffled).view(torch.bfloat16).contiguous()
    )
    w2_weights_shuffled_tensor = (
        torch.stack(w2_weights_shuffled).view(torch.bfloat16).contiguous()
    )

    return w13_weights_shuffled_tensor, w2_weights_shuffled_tensor


def align_fp8_moe_weights_for_fi(
    w13: torch.Tensor, w2: torch.Tensor, is_act_and_mul: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer FP8 MoE kernels require the (gated) intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """

    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape

    min_alignment = 16
    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
        scope="local",
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


def prepare_fp8_moe_layer_for_fi(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor | None,
    is_trtllm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Some FI MoE kernels require internal alignment of 16
    # for the gate-up proj. Pad the weights to respect this.
    if not block_quant:
        w13, w2, new_intermediate = align_fp8_moe_weights_for_fi(
            w13,
            w2,
            layer.moe_config.is_act_and_mul,
        )
        layer.intermediate_size_per_partition = new_intermediate

    # FI kernels require W31 layout rather than W13.
    if layer.moe_config.is_act_and_mul:
        w13 = swap_w13_to_w31(w13)
        if block_quant:
            w13_scale = swap_w13_to_w31(w13_scale)

    # FI TRT-LLM FP8 per-tensor MoE kernel requires weight shuffle
    # and registration of alpha scales.
    if is_trtllm and not block_quant:
        assert w13_input_scale is not None
        assert w2_input_scale is not None

        rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(w13, w2)

    return w13, w2, w13_scale
