# SPDX-License-Identifier: Apache-2.0
"""PairwiseFP4LinearMethod – online rotation + FP4 quantization."""

from __future__ import annotations

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.fp4_quant_policy import (
    estimate_global_scale,
    quantize_weight_to_fp4,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_applier import (
    apply_givens_rotation,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_plan import (
    RotationPlanBuilder,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
    RotationPlan,
    empty_angles,
    empty_pairs,
    load_plan,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class PairwiseFP4LinearMethod(QuantizeMethodBase):
    """Online pairwise-Givens-rotation + FP4 quantization.

    Loads BF16/FP16 weights, applies Givens rotations and quantizes to
    NVFP4 during ``process_weights_after_loading``.  At inference time,
    optionally rotates activations before calling the standard NVFP4 GEMM.
    """

    def __init__(self, quant_config) -> None:
        # Avoid circular import – quant_config is PairwiseFP4Config
        self.quant_config = quant_config
        self.backend = select_nvfp4_linear_backend()

    # ------------------------------------------------------------------
    # create_weights
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size  # unused

        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size

        if input_size_per_partition % (group_size * 2) != 0:
            raise ValueError(
                f"input_size_per_partition ({input_size_per_partition}) must "
                f"be divisible by group_size*2 ({group_size * 2}) for FP4 "
                f"block quantization + packing."
            )

        # Store partition metadata on the layer for later use.
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # BF16 weight – will be rotated + quantised in
        # process_weights_after_loading.
        weight = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"weight_loader": default_weight_loader})

    # ------------------------------------------------------------------
    # process_weights_after_loading
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        cfg = self.quant_config
        mode = cfg.mode
        group_size = cfg.group_size
        weight = layer.weight.data.float()  # work in fp32

        # ---- 1. Build or load rotation plan -------------------------
        layer_name = getattr(layer, "_layer_name", "unknown")

        if cfg.use_prebuilt_plan and cfg.rotation_plan_path:
            plan = load_plan(cfg.rotation_plan_path)
        else:
            builder = RotationPlanBuilder(cfg.get_plan_builder_config())
            plan = builder.build(
                layer_index=layer_name,
                mode=mode,
                weight=weight if mode in ("weight_only", "joint") else None,
                activation=None,  # no calibration data in v1
            )

        # ---- 2. Rotate weight if applicable -------------------------
        if mode in ("weight_only", "joint") and not plan.is_empty:
            weight = apply_givens_rotation(weight, plan.pairs, plan.angles)

        # ---- 3. Quantize weight to packed FP4 -----------------------
        gs = estimate_global_scale(weight, block_size=group_size)
        packed_weight, block_scales_fp8, gs = quantize_weight_to_fp4(
            weight, gs, block_size=group_size,
        )

        # ---- 4. Set kernel-expected layer attributes ----------------
        # weight (packed uint8)
        layer.weight = Parameter(packed_weight, requires_grad=False)
        # per-block scales (fp8)
        layer.weight_scale = Parameter(block_scales_fp8, requires_grad=False)
        # weight global scale
        weight_global_scale = gs
        layer.weight_global_scale = Parameter(
            weight_global_scale.clone(), requires_grad=False,
        )

        # input_global_scale: use default 1.0 (no calibration data)
        input_global_scale = torch.tensor(1.0, dtype=torch.float32)
        layer.input_global_scale = Parameter(
            input_global_scale, requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / input_global_scale).to(torch.float32), requires_grad=False,
        )
        layer.alpha = Parameter(
            input_global_scale * weight_global_scale, requires_grad=False,
        )

        # ---- 5. Convert to backend kernel format --------------------
        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

        # ---- 6. Store rotation params for activation rotation -------
        if mode in ("activation_only", "joint") and not plan.is_empty:
            layer.register_buffer(
                "rotation_pairs", plan.pairs.to(layer.weight.device),
            )
            layer.register_buffer(
                "rotation_angles", plan.angles.to(layer.weight.device),
            )
        else:
            layer.register_buffer(
                "rotation_pairs", empty_pairs().to(layer.weight.device),
            )
            layer.register_buffer(
                "rotation_angles", empty_angles().to(layer.weight.device),
            )

    # ------------------------------------------------------------------
    # apply (forward)
    # ------------------------------------------------------------------

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Optionally rotate activations.
        rotation_pairs = getattr(layer, "rotation_pairs", None)
        if rotation_pairs is not None and rotation_pairs.numel() > 0:
            rotation_angles = layer.rotation_angles
            x = apply_givens_rotation(x, rotation_pairs, rotation_angles)

        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )
