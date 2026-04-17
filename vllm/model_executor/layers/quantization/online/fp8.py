# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import per_block_cast_to_fp8

# ---------------------------------------------------------------------------
# Online FP8 Linear Methods
# ---------------------------------------------------------------------------


class _Fp8OnlineLinearBase(LinearMethodBase):
    """Shared base for online FP8 linear methods. Loads fp16/bf16 checkpoint
    weights onto meta device and materializes them just-in-time."""

    uses_meta_device: bool = True

    def __init__(self):
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)


class Fp8PerTensorOnlineLinearMethod(_Fp8OnlineLinearBase):
    """Online tensorwise FP8 linear quantization.
    Loads fp16/bf16 weights and quantizes them per-tensor during loading."""

    def __init__(self):
        super().__init__()

        self.weight_quant_key = kFp8StaticTensorSym
        # Use per-token quantization for better perf if dynamic and cutlass
        if cutlass_fp8_supported():
            self.activation_quant_key = kFp8DynamicTokenSym
        else:
            self.activation_quant_key = kFp8DynamicTensorSym

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight.t().data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, use BF16 dequant
        if envs.VLLM_BATCH_INVARIANT:
            weight_fp8 = layer.weight.to(torch.bfloat16)
            weight_scale = layer.weight_scale.to(torch.bfloat16)
            if weight_scale.numel() == 1:
                # Per-tensor: simple scalar multiplication
                weight_bf16 = weight_fp8 * weight_scale
            else:
                # Multiple scales (fused modules like QKV)
                if (
                    weight_scale.dim() == 1
                    and weight_scale.shape[0] == weight_fp8.shape[0]
                ):
                    # Per-row scaling
                    weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                else:
                    # Fallback
                    weight_bf16 = weight_fp8 * weight_scale
            return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        return self.fp8_linear.apply_weights(layer, x, bias)


class Fp8PerBlockOnlineLinearMethod(_Fp8OnlineLinearBase):
    """Online blockwise FP8 linear quantization.
    Loads fp16/bf16 weights and quantizes them per-block during loading."""

    def __init__(self):
        super().__init__()
        self.weight_block_size = [128, 128]
        self.activation_quant_key = create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, self.weight_block_size[0]),
        )
        self.weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(*self.weight_block_size)
        )

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.weight_block_size = self.weight_block_size

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        block_size = self.weight_block_size

        qweight, weight_scale_inv = per_block_cast_to_fp8(
            layer.weight, block_size=block_size, use_ue8m0=False
        )

        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)

        self.fp8_linear.process_weights_after_loading(layer)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.weight_block_size is not None

        # Note: batch invariance already handled in the function below
        return self.fp8_linear.apply_weights(
            layer,
            x,
            bias,
        )


# ---------------------------------------------------------------------------
# Online FP8 MoE Methods
# ---------------------------------------------------------------------------


class _Fp8OnlineMoEBase(OnlineMoEMethodBase):
    """Shared base for online FP8 MoE methods. Loads fp16/bf16 checkpoint
    weights onto meta device and materializes them just-in-time."""

    # Declared here for mypy; actual values are set in __init__.
    fp8_backend: "Fp8MoeBackend"
    experts_cls: "type[mk.FusedMoEExperts] | None"
    weight_scale_name: str
    weight_block_size: list[int] | None

    def __init__(
        self,
        *,
        weight_block_size: list[int] | None,
        layer: torch.nn.Module,
    ):
        super().__init__(layer.moe_config)
        self.weight_block_size = weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = kFp8DynamicTensorSym

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def _setup_kernel(
        self,
        layer: "FusedMoE",
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            convert_to_fp8_moe_kernel_format,
            make_fp8_moe_kernel,
        )

        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig":
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            make_fp8_moe_quant_config,
        )

        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
        )

        self._maybe_inject_biases(quant_config, layer)
        return quant_config


class Fp8PerTensorOnlineMoEMethod(_Fp8OnlineMoEBase):
    """Online tensorwise FP8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-tensor during loading."""

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        super().__init__(
            weight_block_size=None,
            layer=layer,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(@ksayers): inplace fp8 quant kernel, initialize scales with ones
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # If checkpoint is fp16, quantize in place.
        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        w13_scale = torch.ones(
            layer.num_experts, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(layer.num_experts, device=w2.device, dtype=torch.float32)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = ops.scaled_fp8_quant(
                layer.w13_weight[expert, :, :]
            )
            w2[expert, :, :], w2_scale[expert] = ops.scaled_fp8_quant(
                layer.w2_weight[expert, :, :]
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8PerBlockOnlineMoEMethod(_Fp8OnlineMoEBase):
    """Online blockwise FP8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-block during loading."""

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        super().__init__(
            weight_block_size=[128, 128],
            layer=layer,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)

        block_size = self.weight_block_size
        assert block_size is not None
        block_n, block_k = block_size

        # Create block-shaped scales (computed here rather than in
        # create_weights because online quant doesn't need them until now).
        num_experts = layer.local_num_experts
        _, w13_out, w13_in = layer.w13_weight.shape
        _, w2_out, w2_in = layer.w2_weight.shape

        w13_scale = torch.ones(
            num_experts,
            (w13_out + block_n - 1) // block_n,
            (w13_in + block_k - 1) // block_k,
            dtype=torch.float32,
            device=w13.device,
        )
        w2_scale = torch.ones(
            num_experts,
            (w2_out + block_n - 1) // block_n,
            (w2_in + block_k - 1) // block_k,
            dtype=torch.float32,
            device=w2.device,
        )

        for expert in range(num_experts):
            w13[expert], w13_scale[expert] = per_block_cast_to_fp8(
                layer.w13_weight[expert],
                block_size=block_size,
                use_ue8m0=False,
            )
            w2[expert], w2_scale[expert] = per_block_cast_to_fp8(
                layer.w2_weight[expert],
                block_size=block_size,
                use_ue8m0=False,
            )

        layer.weight_block_size = block_size

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            layer.w13_input_scale,
            layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True
