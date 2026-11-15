# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online MXFP4 (microscaling FP4, block-32) quantization methods."""

import dataclasses
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import (
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutedExperts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    _backend_activation_key,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.utils import replace_parameter

# MXFP4 block size (elements per shared e8m0 scale), same as the checkpoint
# and MoE oracle paths.
MXFP4_BLOCK_SIZE = 32

# Midpoints between consecutive FP4 (e2m1) magnitudes 0, 0.5, 1, ..., 4, 6.
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _quantize_mxfp4_round_up_scale(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 with block scale `2**ceil(log2(amax / 6))`, so no value saturates.

    The OCP scale used by `mxfp4_quantize` rounds down and clips the largest
    values of a block, which costs accuracy on Hadamard-rotated weights.
    Returns the same packed layout and e8m0 scales as `mxfp4_quantize`.
    """
    blocks = weight.float().unflatten(-1, (-1, MXFP4_BLOCK_SIZE))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scaled = blocks / torch.exp2(exponent)
    midpoints = torch.tensor(_FP4_MIDPOINTS, device=weight.device)
    codes = torch.bucketize(scaled.abs(), midpoints).to(torch.uint8)
    codes |= (scaled < 0).to(torch.uint8) << 3
    codes = codes.flatten(-2)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return packed, scales


def _quantize_mxfp4_moe_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch quantization: bf16/fp16 stacked expert weights -> MXFP4.

    Returns packed FP4 weights (uint8, two values per byte) and per-block
    (group-32) e8m0 scales (uint8), one pair per expert.
    """
    num_experts = weight.size(0)
    first_quant, first_scale = mxfp4_quantize(weight[0])
    # Pre-allocate the output tensors rather than stacking.
    # This is important for consistent memory layout.
    w_quant = torch.empty(
        (num_experts, *first_quant.shape),
        dtype=first_quant.dtype,
        device=weight.device,
    )
    w_scales = torch.empty(
        (num_experts, *first_scale.shape),
        dtype=first_scale.dtype,
        device=weight.device,
    )
    w_quant[0] = first_quant
    w_scales[0] = first_scale
    for i in range(1, num_experts):
        w_quant[i], w_scales[i] = mxfp4_quantize(weight[i])

    return w_quant, w_scales


class Mxfp4OnlineLinearMethod(OnlineLinearBase):
    """Online MXFP4 linear method.
    Loads bf16/fp16 checkpoints and quantizes weights to MXFP4 (microscaling
    FP4 with block-32 scales) during weight loading.
    """

    activation_quant_key: "QuantKey | None" = kMxfp4Dynamic
    accepts_activation_override = True

    def __init__(self, activation_quant_key: "QuantKey | None" = kMxfp4Dynamic):
        super().__init__()
        self.activation_quant_key = self.resolve_activation_quant_key(
            activation_quant_key, explicit=True
        )
        self.kernel = init_mxfp4_linear_kernel(
            activation_quant_key=self.activation_quant_key
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
        if input_size_per_partition % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{MXFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = self.get_weight_for_quantization(layer)
        weight_fp4, weight_scale = mxfp4_quantize(weight.contiguous())

        layer.input_scale = None
        replace_parameter(layer, "weight", weight_fp4.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        self.kernel.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class Mxfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """MoE method for online MXFP4 (block) quantization."""

    mxfp4_backend: Mxfp4MoeBackend
    experts_cls: "type[mk.FusedMoEExperts] | None"
    activation_quant_key: "QuantKey | None" = kMxfp4Dynamic
    accepts_activation_override = True
    # Activation key the kernel format conversion runs with; None keeps the
    # backend default.
    kernel_activation_key: "QuantKey | None" = None
    hadamard_size = 1

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
        activation_quant_key: "QuantKey | None" = kMxfp4Dynamic,
    ):
        super().__init__(moe)
        self.activation_quant_key = self.resolve_activation_quant_key(
            activation_quant_key, explicit=True
        )
        self.weight_block_size: list[int] = [1, MXFP4_BLOCK_SIZE]
        self.weight_scale_name = "weight_scale"
        # Set in `_setup_kernel` for Triton backends.
        self.w13_precision_config: Any = None
        self.w2_precision_config: Any = None

        self.mxfp4_backend, self.experts_cls = self._select_backend()
        # The oracle may fall back to a backend computing another activation
        # format (EMULATION computes MXFP4): reject an override it cannot honour.
        backend_key = (
            kMxfp4Dynamic
            if self.mxfp4_backend == Mxfp4MoeBackend.EMULATION
            else _backend_activation_key(self.mxfp4_backend)
        )
        if (
            self.activation_quant_key != kMxfp4Dynamic
            and self.activation_quant_key != backend_key
        ):
            raise NotImplementedError(
                f"online MXFP4 MoE backend '{self.mxfp4_backend}' cannot honour "
                f"activation={self.activation_quant_key}; it computes "
                f"activation={backend_key}."
            )

    def _select_backend(
        self,
    ) -> "tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]":
        return select_mxfp4_moe_backend(
            config=self.moe, activation_key=self.activation_quant_key
        )

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _quantize_mxfp4_moe_weight(weight)

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: "FusedMoEParallelConfig",
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            hidden_size % MXFP4_BLOCK_SIZE != 0
            or intermediate_size_per_partition % MXFP4_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "Online MXFP4 MoE requires hidden/intermediate sizes divisible "
                f"by {MXFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        layer.weight_block_size = [1, MXFP4_BLOCK_SIZE]

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
    ) -> None:
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                use_separated_a4w4=self.moe.use_mxfp4_w4a4_dsv4,
                activation_key=self.kernel_activation_key,
            )
        )
        if self.hadamard_size > 1:
            layer.weight_schemas = {
                name: dataclasses.replace(
                    schema, hadamard_block_size=self.hadamard_size
                )
                for name, schema in layer.weight_schemas.items()
            }

        # Handle weight/scale assignment based on backend type
        # Copied from `quark_moe.py`.
        # TODO: This should not be here, replace_parameter should handle
        # triton_kernels.tensor.Tensor.
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            # Triton-based backends: w13/w2 are triton_kernels.tensor.Tensor
            # Store on layer for apply(), scales are PrecisionConfig
            layer.w13_weight_triton = w13
            layer.w2_weight_triton = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale
        else:
            # Standard backends: replace parameters
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.keep_constants(layer)
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        # AITER_MXFP4_FP8 needs static activation scales, which online cannot
        # provide, so only the Triton backends store a PrecisionConfig here.
        if self.mxfp4_backend in TRITON_BACKENDS:
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
            w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            w13_weight, w2_weight = self.get_weights_for_quantization(layer)
            self._zero_padding(layer, w13_weight, w2_weight)
            layer._already_called_process_weights_after_loading = True
            return

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        def quantize(w13_weight: torch.Tensor, w2_weight: torch.Tensor):
            self._zero_padding(layer, w13_weight, w2_weight)
            return (
                *self._quantize_weight(w13_weight),
                *self._quantize_weight(w2_weight),
            )

        w13, w13_scale, w2, w2_scale = self.quantize_weights(layer, quantize)

        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
        )
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True


class Mxfp4HadamardOnlineMoEMethod(Mxfp4OnlineMoEMethod):
    """Online MXFP4 MoE with a Sylvester Hadamard rotation on Humming.

    Expert weights are rotated along their input dimension before
    quantization, and Humming fuses the matching activation rotation into its
    input quantization. Activations default to MXFP4 (W4A4).
    """

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
        hadamard_size: int,
        activation_quant_key: "QuantKey | None" = kMxfp4Dynamic,
    ):
        self.hadamard_size = hadamard_size
        super().__init__(moe=moe, activation_quant_key=activation_quant_key)
        self.kernel_activation_key = self.activation_quant_key

    def _select_backend(
        self,
    ) -> "tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]":
        if self.moe.moe_backend not in ("auto", "humming"):
            raise ValueError(
                "Online MXFP4 with a Hadamard rotation requires "
                f"moe_backend='humming', got {self.moe.moe_backend!r}."
            )
        # Humming runs the requested activations itself (kernel_activation_key),
        # so select it as a weight-only backend.
        return select_mxfp4_moe_backend(
            config=dataclasses.replace(self.moe, moe_backend="humming")
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            hidden_size % self.hadamard_size != 0
            or intermediate_size_per_partition % self.hadamard_size != 0
        ):
            raise ValueError(
                "Online MXFP4 Hadamard MoE requires hidden/intermediate sizes "
                f"divisible by hadamard_size={self.hadamard_size}."
            )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.utils.humming import process_input

        rotated = process_input(
            weight.reshape(-1, weight.size(-1)).contiguous(),
            hadamard_block_size=self.hadamard_size,
        )[0]
        return _quantize_mxfp4_round_up_scale(rotated.view(weight.shape))
