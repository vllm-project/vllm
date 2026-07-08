# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _quantize as ct_quantize,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import (
    calculate_qparams,
    calculate_range,
)
from torch.nn import Module

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm import (
    CutlassFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
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
    kFp8StaticChannelSym,
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
from vllm.utils.math_utils import round_up

# Symmetric FP8 (E4M3) quant args. The scale/range/quantize math is delegated to
# compressed-tensors so online quant stays numerically identical to the offline
# export; only ``type``/``num_bits``/``symmetric`` are consulted.
_FP8_QUANT_ARGS = QuantizationArgs(
    num_bits=8, type=QuantizationType.FLOAT, symmetric=True
)


def _quantize_fp8_symmetric(
    x: torch.Tensor,
    reduce_dims: int | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric FP8 weight quant shared by the block/channel helpers,
    bit-identical to compressed-tensors' offline export.

    Reduces ``x`` over ``reduce_dims`` (keepdim so the scale broadcasts back) and
    reuses compressed-tensors for every numeric step: ``calculate_qparams`` for
    the bf16 per-group scale, ``calculate_range`` for the e4m3 clamp, and
    ``ct_quantize`` to scale/clamp/cast. Returns ``(qweight, scale)`` with the
    scale in ``x``'s dtype.
    """
    fp8_dtype = current_platform.fp8_dtype()
    min_vals = x.amin(dim=reduce_dims, keepdim=True)
    max_vals = x.amax(dim=reduce_dims, keepdim=True)
    scale, _ = calculate_qparams(min_vals, max_vals, _FP8_QUANT_ARGS)
    q_min, q_max = calculate_range(_FP8_QUANT_ARGS, x.device)
    q = ct_quantize(
        x=x,
        scale=scale,
        zero_point=None,
        q_min=q_min,
        q_max=q_max,
        args=_FP8_QUANT_ARGS,
        dtype=fp8_dtype,
    )
    return q, scale


def _quantize_fp8_blockwise(
    weight: torch.Tensor,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise FP8 weight quant matching compressed-tensors' ``FP8_BLOCK``
    export.
    """
    assert weight.dim() == 2
    block_n, block_k = block_size
    m, n = weight.shape

    # Zero-pad to a block-size multiple (zeros leave each block's amax unchanged).
    pad_m = (block_n - m % block_n) % block_n
    pad_n = (block_k - n % block_k) % block_k
    padded = torch.nn.functional.pad(weight, (0, pad_n, 0, pad_m), value=0)

    # [num_row_blocks, block_n, num_col_blocks, block_k]
    blocks = padded.view(
        padded.size(0) // block_n, block_n, padded.size(1) // block_k, block_k
    )
    q, scale = _quantize_fp8_symmetric(blocks, reduce_dims=(1, 3))
    qweight = q.view_as(padded)[:m, :n].contiguous()
    scale = scale.view(blocks.size(0), blocks.size(2))
    return qweight, scale


def _quantize_fp8_channelwise(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel FP8 weight quant matching compressed-tensors'
    ``FP8_DYNAMIC`` (``CHANNEL``) export. Returns ``(qweight[out, in],
    scale[out, 1])`` with the scale in the weight dtype (bf16); the caller
    upcasts it into the fp32 buffer losslessly.
    """
    assert weight.dim() == 2
    q, scale = _quantize_fp8_symmetric(weight, reduce_dims=1)
    return q.contiguous(), scale


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

        self.block_quant = False
        self.use_deep_gemm = False
        self.use_marlin = False
        self.marlin_input_dtype = None
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
        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight.t().data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        if self.use_marlin and hasattr(self.fp8_linear, "marlin_input_dtype"):
            self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
        self.fp8_linear.process_weights_after_loading(layer)

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
            if isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel):
                return self.fp8_linear.apply_weights(layer, x, bias)

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

        # See _quantize_fp8_blockwise for offline parity.
        qweight, weight_scale_inv = _quantize_fp8_blockwise(
            layer.weight, block_size=block_size
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


class Fp8PtpcOnlineLinearMethod(_Fp8OnlineLinearBase):
    """Online PTPC FP8 linear quantization.

    Per-output-channel weight scale + dynamic per-token activation scale. The
    layout matches the llmcompressor's FP8_DYNAMIC recipe, so accuracy
    is comparable but no pre-quantized checkpoint is required.
    """

    weight_quant_key = kFp8StaticChannelSym
    activation_quant_key = kFp8DynamicTokenSym

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
        # PTPC requires per-token activation FP8; MarlinFP8 is W8A16 and
        # would silently produce a weight-only fp8 model.
        if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
            raise ValueError(
                "FP8 PTPC online quant requires a kernel that honors "
                "per-token activation quantization; MarlinFP8 is W8A16 "
                "weight-only. Requires SM89+ for Cutlass FP8 or ROCm MI3xx "
                "for rowwise scaled_mm."
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        # See _quantize_fp8_channelwise for offline parity.
        qweight, weight_scale = _quantize_fp8_channelwise(layer.weight)
        # Upcast the bf16 scale into the fp32 buffer the kernel expects (value
        # unchanged).
        weight_scale = weight_scale.to(torch.float32)

        replace_parameter(layer, "weight", qweight.t())
        replace_parameter(layer, "weight_scale", weight_scale)

        self.fp8_linear.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled dequant
        if envs.VLLM_BATCH_INVARIANT and not isinstance(
            self.fp8_linear, CutlassFP8ScaledMMLinearKernel
        ):
            weight_dequant = (
                layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype).t()
            )
            return torch.nn.functional.linear(x, weight_dequant.t(), bias)

        return self.fp8_linear.apply_weights(layer, x, bias)


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
    per_act_token_quant: bool = False
    per_out_ch_quant: bool = False

    def __init__(
        self,
        *,
        weight_block_size: list[int] | None,
        layer: torch.nn.Module,
        weight_key: "QuantKey | None" = None,
        activation_key: "QuantKey | None" = None,
        allow_vllm_cutlass: bool = False,
    ):
        super().__init__(layer.moe_config)
        self.weight_block_size = weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Subclasses may pass explicit kernel keys (PTPC needs channelwise +
        # per-token).
        if weight_key is None or activation_key is None:
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
            allow_vllm_cutlass=allow_vllm_cutlass,
        )

    def _setup_kernel(
        self,
        layer: RoutedExperts,
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
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
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

        return make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            block_shape=self.weight_block_size,
            per_act_token_quant=self.per_act_token_quant,
            per_out_ch_quant=self.per_out_ch_quant,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )


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

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        assert self.weight_block_size is not None
        block_size = self.weight_block_size[0]
        return (
            round_up(hidden_size, block_size),
            round_up(intermediate_size_per_partition, block_size),
        )

    def _zero_padding(self, layer: Module) -> None:
        hidden_size = layer.moe_config.hidden_dim_unpadded
        intermediate_size = layer.moe_config.intermediate_size_per_partition_unpadded

        w13_half_size = layer.w13_weight.shape[1] // 2
        if w13_half_size > intermediate_size:
            layer.w13_weight[:, intermediate_size:w13_half_size, :] = 0
            layer.w13_weight[
                :, w13_half_size + intermediate_size : 2 * w13_half_size, :
            ] = 0
        if layer.w13_weight.shape[2] > hidden_size:
            layer.w13_weight[:, :, hidden_size:] = 0

        if layer.w2_weight.shape[1] > hidden_size:
            layer.w2_weight[:, hidden_size:, :] = 0
        if layer.w2_weight.shape[2] > intermediate_size:
            layer.w2_weight[:, :, intermediate_size:] = 0

        if getattr(layer, "w13_bias", None) is not None:
            w13_bias_half_size = layer.w13_bias.shape[1] // 2
            if w13_bias_half_size > intermediate_size:
                layer.w13_bias[:, intermediate_size:w13_bias_half_size] = 0
                layer.w13_bias[
                    :, w13_bias_half_size + intermediate_size : 2 * w13_bias_half_size
                ] = 0

        if (
            getattr(layer, "w2_bias", None) is not None
            and layer.w2_bias.shape[1] > hidden_size
        ):
            layer.w2_bias[:, hidden_size:] = 0

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._zero_padding(layer)

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

        # See _quantize_fp8_blockwise for offline parity; the fp32 scale buffers
        # hold the bf16-rounded values losslessly.
        for expert in range(num_experts):
            w13[expert], w13_scale[expert] = _quantize_fp8_blockwise(
                layer.w13_weight[expert],
                block_size=block_size,
            )
            w2[expert], w2_scale[expert] = _quantize_fp8_blockwise(
                layer.w2_weight[expert],
                block_size=block_size,
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


class Fp8PtpcOnlineMoEMethod(_Fp8OnlineMoEBase):
    """Online PTPC FP8 MoE quantization.

    Quantizes each expert's weights per output channel during loading.
    Activations are quantized dynamically per token at runtime.
    """

    per_act_token_quant: bool = True
    per_out_ch_quant: bool = True

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

        super().__init__(
            weight_block_size=None,
            layer=layer,
            weight_key=kFp8StaticChannelSym,
            activation_key=kFp8DynamicTokenSym,
            allow_vllm_cutlass=True,
        )
        # Reject backends whose make_fp8_moe_quant_config branch silently
        # drops per_act_token_quant / per_out_ch_quant or collapses scales:
        # MARLIN / CPU route through fp8_w8a16_moe_quant_config; FLASHINFER_*
        # fold scales into a per-tensor alpha (oracle/fp8.py).
        if self.fp8_backend in (
            Fp8MoeBackend.MARLIN,
            Fp8MoeBackend.CPU,
            Fp8MoeBackend.FLASHINFER_CUTLASS,
            Fp8MoeBackend.FLASHINFER_TRTLLM,
        ):
            raise ValueError(
                f"FP8 PTPC online MoE quant is not supported with the "
                f"{self.fp8_backend.value} backend, which does not implement "
                "per-output-channel weight scales."
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        # Scale's leading dim is taken from the fp8 weight tensor by
        # construction, so it cannot drift from the weight's expert count
        # under EP / padded MoE.
        n_w13 = layer.w13_weight.shape[1]
        n_w2 = layer.w2_weight.shape[1]
        w13_scale = torch.ones(
            w13.shape[0], n_w13, 1, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(
            w2.shape[0], n_w2, 1, device=w2.device, dtype=torch.float32
        )
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        # See _quantize_fp8_channelwise for offline parity; the fp32 scale
        # buffers hold the bf16-rounded values losslessly.
        for expert in range(layer.local_num_experts):
            w13[expert], w13_scale[expert] = _quantize_fp8_channelwise(
                layer.w13_weight[expert]
            )
            w2[expert], w2_scale[expert] = _quantize_fp8_channelwise(
                layer.w2_weight[expert]
            )

        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=None,
            w2_input_scale=None,
        )

        layer._already_called_process_weights_after_loading = True
