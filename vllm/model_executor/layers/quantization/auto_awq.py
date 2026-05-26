# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Union

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.nn import Parameter
from transformers import PretrainedConfig

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_marlin_supports_layer,
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    verify_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kInt4Static,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# AWQ uses a non-standard packing order within int32 values.
# For 4-bit: standard order stores values at bit positions [0,4,8,12,16,20,24,28]
# for indices [0,1,2,3,4,5,6,7], while AWQ stores them for indices
# [0,4,1,5,2,6,3,7]. This permutation reverses that ordering.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _replace_or_register_parameter(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor | None,
) -> None:
    if value is None:
        return
    if hasattr(layer, name):
        replace_parameter(layer, name, value)
    else:
        layer.register_parameter(name, Parameter(value, requires_grad=False))


def _convert_awq_to_standard_format(
    layer: torch.nn.Module,
    w_q_name: str,
    w_zp_name: str,
    size_bits: int,
) -> None:
    """Convert AWQ weight and zero-point tensors to standard GPTQ-like format.

    AWQ packs qweight along the output dim with a non-standard bit order.
    This converts to standard bit order and repacks qweight along the input
    dim, matching the format expected by the MPLinearKernel framework.
    """
    pack_factor = 32 // size_bits
    mask = (1 << size_bits) - 1
    device = getattr(layer, w_q_name).device
    reverse_order = torch.tensor(
        _REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=device
    )
    shifts = torch.arange(0, 32, size_bits, dtype=torch.int32, device=device)

    # --- Convert qweight: (K, N // pack) packed_dim=1 → (K // pack, N) packed_dim=0
    qw = getattr(layer, w_q_name).data
    K, N_packed = qw.shape
    N = N_packed * pack_factor

    # Unpack int32 → individual values, fix AWQ ordering
    unpacked = (qw.unsqueeze(-1) >> shifts) & mask  # (K, N_packed, pack_factor)
    unpacked = unpacked[:, :, reverse_order]
    unpacked = unpacked.reshape(K, N)  # (K, N)

    # Repack along input dim (dim 0)
    unpacked = unpacked.reshape(K // pack_factor, pack_factor, N)
    new_qw = (unpacked.to(torch.int32) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32
    )

    def _noop_loader(*args, **kwargs):
        pass

    new_param = PackedvLLMParameter(
        data=new_qw.contiguous(),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=_noop_loader,
    )
    setattr(layer, w_q_name, new_param)

    # --- Convert qzeros: fix AWQ bit ordering and repack
    # AWQ qzeros: (G, N // pack) packed along dim 1, AWQ bit order
    # Target: (N // pack, G) packed along dim 0, standard bit order
    # This matches the CompressedTensors layout expected by the kernels.
    qz = getattr(layer, w_zp_name).data
    G, _ = qz.shape

    unpacked_zp = (qz.unsqueeze(-1) >> shifts) & mask  # (G, N_packed, pack_factor)
    unpacked_zp = unpacked_zp[:, :, reverse_order]
    unpacked_zp = unpacked_zp.reshape(G, N)  # (G, N) individual values

    # Transpose and repack along dim 0 (output dim)
    unpacked_zp = unpacked_zp.T  # (N, G)
    unpacked_zp = unpacked_zp.reshape(N // pack_factor, pack_factor, G)
    new_qz = (unpacked_zp.to(torch.int32) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32
    )

    new_zp_param = PackedvLLMParameter(
        data=new_qz.contiguous(),
        output_dim=0,
        input_dim=1,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=_noop_loader,
    )
    setattr(layer, w_zp_name, new_zp_param)


class AutoAWQConfig(QuantizationConfig):
    """Config class for AutoAWQ quantization.

    Unified config that supports multiple backends: Triton, Marlin, and XPU.
    Reference: https://arxiv.org/abs/2306.00978
    """

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None = None,
        full_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config or {}

        if self.weight_bits not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported num_bits = {self.weight_bits}. "
                f"Supported num_bits = {self.TYPE_MAP.keys()}"
            )

        self.quant_type = self.TYPE_MAP[self.weight_bits]

    def __repr__(self) -> str:
        return (
            f"AutoAWQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "auto_awq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoAWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        # Ensure full_config uses "awq" as quant_method for MoE fallback compatibility.
        # MoeWNA16Config only accepts "gptq" or "awq", so we normalize here.
        full_config = config.copy()
        full_config["quant_method"] = "awq"
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            full_config,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override to use AutoAWQ for compatible AWQ models."""
        # Don't override on CPU - let cpu_awq handle it
        if current_platform.is_cpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method != "awq":
            return None

        is_valid_user_quant = user_quant is None or user_quant in (
            "awq",
            "awq_marlin",
            "auto_awq",
            "marlin",
        )

        if is_valid_user_quant:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()

            # Check if XPU - use XPU-specific linear method
            if current_platform.is_xpu():
                return AutoAWQXPULinearMethod(self)

            # Check if Marlin is supported and not using batch invariant mode
            # (Marlin kernels are not batch invariant)
            use_marlin = (
                not envs.VLLM_BATCH_INVARIANT
                and current_platform.is_cuda()
                and check_marlin_supported(
                    self.quant_type, self.group_size, self.zero_point
                )
            )

            if use_marlin:
                if not check_marlin_supports_layer(layer, self.group_size):
                    logger.warning_once(
                        "Layer '%s' is not supported by AutoAWQMarlin. "
                        "Falling back to unoptimized AWQ kernels.",
                        prefix,
                    )
                    return AutoAWQLinearMethod(self)
                quant_method = AutoAWQMarlinLinearMethod(self)
                quant_method.input_dtype = get_marlin_input_dtype(prefix)
                return quant_method

            return AutoAWQLinearMethod(self)

        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix,
                getattr(self, "modules_to_not_convert", []),
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AutoAWQMoEMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )

            return AutoAWQMoEMethod(self, layer.moe_config)

        return None

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if not current_platform.is_cuda_alike():
            return False

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class AutoAWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ Marlin.

    Uses choose_mp_linear_kernel to select the best available kernel
    (Conch, Exllama, or Marlin) for the current platform.

    Args:
        quant_config: The AWQ Marlin quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: AutoAWQConfig) -> None:
        self.quant_config = quant_config
        self.quant_type = scalar_types.uint4
        self.input_dtype = None

        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
            has_zp=self.quant_config.zero_point,
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
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype if self.input_dtype is None else self.input_dtype,
            group_size=self.quant_config.group_size,
            zero_points=self.quant_config.zero_point,
            has_g_idx=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for AutoAWQMarlinLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Weights are loaded in AWQ checkpoint format (packed along output dim).
        # Conversion to GPTQ-like format happens in process_weights_after_loading.
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # AWQ checkpoints use a non-standard packing order and pack qweight
        # along the output dimension. Convert to the standard format
        # (GPTQ-like: standard bit order, qweight packed along input dim)
        # before handing off to the kernel.
        _convert_awq_to_standard_format(
            layer, "qweight", "qzeros", self.quant_config.quant_type.size_bits
        )
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class AutoAWQMoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: AutoAWQConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AutoAWQMoEMethod only supports 4bit now.")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        self.use_marlin = True
        self.wna16_moe_backend, self.experts_cls = select_wna16_moe_backend(
            moe, kInt4Static, quant_config.weight_bits
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full", intermediate_size_per_partition
        )
        self.is_k_full = intermediate_size_per_partition == intermediate_size_full

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size
        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        device = layer.w13_qweight.device
        layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        (
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_g_idx,
            w2_g_idx,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            w13_bias,
            w2_bias,
        ) = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_moe_backend,
            layer=layer,
            quant_config=self.quant_config,
            input_dtype=self.input_dtype,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_qzeros=layer.w13_qzeros,
            w2_qzeros=layer.w2_qzeros,
            w13_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)

        # The modular kernel expects w13_weight and w2_weight,
        # but AWQ uses w13_qweight and w2_qweight
        # Alias for modular kernel
        layer.w13_weight = layer.w13_qweight
        # Alias for modular kernel
        layer.w2_weight = layer.w2_qweight

        replace_parameter(layer, "w13_scales", w13_scale)
        replace_parameter(layer, "w2_scales", w2_scale)
        _replace_or_register_parameter(
            layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices
        )
        _replace_or_register_parameter(
            layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices
        )
        _replace_or_register_parameter(layer, "w13_g_idx", w13_g_idx)
        _replace_or_register_parameter(layer, "w2_g_idx", w2_g_idx)
        _replace_or_register_parameter(layer, "w13_qzeros", w13_qzeros)
        _replace_or_register_parameter(layer, "w2_qzeros", w2_qzeros)
        _replace_or_register_parameter(
            layer, "w13_input_global_scale", w13_input_global_scale
        )
        _replace_or_register_parameter(
            layer, "w2_input_global_scale", w2_input_global_scale
        )
        _replace_or_register_parameter(layer, "w13_bias", w13_bias)
        _replace_or_register_parameter(layer, "w2_bias", w2_bias)

        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        """Build the FusedMoEKernel for this layer."""

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            is_k_full=self.is_k_full,
            w13_g_idx=getattr(layer, "w13_g_idx", None),
            w2_g_idx=getattr(layer, "w2_g_idx", None),
            w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import (
            awq_marlin_moe_quant_config,
        )

        return awq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None)
            if self.quant_config.zero_point
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if self.quant_config.zero_point
            else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            a1_gscale=getattr(layer, "w13_input_global_scale", None),
            a2_gscale=getattr(layer, "w2_input_global_scale", None),
        )

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: RoutedExperts,
    ):
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


class AutoAWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ using Triton kernels.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AutoAWQConfig):
        self.quant_config = quant_config

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
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        # Batch invariant mode requires torch.matmul path
        # for Triton override
        if FP16_MATMUL_HEURISTIC_CONDITION or envs.VLLM_BATCH_INVARIANT:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class AutoAWQXPULinearMethod(LinearMethodBase):
    """Linear method for AWQ on XPU using int4 GEMM kernel.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AutoAWQConfig):
        self.quant_config = quant_config

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
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        from vllm_xpu_kernels.quantization._quantize_convert import (
            AWQUtils,
            transpose_onednn_woq_format,
        )

        layer.xpu_output_size = layer.qweight.size(1) * self.quant_config.pack_factor
        qweight_new, qzeros_new = AWQUtils.repack(layer.qweight, layer.qzeros)
        if qweight_new.shape != layer.qweight.data.shape:
            layer.qweight.data = layer.qweight.data.view_as(qweight_new)
        if qzeros_new.shape != layer.qzeros.data.shape:
            layer.qzeros.data = layer.qzeros.data.view_as(qzeros_new)
        layer.qweight.data.copy_(qweight_new)
        layer.qzeros.data.copy_(qzeros_new)
        transpose_onednn_woq_format(layer, "awq", False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        # Normalize group_size to match create_weights logic
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = layer.qweight.shape[0]  # input_size_per_partition

        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            group_size,
            None,
        )
        out_shape = x.shape[:-1] + (layer.xpu_output_size,)
        return out.reshape(out_shape)
