# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    CopyNumelCounter,
    Fp8KVCacheMethod,
    _copy_missing_attrs,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    Mxfp8MoeBackend,
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


class Mxfp8Config(QuantizationConfig):
    """Config class for online MXFP8 MoE quantization."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        if activation_scheme != "dynamic":
            raise ValueError("MXFP8 online MoE only supports dynamic activation.")
        self.is_checkpoint_fp8_serialized = False
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        if activation_scheme != "dynamic":
            raise ValueError("online MXFP8  only supports dynamic activation.")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(activation_scheme=activation_scheme, ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            ## TODO: Add Mxfp8LinearMethod
            return UnquantizedLinearMethod()

        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp8OnlineMoEMethod(self, layer)

        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)


class Mxfp8OnlineMoEMethod(FusedMoEMethodBase):
    """Online MoE method specialized for SM100 MXFP8 grouped kernels."""

    def __init__(self, quant_config: Mxfp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.backend_name = select_mxfp8_moe_backend()
        self.backend_impl = self._select_backend_impl()

    def _select_backend_impl(
        self,
    ) -> Callable[
        [FusedMoE, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ]:
        if self.backend_name == Mxfp8MoeBackend.VLLM_CUTLASS_GROUPED_GEMM:
            from vllm.model_executor.layers.fused_moe.cutlass_moe import (
                vllm_cutlass_moe_mxfp8_impl,
            )

            return vllm_cutlass_moe_mxfp8_impl
        raise ValueError(f"Unsupported MXFP8 MoE backend: {self.backend_name.value}")

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if (
            hidden_size % MXFP8_BLOCK_SIZE != 0
            or intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "MXFP8 online MoE requires hidden/intermediate sizes divisible by 32."
            )

        weight_loader = extra_weight_attrs["weight_loader"]
        new_extra_weight_attrs = extra_weight_attrs

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                layer._w13_weight_orig_id = id(layer.w13_weight)
                layer._w2_weight_orig_id = id(layer.w2_weight)

                w13_weight = torch.nn.Parameter(
                    torch.empty_like(layer.w13_weight, device=layer._load_device),
                    requires_grad=False,
                )
                set_weight_attrs(w13_weight, extra_weight_attrs)
                _copy_missing_attrs(layer.w13_weight, w13_weight)
                layer.register_parameter("w13_weight", w13_weight)

                w2_weight = torch.nn.Parameter(
                    torch.empty_like(layer.w2_weight, device=layer._load_device),
                    requires_grad=False,
                )
                set_weight_attrs(w2_weight, extra_weight_attrs)
                _copy_missing_attrs(layer.w2_weight, w2_weight)
                layer.register_parameter("w2_weight", w2_weight)
                del layer._load_device

            if id(param) == layer._w13_weight_orig_id:
                param = layer.w13_weight
            elif id(param) == layer._w2_weight_orig_id:
                param = layer.w2_weight

            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]
            layer._loaded_numel += copy_numel_counter.copied_numel

            target_loaded_numel = layer.w13_weight.numel() + layer.w2_weight.numel()
            if layer._loaded_numel == target_loaded_numel:
                self.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True
            return res

        new_extra_weight_attrs["weight_loader"] = patched_weight_loader
        extra_weight_attrs = new_extra_weight_attrs

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer._load_device = torch.get_default_device()

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // MXFP8_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _quantize_mxfp8_moe_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = weight.contiguous()
        num_experts, m_dim, k_dim = weight.shape
        assert k_dim % MXFP8_BLOCK_SIZE == 0

        flat_weight = weight.view(-1, k_dim).contiguous()
        problem_sizes = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=weight.device
        )
        problem_sizes[:, 0] = m_dim
        problem_sizes[:, 1] = 0
        problem_sizes[:, 2] = k_dim
        expert_offsets = torch.arange(
            0,
            num_experts * m_dim,
            m_dim,
            dtype=torch.int32,
            device=weight.device,
        )
        aligned_m = ((m_dim + 127) // 128) * 128
        blockscale_offsets = torch.arange(
            0,
            num_experts * aligned_m,
            aligned_m,
            dtype=torch.int32,
            device=weight.device,
        )

        qweight = torch.empty_like(flat_weight, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (num_experts * aligned_m, k_dim // MXFP8_BLOCK_SIZE),
            dtype=torch.uint8,
            device=weight.device,
        )
        ops.mxfp8_experts_quant(
            flat_weight,
            problem_sizes,
            expert_offsets,
            blockscale_offsets,
            qweight,
            scales,
        )

        qweight = qweight.view_as(weight)
        scales = scales.view(num_experts, aligned_m, k_dim // MXFP8_BLOCK_SIZE)
        if aligned_m != m_dim:
            scales = scales[:, :m_dim, :]
        return qweight, scales

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if (
            layer.w13_weight.shape[2] % MXFP8_BLOCK_SIZE != 0
            or layer.w2_weight.shape[2] % MXFP8_BLOCK_SIZE != 0
        ):
            raise ValueError("MXFP8 online MoE weights must have K divisible by 32.")

        if layer.w13_weight.device == torch.device("meta"):
            w13_weight = torch.nn.Parameter(
                torch.empty_like(layer.w13_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w13_weight, {"weight_loader": layer.w13_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w13_weight, w13_weight)
            layer.register_parameter("w13_weight", w13_weight)
            initialize_single_dummy_weight(layer.w13_weight)
        if layer.w2_weight.device == torch.device("meta"):
            w2_weight = torch.nn.Parameter(
                torch.empty_like(layer.w2_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w2_weight, {"weight_loader": layer.w2_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w2_weight, w2_weight)
            layer.register_parameter("w2_weight", w2_weight)
            initialize_single_dummy_weight(layer.w2_weight)

        w13_q, w13_scale = self._quantize_mxfp8_moe_weight(layer.w13_weight.data)
        w2_q, w2_scale = self._quantize_mxfp8_moe_weight(layer.w2_weight.data)
        replace_parameter(layer, "w13_weight", w13_q)
        replace_parameter(layer, "w2_weight", w2_q)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        layer._already_called_process_weights_after_loading = True

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ## TODO: Handle shared_experts_input
        del shared_experts_input
        return self.backend_impl(
            layer,
            x,
            topk_weights,
            topk_ids,
        )
