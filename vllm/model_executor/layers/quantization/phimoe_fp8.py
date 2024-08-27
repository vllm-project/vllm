from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, apply_fp8_linear, convert_to_channelwise,
    create_per_tensor_scale_param, cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import is_hip, print_warning_once

logger = init_logger(__name__)


class PhiMoEFp8Config(QuantizationConfig):
    """Config class for PhiMoE FP8."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "phimoe_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80
    
    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PhiMoEFp8Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, FusedMoE):
            return PhiFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class PhiFp8MoEMethod(FusedMoEMethodBase):
    """Phi MoE method for FP8.

    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: PhiMoEFp8Config):
        self.quant_config = quant_config
        self.run_on_sm80 = self.is_sm80()
        self.fp8_dtype = torch.float8_e4m3fn

        if not self.run_on_sm80:
            raise NotImplementedError(
                "Phi FP8 fused MoE is only supported on nvidia sm80.")

        try:
            import vllm._phi_C
        except ImportError:
            raise ImportError("Phi FP8 fused MoE requires the Phi C extension.")

        from vllm.model_executor.layers.phi_ops.moe.tensorrt_llm_moe.ampere_fp8_fused_moe import fused_moe
        self.phi_fused_moe_forward = fused_moe

    def is_sm80(self, device_id=0):
        if not torch.cuda.is_available():
            return False
        device_properties = torch.cuda.get_device_properties(device_id)
        return (device_properties.major == 8 and device_properties.minor == 0)

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=self.fp8_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=self.fp8_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                         2,
                                                         dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Currently used by PhiMOE
        w13_weight = torch.empty_like(layer.w13_weight.data,
                                    dtype=self.fp8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data,
                                    dtype=self.fp8_dtype)
        # Re-initialize w13_scale because we directly quantize
        # merged w13 weights and generate a single scaling factor.
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            layer.num_experts,
            dtype=torch.float32,
            device=w13_weight.device),
                                            requires_grad=False)
        for expert in range(layer.num_experts):
            w13_weight[expert, :, :], layer.w13_weight_scale[
                expert] = ops.scaled_fp8_quant(
                    layer.w13_weight.data[expert, :, :].cuda())
            w2_weight[expert, :, :], layer.w2_weight_scale[
                expert] = ops.scaled_fp8_quant(
                    layer.w2_weight.data[expert, :, :].cuda())

        print_warning_once("Preprocessing weights for A100 FP8 fused MoE")
        w13_weight =  torch.ops._phi_C.preprocess_weights_for_mixed_gemm(
            w13_weight.view(torch.int8).transpose(1,2).contiguous().cpu()).to(w13_weight.device)
        w2_weight =  torch.ops._phi_C.preprocess_weights_for_mixed_gemm(
            w2_weight.view(torch.int8).transpose(1,2).contiguous().cpu()).to(w2_weight.device)
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(dtype=torch.bfloat16)
            .unsqueeze(1)
            .expand(-1, w13_weight.size(-1))
            .contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(dtype=torch.bfloat16)
            .unsqueeze(1)
            .expand(-1, w2_weight.size(-1))
            .contiguous(),
            requires_grad=False,
        )

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        return

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool,
              use_grouped_topk: bool,
              topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None,
              custom_routing_function: Optional[Callable] = None) -> torch.Tensor:

        return self.phi_fused_moe_forward(x,
                                        layer.w13_weight,
                                        layer.w2_weight,
                                        router_logits,
                                        top_k,
                                        renormalize=renormalize,
                                        inplace=True,
                                        use_fp8=True,
                                        w1_scale=layer.w13_weight_scale,
                                        w2_scale=layer.w2_weight_scale,
                                        a1_scale=None,
                                        a2_scale=None,
                                        use_grouped_topk=use_grouped_topk,
                                        num_expert_group=num_expert_group,
                                        topk_group=topk_group,
                                        custom_routing_function=custom_routing_function,
                                        )
