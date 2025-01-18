from typing import Any, Callable, Dict, List, Optional

import torch

from vllm.distributed import get_tp_group
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig, GPTQMarlinLinearMethod)
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig, AWQMarlinLinearMethod)
from vllm.model_executor.utils import set_weight_attrs


class MoeQuantIntConfig(QuantizationConfig):
    """Config class for Int8 experts quantization."""

    def __init__(self, linear_quant_method: str, weight_bits: int,
                 group_size: int, has_zp: bool, lm_head_quantized: bool,
                 modules_to_not_convert: Optional[List[str]],
                 full_config: Dict[str, Any]) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.modules_to_not_convert = modules_to_not_convert

        if self.linear_quant_method == "gptq":
            self.linear_quant_config = GPTQMarlinConfig.from_config(
                full_config)
        elif self.linear_quant_method == "awq":
            self.linear_quant_config = AWQMarlinConfig.from_config(full_config)
        else:
            raise ValueError("MoeQuantInt only support gptq now.")

    @classmethod
    def get_name(cls) -> str:
        return "moe_quant_int"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MoeQuantIntConfig":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method == "awq":
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys(
                config, ["modules_to_not_convert"])
        else:
            raise ValueError("moe_quant_int only support gptq and awq.")

        return cls(linear_quant_method, weight_bits, group_size, has_zp,
                   lm_head_quantized, modules_to_not_convert, config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_moe_quant_int_compatible(hf_quant_cfg)
        is_valid_user_quant = (user_quant is None
                               or user_quant == "moe_quant_int")
        if can_convert and is_valid_user_quant:
            return cls.get_name()
        return None

    @classmethod
    def is_moe_quant_int_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if quant_method == "gptq" and not desc_act and num_bits in [4, 8] and \
                GPTQMarlinConfig.is_gptq_marlin_compatible(quant_config):
            return True
        if quant_method == "awq" and num_bits == 4 and \
                AWQMarlinConfig.is_awq_marlin_compatible(quant_config):
            return True
        return False

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            if self.linear_quant_method == "gptq":
                return GPTQMarlinLinearMethod(self.linear_quant_config)
            elif self.linear_quant_method == "awq":
                return AWQMarlinLinearMethod(self.linear_quant_config)
            else:
                raise ValueError("moe_quant_int only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeQuantIntMethod(self)
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeQuantIntMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: MoeQuantIntConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": False
        })

        assert 'weight_loader' in extra_weight_attrs
        weight_loader = extra_weight_attrs['weight_loader']
        wrapped_weight_loader = MoeQuantIntMethod.get_weight_loader(
            layer, weight_loader)
        extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                     2 * intermediate_size,
                                                     hidden_size //
                                                     bit8_pack_factor,
                                                     dtype=torch.uint8),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size,
                                                    intermediate_size //
                                                    bit8_pack_factor,
                                                    dtype=torch.uint8),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.zeros(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size // group_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.zeros(num_experts,
                                                   hidden_size,
                                                   intermediate_size //
                                                   group_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size // bit8_pack_factor,
                hidden_size // group_size,
                dtype=torch.uint8),
                                            requires_grad=False)
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                hidden_size // bit8_pack_factor,
                intermediate_size // group_size,
                dtype=torch.uint8),
                                           requires_grad=False)
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if self.quant_config.linear_quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys = ["w13_g_idx", "w2_g_idx"]
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(torch.empty((0, ),
                                                       dtype=torch.int32),
                                           requires_grad=False)
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp

        return fused_experts(x,
                             layer.w13_qweight,
                             layer.w2_qweight,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             inplace=True,
                             use_int4_w8a16=weight_bits == 4,
                             use_int8_w8a16=weight_bits == 8,
                             w1_scale=layer.w13_scales,
                             w2_scale=layer.w2_scales,
                             w1_zp=layer.w13_qzeros if has_zp else None,
                             w2_zp=layer.w2_qzeros if has_zp else None,
                             block_shape=[0, layer.group_size])

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor.view(-1,
                                 8)[:,
                                    [0, 4, 1, 5, 2, 6, 3, 7]].view(size0, -1)
            tensor = tensor.T.contiguous()
            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_quant_int_weight_loader(param: torch.nn.Parameter,
                                        loaded_weight: torch.Tensor,
                                        weight_name: str, shard_id: str,
                                        expert_id: int):
            if "g_idx" in weight_name:
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return

            device = get_tp_group().device
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            # convert gptq and awq weight to a standard format
            if layer.quant_config.linear_quant_method == "awq":
                assert layer.quant_config.weight_bits == 4
                if "weight" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight,
                                                       "qweight")
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    loaded_weight = loaded_weight.T
            elif layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [4, 8]
                if "weight" in weight_name:
                    loaded_weight = loaded_weight.T.contiguous().view(
                        torch.uint8)
                elif "zeros" in weight_name:
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(
                            loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T

            # repeat the qzeros/scales to fit new group size
            if layer.group_size_div_factor > 1 and "qzeros" in weight_name or "scales" in weight_name:
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1)

            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(layer.tp_size, -1,
                                            loaded_weight.size(1))[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, :shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2:] = tensor
            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.tp_size, -1)[:, tp_rank]
            else:
                weight_loader(param, loaded_weight, weight_name, shard_id,
                              expert_id)

        return moe_quant_int_weight_loader
