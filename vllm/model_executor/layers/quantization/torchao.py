# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEConfig,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def should_skip(prefix: str, skip_modules: list[str]) -> bool:
    """
    Robust skipping logic:
    should_skip("model.model.layers.1.q_proj",
                ["model.model.layers.1.q_proj"])  # True
    should_skip("model.model.layers.10.o_proj", ["o_proj"])  -> True
    should_skip("visual.model.layers.1.q_proj", ["visual"])   -> True
    should_skip("model.model.layers.1.q_proj", ["layers.1"])  -> True
    should_skip("model.model.layers.11.q_proj", ["layers.1"]) -> False
    """
    for s in skip_modules:
        if prefix == s:
            return True
        if f".{s}." in f".{prefix}.":
            return True
    return False


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(self,
                 torchao_config,
                 skip_modules: Optional[list[str]] = None) -> None:
        """
        # TorchAO quantization relies on tensor subclasses. In order,
        # to enable proper caching this needs standalone compile
        if is_torch_equal_or_newer("2.8.0.dev"):
            os.environ["VLLM_TEST_STANDALONE_COMPILE"] = "1"
            logger.info(
                "Using TorchAO: Setting VLLM_TEST_STANDALONE_COMPILE=1")

        # TODO: remove after the torch dependency is updated to 2.8
        if is_torch_equal_or_newer(
                "2.7.0") and not is_torch_equal_or_newer("2.8.0.dev"):
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
            logger.info("Using TorchAO: Setting VLLM_DISABLE_COMPILE_CACHE=1")
        """
        super().__init__()
        self.torchao_config = torchao_config
        self.skip_modules = skip_modules or []

    def __repr__(self) -> str:
        return f"TorchAOConfig({self.torchao_config})"

    def get_name(self) -> QuantizationMethods:
        return "torchao"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchAOConfig":
        """Create the quant config from an hf model config"""
        try:
            from torchao.core.config import config_from_dict
        except ImportError as err:
            raise ImportError(
                "Please install torchao>=0.10.0 via "
                "`pip install torchao>=0.10.0` to use torchao quantization."
            ) from err

        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        assert hf_config is not None, "quant_type must be specified"
        assert len(hf_config) == 1 and "default" in hf_config, (
            "Expected only one key 'default' in quant_type dictionary")
        quant_type = hf_config["default"]
        ao_config = config_from_dict(quant_type)

        # Adds skipped modules defined in "modules_to_not_convert"
        skip_modules = config.get("modules_to_not_convert", []) or []

        # Adds skipped modules defined in "module_fqn_to_config"
        _data = quant_type.get("_data", {})
        if not isinstance(_data, dict):
            _data = {}

        module_fqn = _data.get("module_fqn_to_config", {})
        if not isinstance(module_fqn, dict):
            module_fqn = {}

        for layer, layer_cfg in module_fqn.items():
            if layer_cfg is None:
                skip_modules.append(layer)

        return cls(ao_config, skip_modules)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from torchao.quantization import ModuleFqnToConfig

        if isinstance(layer, FusedMoE):
            module_fqn = prefix

            # TODO(future, low pri): also support TorchAOConfig, but low-pri
            # because we need to keep gates in high precision to maximize
            # acccuracy, and any setup which does that will use
            # `ModuleFqnToConfig` (per-layer) and not `TorchAOConfig` (global
            # per-model config).
            assert isinstance(
                self.torchao_config, ModuleFqnToConfig
            ), f'unsupported type {type(self.torchao_config)}'

            # module_fqn is FQN of the MoE layer, for example
            #
            #   model.layers.0.mlp.experts
            #
            # module_fqn_to_config has configs for individual linears, for
            # example
            #
            #   {
            #     'model.layers.9.mlp.experts.9.up_proj':
            #       Float8DynamicActivationFloat8WeightConfig(...),
            #     ...,
            #   }
            #
            # (i) to properly stitch E 2d experts into one 3d expert we need
            #     all E 2d experts to be quantized the same way
            # (ii) to call existing fused MoE kernels, we need w13 and w2 to
            #     be quantized the same way. Note that this technically doesn't
            #     apply to weight-only quant, but for now let's keep things
            #     simple and enforce it.
            #
            # (i) && (ii) means that we can only have one unique quant config
            # on all of the expert weights.  The code below enforces this.
            #
            # Note: torchao configs are not hashable
            # (see https://github.com/pytorch/ao/issues/3062),
            # so for now we do this check with a for loop.
            first_config = None
            for k, cur_config in self.torchao_config.module_fqn_to_config.items(
            ):
                if k.startswith(module_fqn):
                    if first_config is None:
                        first_config = cur_config
                    elif cur_config == first_config:
                        pass
                    else:
                        raise AssertionError(
                            f'inconsistent configs {first_config} and '
                            f'{cur_config} in a single MoE module, this is '
                            'not supported')
            return TorchAOMoEMethod(
                TorchAOConfig(first_config, self.skip_modules),
                layer.moe_config)

        if not isinstance(layer, LinearBase):
            return None

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        module_fqn = prefix
        if isinstance(self.torchao_config, ModuleFqnToConfig):
            module_fqn_to_config = self.torchao_config.module_fqn_to_config
            c = module_fqn_to_config.get(
                module_fqn) or module_fqn_to_config.get("_default", None)
            if c is not None:
                current_torchao_config = TorchAOConfig(c, self.skip_modules)
                return TorchAOLinearMethod(current_torchao_config)
            else:
                return UnquantizedLinearMethod()

        return TorchAOLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


def torchao_quantize_param_data(param: torch.Tensor,
                                torchao_config: Any) -> torch.nn.Parameter:
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
        param: weight parameter of the linear module
        torchao_config: type of quantization and their arguments we want to
            use to quantize the Tensor
    """
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_

    assert isinstance(torchao_config, AOBaseConfig), f"{torchao_config}"
    """
    Avoid real weight allocation for faster load, since we will
    end up setting it to param.
    """
    with torch.device("meta"):
        # linear can't be top level module since quantize_ is inplace
        # while some of our configs need to do module swap, and only non-top
        # level modules support module swap
        dummy_linear = torch.nn.Sequential(
            torch.nn.Linear(param.shape[1], param.shape[0], bias=False))

    dummy_linear[0].weight = param
    quantize_(dummy_linear, torchao_config)
    return dummy_linear[0].weight


class TorchAOLinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        quant_config: The torchao quantization config, a string that encodes 
            the type of quantization and all relevant arguments.
    """

    def __init__(self, quant_config: TorchAOConfig):
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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        weight = torchao_quantize_param_data(weight,
                                             self.quant_config.torchao_config)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class TorchAOMoEMethod(FusedMoEMethodBase):
    """
    TODO(before land): write me
    """

    def __init__(
        self,
        quant_config: TorchAOConfig,
        moe_config: FusedMoEConfig,
    ):
        super().__init__(moe_config)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
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

        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        w13_weight = torchao_quantize_param_data(
            w13_weight, self.quant_config.torchao_config)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        w2_weight = torchao_quantize_param_data(
            w2_weight, self.quant_config.torchao_config)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

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
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm.model_executor.layers.fused_moe import fused_experts
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `TorchAOMoEMethod` yet.")

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
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        # for now, just dequantize the weights and run MoE in bf16
        # TODO(future): logic to select best MoE kernel to run should go here
        w13 = layer.w13_weight.dequantize()
        w2 = layer.w2_weight.dequantize()
        return fused_experts(
            hidden_states=x,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )
