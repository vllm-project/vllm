# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, Optional

import torch
import torch.nn.functional as F
from packaging import version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def torchao_version_at_least(torchao_version: str) -> bool:
    if find_spec("torchao"):
        try:
            if version.parse(importlib.metadata.version("torchao")) >= version.parse(
                torchao_version
            ):
                return True
        except (ImportError, version.InvalidVersion):
            return False
    return False


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

    # Temporary hacky workaround for:
    # 1. prefix == 'model.layers.0.self_attn.qkv_proj'
    # 2. skip_modules containing 'model.layers.0.self_attn.q_proj',
    #    'model.layers.0.self_attn.k_proj',
    #    'model.layers.0.self_attn.v_proj
    if "self_attn.qkv_proj" in prefix:
        unfused_q = prefix.replace("qkv_proj", "q_proj")
        unfused_k = prefix.replace("qkv_proj", "k_proj")
        unfused_v = prefix.replace("qkv_proj", "v_proj")
        if (
            (unfused_q in skip_modules)
            and (unfused_k in skip_modules)
            and (unfused_v in skip_modules)
        ):
            return True

    # Temporary hacky workaround for:
    # 1. prefix == 'model.layers.0.mlp.shared_expert.gate_up_proj'
    # 2. skip_modules containing 'model.layers.0.mlp.shared_expert.gate_proj'
    if "gate_up_proj" in prefix:
        unfused_gate = prefix.replace("gate_up_proj", "gate_proj")
        if unfused_gate in skip_modules:
            return True

    for s in skip_modules:
        if prefix == s:
            return True
        if f".{s}." in f".{prefix}.":
            return True
    return False


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(
        self,
        torchao_config,
        skip_modules: Optional[list[str]] = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
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
        self.is_checkpoint_torchao_serialized = is_checkpoint_torchao_serialized

    def __repr__(self) -> str:
        return (
            f"TorchAOConfig({self.torchao_config=}, {self.skip_modules=}, "
            f"{self.is_checkpoint_torchao_serialized=})"
        )

    def get_name(self) -> QuantizationMethods:
        return "torchao"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        """torchao doesn't require additional config files, we use
        `config.json` from huggingface: `model_config.hf_config`
        """
        return []

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

        quant_method = cls.get_from_keys_or(config, ["quant_method"], None)
        is_checkpoint_torchao_serialized = (
            quant_method is not None and "torchao" in quant_method
        )

        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        assert hf_config is not None, "quant_type must be specified"
        assert len(hf_config) == 1 and "default" in hf_config, (
            "Expected only one key 'default' in quant_type dictionary"
        )
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

        return cls(ao_config, skip_modules, is_checkpoint_torchao_serialized)

    @classmethod
    def from_config_file(cls, config_file: str) -> "TorchAOConfig":
        """Initialize class from a config file. Example:
        ```
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        fn = "torchao_config.json"

        with open(fn, "w") as f:
            f.write(json.dumps(config_to_dict(config)))
        ```
        """
        with open(config_file) as f:
            f.seek(0)
            f_read = f.read()
            config_dict = json.loads(f_read)

        hf_config = {"quant_type": {"default": config_dict}}
        return cls.from_config(hf_config)

    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> "TorchAOConfig":
        """Iniitalize class from a config_dict json string, got from
        torchao_config_object = some AOBaseConfig object
        json.dumps(config_to_dict(torchao_config_object))
        """
        config_dict = json.loads(config_dict_json)
        hf_config = {"quant_type": {"default": config_dict}}
        return cls.from_config(hf_config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from torchao.quantization import ModuleFqnToConfig

        if isinstance(layer, FusedMoE):
            module_fqn = prefix

            # Note: `TorchAOConfig` is not supported because for any reasonable
            # MoE quantization we need to leave gates in high precision.
            assert isinstance(self.torchao_config, ModuleFqnToConfig), (
                f"unsupported type {type(self.torchao_config)}"
            )

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
            for k, cur_config in self.torchao_config.module_fqn_to_config.items():
                if not k.startswith(module_fqn):
                    continue
                if first_config is None:
                    first_config = cur_config
                elif cur_config == first_config:
                    pass
                else:
                    raise AssertionError(
                        f"inconsistent configs {first_config} and "
                        f"{cur_config} in a single MoE module, this is "
                        "not supported"
                    )
            if first_config is None:
                first_config = self.torchao_config.module_fqn_to_config["_default"]
            assert first_config is not None
            return TorchAOMoEMethod(
                TorchAOConfig(first_config, self.skip_modules), layer.moe_config
            )

        if not isinstance(layer, LinearBase):
            return None

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        module_fqn = prefix
        if isinstance(self.torchao_config, ModuleFqnToConfig):
            module_fqn_to_config = self.torchao_config.module_fqn_to_config
            c = module_fqn_to_config.get(module_fqn) or module_fqn_to_config.get(
                "_default", None
            )
            if c is not None:
                current_torchao_config = TorchAOConfig(
                    c, self.skip_modules, self.is_checkpoint_torchao_serialized
                )
                return TorchAOLinearMethod(current_torchao_config)
            else:
                return UnquantizedLinearMethod()

        return TorchAOLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


def torchao_quantize_param_data(
    param: torch.Tensor, torchao_config: Any
) -> torch.nn.Parameter:
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
            torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
        )

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
        if self.quant_config.is_checkpoint_torchao_serialized:
            weight = torchao_quantize_param_data(
                weight, self.quant_config.torchao_config
            )

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.is_checkpoint_torchao_serialized:
            return

        # quantize the weight on the fly if the checkpoint is not already
        # quantized by torchao
        weight = torchao_quantize_param_data(
            layer.weight, self.quant_config.torchao_config
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)


class TorchAOMoEMethod(FusedMoEMethodBase):
    """
    A Mixture of Experts method for TorchAO checkpoints.
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

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w13_weight = torchao_quantize_param_data(
            w13_weight, self.quant_config.torchao_config
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torchao_quantize_param_data(
            w2_weight, self.quant_config.torchao_config
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (  # noqa: E501
            Float8Tensor,
        )

        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError("EPLB not supported for `TorchAOMoEMethod` yet.")

        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )

        logger.info_once("torchao MoE weight_only fallback")
        # weight-only fallback
        # TODO(before land): torchao should have a consistent way to convert
        # AOBaseTensor to high precision, so we can remove the if statement
        # below. Context: https://github.com/pytorch/ao/issues/3118
        if isinstance(layer.w13_weight, (NVFP4Tensor, MXTensor)):
            w13 = layer.w13_weight.to_dtype(x.dtype)
            w2 = layer.w2_weight.to_dtype(x.dtype)
        else:
            assert isinstance(layer.w13_weight, Float8Tensor), (
                f"unsupported type {type(layer.w13_weight)}"
            )
            w13 = layer.w13_weight.dequantize()
            w2 = layer.w2_weight.dequantize()

        result = fused_experts(
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

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result
