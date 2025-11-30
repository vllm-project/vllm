# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
import types
from importlib.util import find_spec
from typing import Any, Optional

import regex as re
import torch
import torch.nn.functional as F
from packaging import version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.torchao_utils import (
    maybe_get_torchao_config_for_moe_layer,
    torchao_config_to_compressed_tensors_config,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def _bond_method_to_cls(func, obj):
    if hasattr(func, "__self__") or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)


def _get_weight_attrs(param):
    # record attributes attached to the weight, so we can
    # recover later
    recorded_weight_attr = {}
    for key in param.__dict__:
        if hasattr(param, key):
            attr = getattr(param, key)
            if not callable(attr):
                recorded_weight_attr[key] = attr
            elif hasattr(attr, "__self__") and param is attr.__self__:
                # if attr is a bonded method for an instance, and
                # attr.__self__ points to the instance (param)
                # we'll record the underlying function object
                recorded_weight_attr[key] = attr.__func__
            else:
                recorded_weight_attr[key] = attr
    return recorded_weight_attr


def _restore_weight_attrs(param, recorded_weight_attr):
    for attr_name, attr in recorded_weight_attr.items():
        if not hasattr(param, attr_name):
            setattr(param, attr_name, _bond_method_to_cls(attr, param))


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


if torchao_version_at_least("0.15.0"):
    from torchao.prototype.tensor_conversion.api import (
        convert_to_packed_tensor_based_on_current_hardware,
    )
else:
    convert_to_packed_tensor_based_on_current_hardware = lambda t: t


class TorchAOWrappingCompressedTensorsW8A8Fp8MoEMethod(
    CompressedTensorsW8A8Fp8MoEMethod
):
    """
    A thin wrapper around `CompressedTensorsW8A8Fp8MoEMethod` for compatibility
    with torchao checkpoints.

    How it works:
    1. `super().create_weights(...)` creates layer.w13_weight,
       layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale as plain
       tensors.
    2. `self.create_weights(...)` temporarily changes layer.w13_weight and
       layer.w2_weight to be `TorchAOBaseTensor` objects to match the
       checkpoint format
    3. the weights are loaded (model-specific code), torchao checkpoint weights
       are properly loaded to layer.w13_weight and layer.w2_weight
    4. `self.process_weights_after_loading(...)` changes layer.w13_weight and
       layer.w2_weight back to plain tensors, and properly stashes `qdata` and
       `scale` to their plain tensor locations (matching where compressed-tensors
       path expects them).
    5. `super().process_weights_after_loading(...)` proceeds as usual
    """

    torchao_config: Any
    saved_extra_weight_attrs: Any

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w13_weight = torchao_quantize_param_data(w13_weight, self.torchao_config)
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
        w2_weight = torchao_quantize_param_data(w2_weight, self.torchao_config)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # save this so we can apply it again in process_weights_after_loading
        self.saved_extra_weight_attrs = extra_weight_attrs

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerRow,
        )

        # convert from TorchAOBaseTensor (in torchao checkpoint format)
        # to plain tensors (in compressed-tensors format).
        if isinstance(
            self.torchao_config, Float8DynamicActivationFloat8WeightConfig
        ) and self.torchao_config.granularity == [PerRow(), PerRow()]:
            # float8 per-token activation, float8 per-channel weight
            layer.w13_weight_scale = torch.nn.Parameter(
                layer.w13_weight.scale, requires_grad=False
            )
            set_weight_attrs(layer.w13_weight_scale, self.saved_extra_weight_attrs)

            layer.w13_weight = torch.nn.Parameter(
                layer.w13_weight.qdata, requires_grad=False
            )
            set_weight_attrs(layer.w13_weight, self.saved_extra_weight_attrs)

            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight.scale, requires_grad=False
            )
            set_weight_attrs(layer.w2_weight_scale, self.saved_extra_weight_attrs)

            layer.w2_weight = torch.nn.Parameter(
                layer.w2_weight.qdata, requires_grad=False
            )
            set_weight_attrs(layer.w2_weight, self.saved_extra_weight_attrs)

        else:
            raise AssertionError(f"unsupported torchao config {self.torchao_config}")

        # After this point there are no more TorchAOBaseTensor subclasses,
        # we only have plain tensors, and vLLM owns the kernels which are
        # going through the unmofidied compressed-tensors path.
        super().process_weights_after_loading(layer)


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(
        self,
        torchao_config,
        skip_modules: list[str] | None = None,
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

            # Get the torchao config for all of the MoE weights.
            first_config = maybe_get_torchao_config_for_moe_layer(
                self.torchao_config,
                module_fqn,
            )
            assert first_config is not None, "unsupported"

            # map from torchao config format to compressed-tensors config format
            compressed_tensors_quant_config = (
                torchao_config_to_compressed_tensors_config(first_config)
            )
            new_method = TorchAOWrappingCompressedTensorsW8A8Fp8MoEMethod(
                compressed_tensors_quant_config, layer.moe_config
            )
            logger.info(
                "FusedMoE quant: delegating to %s for fqn %s",
                type(new_method),
                module_fqn,
            )

            new_method.torchao_config = first_config
            return new_method

        if not isinstance(layer, LinearBase):
            return None

        if should_skip(prefix, self.skip_modules):
            return UnquantizedLinearMethod()

        module_fqn = prefix
        if isinstance(self.torchao_config, ModuleFqnToConfig):
            module_fqn_to_config = self.torchao_config.module_fqn_to_config
            c = None
            if module_fqn in module_fqn_to_config:
                assert not module_fqn.startswith("re:"), (
                    "module fqn should not start with"
                    "`re:`, which is used for specifying regex"
                )
                c = module_fqn_to_config[module_fqn]
            else:
                for maybe_module_fqn_pattern in module_fqn_to_config:
                    if not maybe_module_fqn_pattern.startswith("re:"):
                        continue
                    elif re.fullmatch(maybe_module_fqn_pattern[3:], module_fqn):
                        # we'll apply the config for first fully matched pattern
                        c = module_fqn_to_config[maybe_module_fqn_pattern]
                        break
                else:
                    # fallback to use default if no module specific
                    # config is provided
                    c = module_fqn_to_config.get("_default", None)

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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.is_checkpoint_torchao_serialized:
            if not hasattr(layer, "weight"):
                return

            # record attributes attached to the weight, so we can
            # recover later
            recorded_weight_attr = _get_weight_attrs(layer.weight)

            layer.weight = Parameter(
                convert_to_packed_tensor_based_on_current_hardware(layer.weight),
                requires_grad=layer.weight.requires_grad,
            )

            _restore_weight_attrs(layer.weight, recorded_weight_attr)
            return

        # online quantize the weight if the checkpoint is not already
        # quantized by torchao
        recorded_weight_attr = _get_weight_attrs(layer.weight)

        weight = torchao_quantize_param_data(
            layer.weight, self.quant_config.torchao_config
        )
        weight = torch.nn.Parameter(
            convert_to_packed_tensor_based_on_current_hardware(weight),
            weight.requires_grad,
        )

        _restore_weight_attrs(weight, recorded_weight_attr)
        layer.register_parameter("weight", weight)
