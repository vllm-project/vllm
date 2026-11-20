# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import json
import types
from importlib.util import find_spec
from typing import Any

import regex as re
import torch
import torch.nn.functional as F
from packaging import version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
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
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

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
    """Robust skipping logic:
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


if torchao_version_at_least("0.15.0"):
    from torchao.prototype.tensor_conversion.api import (
        convert_to_packed_tensor_based_on_current_hardware,
    )
else:
    convert_to_packed_tensor_based_on_current_hardware = lambda t: t


def _check_torchao_fp8_activation_capability(torchao_config) -> None:
    """Check if the current GPU supports FP8 activation quantization.

    FP8 activation configs (e.g., Float8DynamicActivationFloat8WeightConfig)
    require GPU compute capability >= 8.9 (Ada Lovelace / Hopper) on NVIDIA,
    or MI300+ on AMD. This check provides a clear error message before
    torchao's internal assertion fires with a confusing message.
    """
    config_name = type(torchao_config).__name__
    if "Float8" not in config_name or "Activation" not in config_name:
        return

    from vllm.platforms import current_platform

    if current_platform.supports_fp8():
        return

    capability = current_platform.get_device_capability()
    capability_str = (
        f" (current GPU compute capability: {capability.major}.{capability.minor})"
        if capability is not None
        else ""
    )
    raise ValueError(
        f"torchao FP8 activation quantization config '{config_name}' "
        f"requires GPU compute capability >= 8.9 (e.g., NVIDIA Ada Lovelace "
        f"/ Hopper or AMD MI300+){capability_str}. "
        f"For older GPUs, consider using a non-FP8 config such as "
        f"Int8WeightOnlyConfig or Int4WeightOnlyConfig."
    )


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(
        self,
        torchao_config,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
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
        """Torchao doesn't require additional config files, we use
        `config.json` from huggingface: `model_config.hf_config`
        """
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchAOConfig":
        """Create the quant config from an hf model config."""
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
        """Initialize class from a config_dict json string, got from
        torchao_config_object = some AOBaseConfig object
        json.dumps(config_to_dict(torchao_config_object))
        """
        config_dict = json.loads(config_dict_json)
        hf_config = {"quant_type": {"default": config_dict}}
        return cls.from_config(hf_config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if not isinstance(layer, LinearBase):
            return None

        from torchao.quantization import ModuleFqnToConfig

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


def torchao_quantize_param_data(
    param: torch.Tensor, torchao_config: Any
) -> torch.nn.Parameter:
    """Quantize a Tensor with torchao quantization specified by torchao_config.

    Args:
        param: weight parameter of the linear module
        torchao_config: type of quantization and their arguments we want to
            use to quantize the Tensor

    """
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_

    assert isinstance(torchao_config, AOBaseConfig), f"{torchao_config}"
    _check_torchao_fp8_activation_capability(torchao_config)
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


def _int4_tile_packed_to_humming(
    weight: torch.Tensor, params_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Unpack a torchao `Int4TilePackedTo4dTensor` (tinygemm layout) into Humming's
    packed uint4 weight `[N, K // 8]`, group scale and float zero point
    `[N, K // group_size]`, or None if the layout has no Humming equivalent.

    tinygemm dequantizes `(q - 8) * scale + zero` and Humming dequantizes
    `(q - zero_point) * scale`, so `zero_point = 8 - zero / scale`.
    """
    from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_cols

    n, k = weight.shape
    group_size = weight.block_size[-1]
    if weight.block_size[0] != 1 or k % group_size or weight.act_pre_scale is not None:
        return None
    num_groups = k // group_size
    scale, zero = (
        t.squeeze(-1)[:n, :num_groups].float()
        for t in unpack_tinygemm_scales_and_zeros(weight.scale_and_zero)
    )

    # tinygemm has no unpack op. Running it on identity rows with unit scales
    # and zero zeros returns q - 8, which bf16 holds exactly.
    unit_scale_and_zero = torch.zeros_like(weight.scale_and_zero)
    unit_scale_and_zero[..., 0] = 1
    padded_k = weight.scale_and_zero.shape[0] * group_size
    device = weight.qdata.device
    q = torch.empty(n, k, dtype=torch.int32, device=device)
    chunk = group_size * max(1, 4096 // group_size)
    for k0 in range(0, k, chunk):
        k1 = min(k0 + chunk, k)
        rows = torch.arange(k1 - k0, device=device)
        eye = torch.zeros(k1 - k0, padded_k, dtype=torch.bfloat16, device=device)
        eye[rows, rows + k0] = 1
        out = torch.ops.aten._weight_int4pack_mm(
            eye, weight.qdata, group_size, unit_scale_and_zero
        )
        q[:, k0:k1] = out[:, :n].T.to(torch.int32) + 8

    zero_point = 8 - zero / scale
    return (
        pack_cols(q, 4, n, k),
        scale.to(params_dtype),
        zero_point.to(params_dtype),
    )


def _int4_plain_to_humming(
    weight: torch.Tensor, params_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Convert a torchao `Int4Tensor` (plain layout) to Humming's packed uint4
    weight, group scale and float zero point, or None if it has no Humming
    equivalent.

    `Int4Tensor` packs signed int4 two per byte (even k in the low nibble) and
    dequantizes `q * scale + zero` with `scale` and `zero` of shape
    `[K // group_size, N]`; with `u = q + 8` that is `(u - 8) * scale + zero`.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_cols

    n, k = weight.shape
    group_size = weight.block_size[-1]
    if (
        weight.qdata.dim() != 2
        or weight.block_size[0] != 1
        or k % group_size
        or weight.act_pre_scale is not None
    ):
        return None
    packed = weight.qdata.to(torch.int32) & 0xFF
    q = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(n, k) ^ 8
    scale = weight.scale.t().float()
    zero = weight.zero_point.t().float()
    zero_point = 8 - zero / scale
    return (
        pack_cols(q, 4, n, k),
        scale.to(params_dtype),
        zero_point.to(params_dtype),
    )


def _humming_supports_layer(weight: torch.Tensor) -> bool:
    if not (current_platform.is_cuda() and has_humming()):
        return False
    from vllm.model_executor.kernels.linear.mixed_precision.humming import (
        HummingLinearKernel,
    )

    if not current_platform.has_device_capability(
        HummingLinearKernel.get_min_capability()
    ):
        return False
    try:
        from torchao.quantization import Int4Tensor, Int4TilePackedTo4dTensor
    except ImportError:
        return False
    if isinstance(weight, Int4Tensor):
        return weight.activation_dtype == torch.bfloat16
    return isinstance(weight, Int4TilePackedTo4dTensor)


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
        humming_layer_config = getattr(layer, "humming_layer_config", None)
        if humming_layer_config is not None:
            from vllm.model_executor.layers.quantization.utils.humming import (
                apply_humming_linear,
            )

            return apply_humming_linear(
                layer,
                x,
                skip_bias_add=bias is None,
                layer_config=humming_layer_config,
                compute_config=layer.humming_compute_config,
                locks=layer.humming_locks,
            )
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
            self._maybe_use_humming(layer)
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
        self._maybe_use_humming(layer)

    @staticmethod
    def _maybe_use_humming(layer: torch.nn.Module) -> None:
        """Run torchao int4 weights on Humming. tinygemm (`Int4TilePackedTo4dTensor`)
        is tuned for batch size 1, and the mslk kernel of `Int4Tensor` needs SM90,
        where torchao repacks it before this runs."""
        if not _humming_supports_layer(layer.weight):
            return
        from torchao.quantization import Int4Tensor

        group_size = layer.weight.block_size[-1]
        to_humming = (
            _int4_plain_to_humming
            if isinstance(layer.weight, Int4Tensor)
            else _int4_tile_packed_to_humming
        )
        tensors = to_humming(layer.weight, layer.params_dtype)
        if tensors is None:
            return
        from vllm.model_executor.layers.quantization.utils.humming import (
            get_humming_linear_compute_config,
            prepare_humming_linear_layer_config,
        )
        from vllm.utils.humming import create_humming_locks

        weight, weight_scale, zero_point = tensors
        layer.weight = Parameter(weight, requires_grad=False)
        layer.register_parameter("weight_scale", Parameter(weight_scale, False))
        layer.register_parameter("zero_point", Parameter(zero_point, False))
        layer.humming_layer_config = prepare_humming_linear_layer_config(
            layer,
            {
                "quant_method": "humming",
                "dtype": "uint4",
                "group_size": group_size,
                "has_zero_point": True,
                "is_fp_zero_point": True,
            },
        )
        layer.humming_compute_config = get_humming_linear_compute_config()
        layer.humming_locks = create_humming_locks(weight.device)
