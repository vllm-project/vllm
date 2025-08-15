from typing import Any, Dict, List, Optional

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform


class IPEXConfig(QuantizationConfig):
    """INT8 quantization config class using IPEX for the CPU backend,
    including AWQ.
    """

    IPEX_QUANT_METHOD_MAP = {
        "awq": 1,
        "gptq": 2,
    }

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
    ) -> None:
        self.method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // self.weight_bits

        if self.weight_bits not in [4]:
            raise ValueError(
                f"IPEX quantization supports weight bits [4], "
                f"but got {self.weight_bits}."
            )

        if self.method == "awq":
            self.quant_method = IPEXAWQLinearMethod
        else:
            raise ValueError(
                f"IPEX quantization supports [awq], " f"but got {self.method}."
            )

    def __repr__(self) -> str:
        return (
            f"IPEXConfig(method={self.method}"
            f"weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}"
        )

    def get_ipex_quant_method_id(self) -> int:
        return IPEXConfig.IPEX_QUANT_METHOD_MAP[self.method]

    @classmethod
    def get_name(cls) -> str:
        return "ipex"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "IPEXConfig":
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        return cls(method, weight_bits, group_size)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if not current_platform.is_cpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq"]:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            return self.quant_method(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        if self.method == "awq":
            return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]
        else:
            return []


class IPEXAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method using IPEX for the CPU backend."""

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)

        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if ipex.__version__ < "2.4.0":
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    "intel_extension_for_pytorch>=2.4.0."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                "intel_extension_for_pytorch>=2.4.0 via "
                "`pip install intel_extension_for_pytorch>=2.4.0`"
                " to use IPEX-AWQ linear method."
            ) from err

        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )

        layer.ipex_output_size = layer.qweight.size(1) * self.quant_config.pack_factor
        layer.ipex_qlinear = ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            bias=bias,
            group_size=self.quant_config.group_size,
            quant_method=self.quant_config.get_ipex_quant_method_id(),  # type: ignore
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)

        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))
