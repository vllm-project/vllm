# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
from typing import Any, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class AutoRoundConfig(QuantizationConfig):
    """Config class for AutoRound.
    Reference: https://arxiv.org/pdf/2309.05516
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {
        "auto", "gptq", "gptq:marlin", "awq", "awq:marlin", "marlin", "ipex"
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: Optional[Union[str, list[str]]] = None,
        extra_config: Optional[dict[str, Any]] = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(f"Unsupported weight_bits: {weight_bits}, "
                             f"currently only support  {self.SUPPORTED_BITS}")
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}")
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support  {self.SUPPORTED_FORMATS}")
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support  {self.SUPPORTED_BACKENDS}")

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (block_name_to_quantize.split(",") if
                                       isinstance(block_name_to_quantize, str)
                                       else block_name_to_quantize)
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)

    def __repr__(self) -> str:
        return (f"AutoRoundConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, sym={self.sym})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundConfig":
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(config, ["packing_format"],
                                                "auto_round:auto_gptq"),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"],
                None),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"],
                                         "auto"),
        )

    def get_layer_config(self, layer, layer_name: str):
        # Priority: extra_config > block_name_to_quantize > type fallback
        if self.extra_config and layer_name in self.extra_config:
            cfg = self.extra_config[layer_name]
            return cfg.get("bits", self.weight_bits), cfg.get(
                "group_size", self.group_size), cfg.get("sym", self.sym)

        quantized = True
        if self.block_name_to_quantize:
            quantized = any(name in layer_name
                            for name in self.block_name_to_quantize)
        elif isinstance(layer, ParallelLMHead):
            quantized = False

        return (self.weight_bits, self.group_size,
                self.sym) if quantized else (16, -1, True)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported, check_moe_marlin_supports_layer)

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug("[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
                     prefix, layer.__class__.__name__, weight_bits, group_size,
                     sym)
        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits
                          in AWQ_TYPE_MAP) and check_marlin_supported(
                              AWQ_TYPE_MAP[weight_bits], group_size, not sym)

            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size)

        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinConfig, AWQMarlinLinearMethod, AWQMoEMethod)
            quant_args_marlin = AWQMarlinConfig(weight_bits=weight_bits,
                                                group_size=group_size,
                                                zero_point=not sym,
                                                lm_head_quantized=False,
                                                full_config={},
                                                modules_to_not_convert=[])
        else:
            from vllm.model_executor.layers.quantization.awq import (
                AWQConfig, AWQLinearMethod)
            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return AWQMoEMethod(quant_args_marlin)
            from vllm.model_executor.layers.quantization.moe_wna16 import (
                MoeWNA16Config)
            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(
                layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return AWQMarlinLinearMethod(quant_args_marlin)
            else:
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self,
                               layer,
                               prefix: str,
                               backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported, check_moe_marlin_supports_layer)
        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug("[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
                     prefix, layer.__class__.__name__, weight_bits, group_size,
                     sym)
        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = ((weight_bits, sym) in GPTQ_TYPE_MAP
                          and check_marlin_supported(
                              GPTQ_TYPE_MAP[(weight_bits, sym)],
                              group_size,
                              has_zp=not sym))
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size)
        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig, GPTQMarlinLinearMethod, GPTQMarlinMoEMethod)
            quant_args_marlin = GPTQMarlinConfig(weight_bits=weight_bits,
                                                 group_size=group_size,
                                                 is_sym=sym,
                                                 lm_head_quantized=False,
                                                 desc_act=False,
                                                 dynamic={},
                                                 full_config={})
        else:
            from vllm.model_executor.layers.quantization.gptq import (
                GPTQConfig, GPTQLinearMethod)
            quant_args = GPTQConfig(weight_bits=weight_bits,
                                    group_size=group_size,
                                    lm_head_quantized=False,
                                    desc_act=False,
                                    dynamic={})

        if isinstance(layer, FusedMoE):
            if use_marlin:
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config)
                config = {
                    "quant_method": "gptq",
                    "bits": weight_bits,
                    "group_size": group_size,
                    "sym": sym,
                    "lm_head": False,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix)
            return GPTQMarlinMoEMethod(quant_args_marlin)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def apply_ipex_quant_layer(self, layer, prefix: str):
        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None
        from vllm.model_executor.layers.quantization.ipex_quant import (
            IPEXAWQLinearMethod, IPEXConfig, IPEXGPTQLinearMethod)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if "awq" in self.packing_format:
                config = IPEXConfig(method="awq",
                                    weight_bits=weight_bits,
                                    group_size=group_size)
                return IPEXAWQLinearMethod(config)
            elif "gptq" in self.packing_format:
                config = IPEXConfig(method="gptq",
                                    weight_bits=weight_bits,
                                    group_size=group_size)
                return IPEXGPTQLinearMethod(config)
            else:
                raise ValueError(
                    f"ipex backend only supports awq "
                    f"and gtpq format,but got {self.packing_format}")
        else:
            return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if (current_platform.is_cpu() or current_platform.is_xpu()
                or self.backend == "ipex"):
            return self.apply_ipex_quant_layer(layer, prefix)
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix)
