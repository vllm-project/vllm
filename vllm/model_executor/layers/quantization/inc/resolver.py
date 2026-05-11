# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re

from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

if TYPE_CHECKING:
    import torch

    from .inc import INCConfig


@dataclass(frozen=True)
class INCLayerConfig:
    bits: int
    group_size: int
    sym: bool
    packing_format: str
    backend: str
    data_type: str
    quantized: bool

    @property
    def is_gptq(self) -> bool:
        return "gptq" in self.packing_format or "gptq" in self.backend

    @property
    def is_awq(self) -> bool:
        return "awq" in self.packing_format or "awq" in self.backend

    @property
    def is_wna16_int(self) -> bool:
        return self.data_type == "int" and self.quantized

    @property
    def is_mxfp4(self) -> bool:
        return self.data_type == "mx_fp" and self.bits == 4

    @property
    def is_mxfp8(self) -> bool:
        return self.data_type == "mx_fp" and self.bits == 8


class INCConfigResolver:
    def __init__(self, config: "INCConfig") -> None:
        self._config = config

    def resolve(self, layer: "torch.nn.Module", layer_name: str) -> INCLayerConfig:
        bits, group_size, sym = self._resolve_raw(layer, layer_name)
        return INCLayerConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            packing_format=self._config.packing_format,
            backend=self._config.backend,
            data_type=self._config.data_type,
            quantized=bits < 16,
        )

    def get_layer_config(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int, bool]:
        layer_config = self.resolve(layer, layer_name)
        return layer_config.bits, layer_config.group_size, layer_config.sym

    def _resolve_raw(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int, bool]:
        def get_config(name: str, quantized: bool = True) -> tuple[int, int, bool]:
            if not self._config.extra_config:
                return (
                    self._config.weight_bits if quantized else 16,
                    self._config.group_size if quantized else -1,
                    self._config.sym if quantized else True,
                )

            if name in self._config.extra_config:
                cfg = self._config.extra_config[name]
                return (
                    cfg.get("bits", self._config.weight_bits if quantized else 16),
                    cfg.get(
                        "group_size",
                        self._config.group_size if quantized else -1,
                    ),
                    cfg.get("sym", self._config.sym if quantized else True),
                )

            regex_special_chars = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self._config.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in regex_special_chars for c in pattern
                ):
                    continue

                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return (
                            cfg.get(
                                "bits",
                                self._config.weight_bits if quantized else 16,
                            ),
                            cfg.get(
                                "group_size",
                                self._config.group_size if quantized else -1,
                            ),
                            cfg.get("sym", self._config.sym if quantized else True),
                        )
                except re.error:
                    continue

            return (
                self._config.weight_bits if quantized else 16,
                self._config.group_size if quantized else -1,
                self._config.sym if quantized else True,
            )

        if self._config.extra_config and layer_name in self._config.extra_config:
            return get_config(layer_name)

        quantized = not isinstance(layer, ParallelLMHead)
        if self._config.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name)
                for name in self._config.block_name_to_quantize
            )

        if self._config.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self._config.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        if self._config.extra_config:
            for fusion_key, sub_keys in self._config.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        return get_config(layer_name, quantized)
