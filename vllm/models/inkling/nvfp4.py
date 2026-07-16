# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 (ModelOpt) support for the Inkling mixture-of-experts.

Only the routed MoE experts are quantized in the Inkling checkpoint;
attention, the dense MLP, and the shared "sink" experts stay bf16 (they are
in the checkpoint ``exclude_modules``). The routed experts are served by
vLLM's standard ModelOpt NVFP4 fused-MoE stack (see ``moe.py``); this module
keeps the checkpoint detection.
"""

from __future__ import annotations

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


class InklingNvfp4Config:
    """Lightweight NVFP4 descriptor parsed from the checkpoint quant config.

    Holds the (mapped) ``exclude_modules`` so the model can decide, per MoE
    layer and per expert group, whether the weights are NVFP4 or plain bf16.
    """

    def __init__(self, group_size: int, exclude_modules: list[str]) -> None:
        self.group_size = group_size
        self.exclude_modules = set(exclude_modules)

    @staticmethod
    def _is_nvfp4(quant_cfg: dict) -> bool:
        wq = quant_cfg["modelopt_quant_config"]["quant_cfg"]["*weight_quantizer"]
        return tuple(wq["num_bits"]) == (2, 1) and tuple(
            wq["block_sizes"].get("scale_bits", [])
        ) == (4, 3)

    @classmethod
    def from_hf_config(cls, hf_config) -> InklingNvfp4Config | None:
        quant_cfg = getattr(hf_config, "quantization_config", None)
        text_config = getattr(hf_config, "text_config", None)
        if quant_cfg is None and text_config is not None:
            quant_cfg = getattr(text_config, "quantization_config", None)
        if quant_cfg is None:
            return None
        # ModelOpt <=0.29 nests everything under "quantization".
        if "quantization" in quant_cfg:
            quant_cfg = quant_cfg["quantization"]
        if not cls._is_nvfp4(quant_cfg):
            return None
        group_size = quant_cfg.get("group_size", 16)
        if group_size != 16:
            raise ValueError("Inkling NVFP4 only supports group size 16")
        exclude = list(quant_cfg.get("exclude_modules", []) or [])
        return cls(group_size=group_size, exclude_modules=exclude)

    def experts_quantized(self, layer_id: int) -> bool:
        """Whether the routed experts of ``layer_id`` are NVFP4 (vs excluded)."""
        return f"model.llm.layers.{layer_id}.mlp.experts" not in self.exclude_modules

    def shared_experts_quantized(self, layer_id: int) -> bool:
        """Whether the shared sink experts of ``layer_id`` are NVFP4."""
        return (
            f"model.llm.layers.{layer_id}.mlp.shared_experts"
            not in self.exclude_modules
        )
