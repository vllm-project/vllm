# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Agnes 3.0 model compatible with HuggingFace weights.

Agnes is Qwen3.5 with a parallel FFN branch beside each layer's MLP. Layer
plan, attention, GDN projections, MTP block and vision tower are identical
tensor for tensor, so everything here rides on `qwen3_5`: the checkpoint's
`delta_attn`/`global_attn` names are renamed onto Qwen3.5's, and the extra
branch is attached to the layers the Qwen3.5 model already built.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from .qwen3_5_mtp import Qwen3_5MTP
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from .utils import WeightsMapper, maybe_prefix

_AGNES_TO_QWEN3_5 = WeightsMapper(
    orig_to_new_substr={
        "delta_attn.": "linear_attn.",
        "global_attn.": "self_attn.",
    },
    orig_to_new_stacked={
        ".parallel_ffn.gate_proj": (".parallel_ffn.gate_up_proj", 0),
        ".parallel_ffn.up_proj": (".parallel_ffn.gate_up_proj", 1),
    },
)

# The checkpoint's GPTQ `dynamic` keys name checkpoint modules, but vLLM
# matches them against its own module paths, where `in_proj_b`/`in_proj_a` have
# become one merged `in_proj_ba`. The vision tower carries no key at all: it was
# never entered for quantization.
_UNQUANTIZED_MODULES = (r"-:.*\.in_proj_ba", r"-:.*\.visual\.")


def _skip_unquantized_modules(quant_config) -> None:
    dynamic = getattr(quant_config, "dynamic", None)
    if isinstance(dynamic, dict):
        for pattern in _UNQUANTIZED_MODULES:
            dynamic.setdefault(pattern, {})


class AgnesMLP(nn.Module):
    """Qwen3.5's SwiGLU MLP with Agnes's parallel branch summed onto it.

    Adopts the projections the layer already holds so the main branch keeps
    its `mlp.gate_up_proj` / `mlp.down_proj` names and its loaded weights.
    """

    def __init__(self, mlp: nn.Module, parallel_ffn: nn.Module) -> None:
        super().__init__()
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.parallel_ffn = parallel_ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out + self.parallel_ffn(x)


class AgnesProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        # Agnes ships its own Qwen3-VL-shaped processor as remote code, so it
        # cannot be type-checked against `Qwen3VLProcessor`.
        return self.ctx.get_hf_processor(
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=AgnesProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AgnesForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    hf_to_vllm_mapper = (
        _AGNES_TO_QWEN3_5 | Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        _skip_unquantized_modules(vllm_config.quant_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_text_config
        intermediate_size = getattr(config, "parallel_ffn_intermediate_size", 0)
        if not intermediate_size:
            return

        layers_prefix = maybe_prefix(
            maybe_prefix(prefix, "language_model"), "model.layers"
        )
        for idx, layer in enumerate(self.language_model.model.layers):
            if not isinstance(layer, Qwen3_5DecoderLayer):
                continue  # PPMissingLayer on ranks that do not hold this layer
            layer.mlp = AgnesMLP(
                layer.mlp,
                Qwen3NextMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{layers_prefix}.{idx}.mlp.parallel_ffn",
                ),
            )


class AgnesMTP(Qwen3_5MTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Every tensor of Agnes's MTP block is 16-bit, and the checkpoint has no
        # `-:` key naming it, so the quantized path would look for weights that
        # are not there.
        quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_config.quant_config = quant_config
        # The predictor loads `mtp.*` through its own mapper, which never sees
        # this model's top-level rename.
        self.model.hf_to_vllm_mapper = _AGNES_TO_QWEN3_5 | self.model.hf_to_vllm_mapper
