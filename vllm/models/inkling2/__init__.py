# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .dspark import Inkling2DSparkModel, Inkling2MarkovHead
    from .model import Inkling2ForConditionalGeneration
    from .ops.attention import (
        block_decode_attention,
        gate_sink,
        rel_decode_attention,
    )
    from .ops.attn_res import attn_res
    from .ops.dspark import add_softcap, markov_gate, mix_rmsnorm
    from .ops.norm import rms_norm
    from .ops.sconv import attention_prologue, shift_rel_proj, short_conv
    from .ops.vision import gather_patches
    from .quant_config import Inkling2QuantConfig
    from .vision import Inkling2VisionEncoder

# The platform's module of each export, under `amd/` on ROCm.
_MODULES = {
    "Inkling2DSparkModel": "dspark",
    "Inkling2ForConditionalGeneration": "model",
    "Inkling2MarkovHead": "dspark",
    "Inkling2QuantConfig": "quant_config",
    "Inkling2VisionEncoder": "vision",
    "add_softcap": "ops.dspark",
    "attention_prologue": "ops.sconv",
    "attn_res": "ops.attn_res",
    "block_decode_attention": "ops.attention",
    "gate_sink": "ops.attention",
    "gather_patches": "ops.vision",
    "markov_gate": "ops.dspark",
    "mix_rmsnorm": "ops.dspark",
    "rel_decode_attention": "ops.attention",
    "rms_norm": "ops.norm",
    "shift_rel_proj": "ops.sconv",
    "short_conv": "ops.sconv",
}

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2MarkovHead",
    "Inkling2QuantConfig",
    "Inkling2VisionEncoder",
    "add_softcap",
    "attention_prologue",
    "attn_res",
    "block_decode_attention",
    "gate_sink",
    "gather_patches",
    "markov_gate",
    "mix_rmsnorm",
    "rel_decode_attention",
    "rms_norm",
    "shift_rel_proj",
    "short_conv",
]


def __getattr__(name: str):
    if name in _MODULES:
        platform = ".amd" if current_platform.is_rocm() else ""
        module = import_module(f"{__name__}{platform}.{_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
