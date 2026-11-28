# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/` and
`amd/` (ROCm); configs, processing, quantization metadata and the DSpark
speculator are shared."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel, Inkling2MarkovHead
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration
    from .nvidia.ops.attention import (
        block_decode_attention,
        gate_sink,
        rel_decode_attention,
    )
    from .nvidia.ops.attn_res import attn_res
    from .nvidia.ops.dspark import add_softcap, markov_gate, mix_rmsnorm
    from .nvidia.ops.norm import rms_norm
    from .nvidia.ops.sconv import attention_prologue, shift_rel_proj, short_conv
    from .nvidia.ops.vision import gather_patches
    from .nvidia.vision import Inkling2VisionEncoder

# The module of each export in the platform's package.
_MODULES = {
    "Inkling2DSparkModel": "dspark",
    "Inkling2ForConditionalGeneration": "model",
    "Inkling2LoopForConditionalGeneration": "loop",
    "Inkling2MarkovHead": "dspark",
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
# ROCm has no looped model: `amd/model.py` names the loop architecture and
# rejects looped checkpoints.
_AMD_MODULES = {**_MODULES, "Inkling2LoopForConditionalGeneration": "model"}

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2MarkovHead",
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
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            module = f"{__name__}.amd.{_AMD_MODULES[name]}"
        else:
            module = f"{__name__}.nvidia.{_MODULES[name]}"
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
