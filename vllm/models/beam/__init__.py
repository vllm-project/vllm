# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia import (
        activation,
        attention,
        attention_ops,
        dflash2_conv_ops,
        dflash2_ops,
        layernorm,
        model,
    )
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM
elif current_platform.is_rocm():
    from .amd import (
        activation,
        attention,
        attention_ops,
        dflash2_conv_ops,
        dflash2_ops,
        layernorm,
        model,
    )
    from .amd.dflash2 import DFlash2BeamForCausalLM
    from .amd.model import BeamForCausalLM
else:
    from .nvidia import (
        activation,
        attention,
        attention_ops,
        dflash2_conv_ops,
        dflash2_ops,
        layernorm,
        model,
    )
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM

beam_output_gate = attention_ops.beam_output_gate
beam_qk_norm_rope = attention_ops.beam_qk_norm_rope
beam_norm_residual_norm = layernorm.beam_norm_residual_norm
beam_rms_norm = layernorm.beam_rms_norm
beam_silu_mul = activation.beam_silu_mul
beam_softcap_logits = activation.beam_softcap_logits
candidate_edges = dflash2_ops.candidate_edges
conv_residual_norm = dflash2_conv_ops.conv_residual_norm

__all__ = [
    "BeamForCausalLM",
    "DFlash2BeamForCausalLM",
    "activation",
    "attention",
    "attention_ops",
    "beam_norm_residual_norm",
    "beam_output_gate",
    "beam_qk_norm_rope",
    "beam_rms_norm",
    "beam_silu_mul",
    "beam_softcap_logits",
    "candidate_edges",
    "conv_residual_norm",
    "dflash2_conv_ops",
    "dflash2_ops",
    "layernorm",
    "model",
]
