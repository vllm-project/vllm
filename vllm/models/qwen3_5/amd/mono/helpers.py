# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The three helpers the dense kernels take from the MoE mono layer.

Extracted from SGLang ``qwen3_5_mono_flydsl/layer.py`` so this port does not
vendor the 2.4T MoE K2.
"""

import math

import flydsl.expr as fx

from vllm.models.qwen3_5.amd.mono.common.ops import hw_exp2, ld_i32, uniform

LOG2E = 1.0 / math.log(2.0)
TAG_SLOTS = 256


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch and the launch's slot."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def pow2(code):
    """The fp32 power of two an e8m0 code stands for."""
    return (code << 23).bitcast(fx.Float32)
