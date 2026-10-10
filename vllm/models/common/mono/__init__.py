# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scaffolding for MonoKernels: persistent single-launch decode layers.

Three pieces, in the order a model meets them:

- :class:`MonoSpec` says what the kernel can run, and refuses a configuration
  before anything is allocated.
- :class:`MonoRuntime` owns the per-rank resources the persistent launch waits on
  and takes the one go / no-go decision each step.
- :class:`MonoOp` is the call that stands in for vLLM's path, reached from
  :func:`mono_layer`.
"""

from vllm.models.common.mono.op import (
    MonoOp,
    active_mono_layer_op,
    mono_layer,
    register_mono_layer_op,
)
from vllm.models.common.mono.runtime import MonoRuntime, StepDecision
from vllm.models.common.mono.spec import Feature, MonoSpec
from vllm.models.common.mono.weights import check_tensors

__all__ = [
    "Feature",
    "MonoOp",
    "MonoRuntime",
    "MonoSpec",
    "StepDecision",
    "active_mono_layer_op",
    "check_tensors",
    "mono_layer",
    "register_mono_layer_op",
]
