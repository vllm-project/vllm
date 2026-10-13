# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scaffolding for MonoKernels: persistent single-launch decode layers.

Four pieces, in the order a model meets them:

- :class:`MonoSpec` says what the kernel can run, and refuses a configuration
  before anything is allocated.
- :class:`MonoRuntime` owns the per-rank resources the persistent launch waits on
  and takes the one go / no-go decision each step.
- :class:`MonoOp` is the call that stands in for vLLM's path, reached from
  :func:`mono_layer` or from the :class:`SupportsMonoKernel` module it binds to.
- :class:`MonoKernels` is the launch itself, the one piece a new model writes
  from scratch.
"""

from vllm.models.common.mono.kernels import MonoKernels
from vllm.models.common.mono.op import (
    MonoOp,
    SupportsMonoKernel,
    active_mono_layer_op,
    mono_layer,
    register_mono_layer_op,
)
from vllm.models.common.mono.runtime import MonoRuntime, StepDecision
from vllm.models.common.mono.spec import Constraint, Feature, MonoSpec
from vllm.models.common.mono.weights import check_tensors

__all__ = [
    "Constraint",
    "Feature",
    "MonoKernels",
    "MonoOp",
    "MonoRuntime",
    "MonoSpec",
    "StepDecision",
    "SupportsMonoKernel",
    "active_mono_layer_op",
    "check_tensors",
    "mono_layer",
    "register_mono_layer_op",
]
