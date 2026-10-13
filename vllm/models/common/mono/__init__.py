# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): persistent kernels that run most or all of one
decoder layer's decode step in a single launch. A model brings a ``MonoSpec``
(``spec``); this package owns the rest: refusals (``guards``), the per-step
decision, cache re-binding and step counters (``runtime``), the layer-loop hook
(``layer``) and peer memory (``peer``).

Detecting and reporting an in-kernel failure is the model's own: only its
kernels know what they record, so a model that needs fail-stop reads it from
the launch state it built."""

from vllm.models.common.mono.guards import configure
from vllm.models.common.mono.layer import (
    account_step,
    attach,
    forward_layer,
    kernel_for,
)

__all__ = [
    "account_step",
    "attach",
    "configure",
    "forward_layer",
    "kernel_for",
]
