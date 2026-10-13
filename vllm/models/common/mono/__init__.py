# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): persistent kernels that run most or all of one
decoder layer's decode step in a single launch. A model brings a ``MonoSpec``
(``spec``); this package owns the rest: refusals and TP agreement (``guards``),
the per-step decision, cache re-binding and step counters (``runtime``), and the
layer-loop hook (``layer``).

Peer memory and in-kernel failure reporting are the model's own: only its
kernels know what they put where and what they record, so a model that needs
either builds it in the launch state it brings."""

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
