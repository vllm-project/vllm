# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): persistent kernels that run most or all of one
decoder layer's decode step in a single launch. A model brings a ``MonoSpec``
(``spec``); this package owns the rest: refusals (``guards``), the per-step
decision and cache re-binding (``runtime``), the layer-loop hook (``layer``),
fail-stop and metrics (``failstop``) and peer memory (``peer``)."""

from vllm.models.common.mono.guards import configure
from vllm.models.common.mono.layer import (
    account_step,
    attach,
    forward_layer,
    health,
    kernel_for,
)

__all__ = [
    "account_step",
    "attach",
    "configure",
    "forward_layer",
    "health",
    "kernel_for",
]
