# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): persistent kernels that run most or all of one
decoder layer's decode step in a single launch. A model brings a ``MonoSpec``
(``spec``); this package owns the rest: refusals (``guards``), the per-step
decision and cache re-binding (``routing``), the layer-loop hook (``layer``),
fail-stop and metrics (``failstop``), peer memory (``peer``) and the
error-word layout (``abi``)."""

from vllm.models.common.mono.layer import attach, forward_layer, health

__all__ = ["attach", "forward_layer", "health"]
