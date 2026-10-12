# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hooks a model calls: ``attach`` after load, ``forward_layer`` in its
layer loop, ``health`` from ``step_health``."""

from typing import TYPE_CHECKING

import torch

from vllm.forward_context import get_forward_context
from vllm.models.common.mono.guards import enabled, load_spec
from vllm.models.common.mono.routing import Mono

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def attach(model, vllm_config: "VllmConfig", spec_path: str) -> Mono | None:
    """Packs weights and builds this rank's layer kernels; None with the switch
    off. Collective over the TP group."""
    if not enabled(vllm_config):
        return None
    return Mono(load_spec(spec_path), model, vllm_config)


def begin_step(mono: Mono, positions: torch.Tensor) -> str | None:
    return mono.begin_step(positions)


def forward_layer(mono: Mono | None, i: int, layer, positions, *args, **kwargs):
    """Layer ``i`` on its kernel when this step runs on the kernels, else the
    vLLM layer unchanged. The first call under a forward context decides for
    every layer (``begin_step``)."""
    if mono is not None:
        fc = get_forward_context()
        if fc is not mono._fc:
            mono._fc = fc
            begin_step(mono, positions)
        step = mono.step
        if step is not None and (k := mono.kernels.get((i, step.width))) is not None:
            return k.launch(step, positions, *args, **kwargs)
    return layer(positions, *args, **kwargs)


def health(mono: Mono | None) -> torch.Tensor | None:
    """The step's error words, read by the runner after each step: any set
    word fails the step before its tokens are returned (``failstop``)."""
    return None if mono is None else mono.health()
