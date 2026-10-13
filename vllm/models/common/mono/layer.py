# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hooks a model calls: ``attach`` after load, ``forward_layer`` in its
layer loop, ``health`` from ``step_health``."""

from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.mono import failstop
from vllm.models.common.mono.guards import enabled, load_specs, refused, select
from vllm.models.common.mono.routing import Mono
from vllm.models.common.mono.spec import MonoUnsupported

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_ATTACHED: Mono | None = None


def attach(model, vllm_config: "VllmConfig", specs_path: str) -> Mono | None:
    """Picks the variant, packs the weights and builds this rank's layer
    kernels. Collective over the TP group.

    Args:
        model: The loaded model, in ``process_weights_after_loading``.
        vllm_config: The engine's config.
        specs_path: ``"package.module:MONO_SPECS"``.

    Returns:
        This rank's ``Mono``, or None with the switch off or when no layer
        takes a kernel.

    Raises:
        MonoUnsupported: Every variant refused the config, or this rank (or
            another) refused the loaded model.

    """
    global _ATTACHED
    if not enabled(vllm_config):
        return None
    spec, refusals = select(load_specs(specs_path), vllm_config)
    if spec is None:
        raise MonoUnsupported("MonoKernel refused: " + refused(refusals))
    mono = Mono(spec, model, vllm_config)
    if not mono.kernels:
        logger.info("MonoKernel %s: no layer takes a kernel", spec.name)
        return None
    if not hasattr(model, "step_health"):
        logger.warning_once(
            "MonoKernel %s: the model exposes no step_health(), so the runner "
            "cannot fail a step whose in-kernel wait gave up",
            spec.name,
        )
    _ATTACHED = mono
    return mono


def forward_layer(mono: Mono | None, i: int, layer, positions, *args, **kwargs):
    """Layer ``i`` on its kernel when this step runs on the kernels, else the
    vLLM layer unchanged. The first call under a forward context decides for
    every layer."""
    if mono is not None:
        mono.begin_once(positions)
        step = mono.step
        if step is not None and (k := mono.kernels.get((i, step.width))) is not None:
            return k.launch(step, positions, *args, **kwargs)
    return layer(positions, *args, **kwargs)


def health(mono: Mono | None) -> torch.Tensor | None:
    """The step's error words, read by the runner after each step: any set
    word fails the step before its tokens are returned (``failstop``)."""
    return None if mono is None else mono.health()


def account_step(num_tokens: int, runtime_mode: CUDAGraphMode) -> dict[str, int] | None:
    """Called by the runner after every served step: counts it, and returns the
    counters to export since the last call (``failstop.drain``)."""
    if _ATTACHED is None:
        return None
    _ATTACHED.account(num_tokens, runtime_mode)
    return failstop.drain()
