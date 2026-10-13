# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hooks a model calls: ``attach`` after load, ``forward_layer`` in its
layer loop, ``health`` from ``step_health``."""

from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.mono import failstop
from vllm.models.common.mono.guards import (
    _enabled,
    _load_specs,
    _refusal_message,
    _select_spec,
)
from vllm.models.common.mono.runtime import MonoRuntime
from vllm.models.common.mono.spec import LayerKernel, MonoUnsupported

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_ATTACHED: MonoRuntime | None = None


def attach(model, vllm_config: "VllmConfig", specs_path: str) -> MonoRuntime | None:
    """Picks the variant, builds the launch state and this rank's layer
    kernels. Collective over the TP group.

    Args:
        model: The loaded model, in ``process_weights_after_loading``.
        vllm_config: The engine's config.
        specs_path: ``"package.module:MONO_SPECS"``.

    Returns:
        This rank's ``MonoRuntime``, or None with the switch off or when no layer
        takes a kernel.

    Raises:
        MonoUnsupported: Every variant refused the config, or this rank (or
            another) refused the loaded model.

    """
    global _ATTACHED
    if not _enabled(vllm_config):
        return None
    spec, refusals = _select_spec(_load_specs(specs_path), vllm_config)
    if spec is None:
        raise MonoUnsupported("MonoKernel refused: " + _refusal_message(refusals))
    mono = MonoRuntime(spec, model, vllm_config)
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


def kernel_for(mono: MonoRuntime | None, i: int, positions) -> LayerKernel | None:
    """Layer ``i``'s kernel when this step runs on the kernels, else None.

    The first call under a forward context decides for every layer. A model
    whose layer signature or dispatch site does not fit ``forward_layer`` looks
    the kernel up itself and calls ``launch`` where it suits the layer.
    """
    if mono is None:
        return None
    mono.begin_once(positions)
    step = mono.step
    if step is None:
        return None
    return mono.kernels.get((i, step.width))


def forward_layer(mono: MonoRuntime | None, i: int, layer, positions, *args, **kwargs):
    """Layer ``i`` on its kernel when this step runs on the kernels, else the
    vLLM layer unchanged."""
    k = kernel_for(mono, i, positions)
    step = None if mono is None else mono.step
    if k is not None and step is not None:
        out = k.launch(step, positions, *args, **kwargs)
        if out is not None:
            return out
    return layer(positions, *args, **kwargs)


def health(mono: MonoRuntime | None) -> torch.Tensor | None:
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
