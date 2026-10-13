# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hooks a model calls: ``attach``, then ``kernel_for_layer`` per layer."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
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
    if not _enabled(vllm_config):
        return None
    spec, refusals = _select_spec(_load_specs(specs_path), vllm_config)
    if spec is None:
        raise MonoUnsupported("MonoKernel refused: " + _refusal_message(refusals))
    mono = MonoRuntime(spec, model, vllm_config)
    if not mono.kernels:
        logger.info("MonoKernel %s: no layer takes a kernel", spec.name)
        return None
    return mono


def kernel_for_layer(mono: MonoRuntime | None, i: int, positions) -> LayerKernel | None:
    """Layer ``i``'s kernel when this step runs on the kernels, else None.

    The first call under a forward context decides for every layer. The model
    calls ``launch`` where it suits the layer, and runs the vLLM layer
    unchanged when this returns None.
    """
    if mono is None:
        return None
    mono.begin_once(positions)
    if mono.width is None:
        return None
    return mono.kernels.get((i, mono.width))
