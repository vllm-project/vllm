# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Model preparation and binding for config-driven rotation transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import PPMissingLayer

from .adapter import RotationAdapter
from .config import RotationApplication, RotationConfig, parse_rotation_config
from .norm import PreRotationRMSNorm
from .registry import RotationRegistry

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


@dataclass
class PreparedRotationTransforms:
    """Per-model state produced before checkpoint loading."""

    consumers: dict[str, RotationAdapter]
    unresolved_targets: set[str]
    owns_registry: bool


def _resolve_path(root: nn.Module, path: str) -> nn.Module | None:
    try:
        return root.get_submodule(path)
    except AttributeError:
        return None


def _wrap_target(
    module: nn.Module, application: RotationApplication, target: str
) -> RotationAdapter:
    """Upgrade the target module in place for the application mode."""
    if application.mode == "pre" and isinstance(module, RMSNorm):
        return PreRotationRMSNorm.upgrade(
            module,
            tag=application.tag,
            inverse=application.inverse,
        )
    raise TypeError(
        f"rotation target {target!r} does not support mode "
        f"{application.mode!r}: {type(module).__name__}"
    )


def _install_registry(model: nn.Module, config: RotationConfig) -> None:
    if getattr(model, "rotations", None) is not None:
        raise ValueError("rotations is already registered on the target model")
    model.add_module("rotations", RotationRegistry(config.rotations))


def prepare_rotation_transforms(
    model: nn.Module, vllm_config: VllmConfig, model_config: ModelConfig
) -> None:
    """Prepare root model consumers; nested submodel views defer to the root."""
    if model_config.is_submodel_config:
        return
    config = parse_rotation_config(vllm_config.model_config.hf_config)
    if config is None:
        return
    if hasattr(model, "_rotation_transform_state"):
        raise RuntimeError("rotation transforms were already prepared")

    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    owns_registry = model_config is not draft_model_config
    if owns_registry:
        _install_registry(model, config)

    consumers: dict[str, RotationAdapter] = {}
    unresolved: set[str] = set()
    for application in config.applications:
        for target in application.targets:
            module = _resolve_path(model, target)
            if module is None:
                unresolved.add(target)
                continue
            if isinstance(module, PPMissingLayer):
                continue
            consumers[target] = _wrap_target(module, application, target)

    if owns_registry:
        dtypes = [model_config.dtype]
        if draft_model_config is not None:
            dtypes.append(draft_model_config.dtype)
        for application in config.applications:
            rotation = model.rotations[application.tag]
            for dtype in dtypes:
                rotation.register_warmup(inverse=application.inverse, dtype=dtype)

    state = PreparedRotationTransforms(
        consumers,
        unresolved,
        owns_registry,
    )
    object.__setattr__(model, "_rotation_transform_state", state)


def _bind_consumers(
    state: PreparedRotationTransforms,
    registry: RotationRegistry,
) -> set[str]:
    for consumer in state.consumers.values():
        consumer.bind_rotation(registry[consumer.rotation_tag])
    return set(state.unresolved_targets)


def bind_rotation_transforms(
    target_model: nn.Module, draft_model: nn.Module | None = None
) -> None:
    """Bind prepared target and draft consumers to the target registry."""
    state = getattr(target_model, "_rotation_transform_state", None)
    if state is None:
        return
    if not isinstance(state, PreparedRotationTransforms) or not state.owns_registry:
        raise RuntimeError("target model does not own a rotation registry")
    registry = getattr(target_model, "rotations", None)
    if not isinstance(registry, RotationRegistry):
        raise RuntimeError("target model rotation registry is missing")
    registry.validate_loaded()

    unresolved = _bind_consumers(state, registry)
    if draft_model is not None:
        draft_state = getattr(draft_model, "_rotation_transform_state", None)
        if not isinstance(draft_state, PreparedRotationTransforms):
            raise RuntimeError("rotation transforms were not prepared for draft model")
        unresolved &= _bind_consumers(draft_state, registry)
    elif unresolved:
        logger.warning_once(
            "Skipping unresolved rotation targets because no draft model is loaded: %s",
            ", ".join(sorted(unresolved)),
        )
        return
    if unresolved:
        missing = ", ".join(sorted(unresolved))
        raise ValueError(
            "configured rotation targets were not found in target/draft "
            f"models: {missing}"
        )


__all__ = ["bind_rotation_transforms", "prepare_rotation_transforms"]
