# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import ModuleType
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_HW_AGNOSTIC_PKG = "vllm.model_executor.hw_agnostic.layers"
_IN_TREE_PKG = "vllm.model_executor.layers"


def _import_hw_agnostic(module: str) -> ModuleType | None:
    """Import `hw_agnostic.layers.<module>`, or return None if there is none.

    Only a missing hw-agnostic module counts as absent; any other import error,
    e.g. a missing dependency of an existing port, propagates.
    """
    name = f"{_HW_AGNOSTIC_PKG}.{module}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name is not None and (name == e.name or name.startswith(f"{e.name}.")):
            return None
        raise


def resolve(module: str, name: str) -> Any:
    """Return layer `name` from `module` for the active path.

    Returns `vllm.model_executor.hw_agnostic.layers.<module>.<name>` if
    `VLLM_USE_HW_AGNOSTIC` is set and that layer exists, else
    `vllm.model_executor.layers.<module>.<name>`. Falling back to the in-tree
    layer while `VLLM_USE_HW_AGNOSTIC` is set logs a warning.

    The Transformers backend builds its layers through this function, so an
    out-of-tree plugin that subclasses the result and registers it with its
    `register_oot` overrides the class that backend builds on either path.
    Models with a native vLLM implementation always build the in-tree class, and
    overrides do not yet reach the backend's norms, which are subclasses of
    `RMSNorm` and `GemmaRMSNorm`.

    Example:
        SiluAndMul = hw_agnostic.resolve("activation", "SiluAndMul")

        @SiluAndMul.register_oot
        class MySiluAndMul(SiluAndMul): ...

    Args:
        module: Module path relative to `vllm.model_executor.layers`, e.g.
            `"layernorm"`.
        name: Name of the layer in that module, e.g. `"RMSNorm"`.

    Returns:
        The hw-agnostic or the in-tree object named `name`.

    Raises:
        AttributeError: `name` exists in neither module.
        ImportError: The hw-agnostic module exists but fails to import, or the
            in-tree module does not exist.

    """
    if envs.VLLM_USE_HW_AGNOSTIC:
        hw_module = _import_hw_agnostic(module)
        if (layer := getattr(hw_module, name, None)) is not None:
            logger.info_once("Using hardware agnostic layer %s.%s", module, name)
            return layer
        logger.warning_once(
            "hw-agnostic layer %s.%s is not available; using the in-tree layer",
            module,
            name,
        )
    return getattr(importlib.import_module(f"{_IN_TREE_PKG}.{module}"), name)
