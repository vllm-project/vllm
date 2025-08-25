# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import (LogitBiasLogitsProcessor,
                                                     MinPLogitsProcessor,
                                                     MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       LogitsProcessor,
                                                       MoveDirectionality)
from vllm.v1.sample.logits_processor.state import (BatchUpdateBuilder,
                                                   LogitsProcessors)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Error message when the user tries to initialize vLLM with a pooling model
# and custom logitsproces
STR_POOLING_REJECTS_LOGITSPROCS = ("Pooling models do not support custom"
                                   " logits processors.")

LOGITSPROCS_GROUP = 'vllm.logits_processors'

BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


def _load_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    """Load all installed logit processor plugins"""

    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    installed_logitsprocs_plugins = entry_points(group=LOGITSPROCS_GROUP)
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).",
                     LOGITSPROCS_GROUP)
        return []

    # Load logitsprocs plugins
    logger.debug("Loading installed logitsprocs plugins (group %s):",
                 LOGITSPROCS_GROUP)
    classes: list[type[LogitsProcessor]] = []
    for entrypoint in installed_logitsprocs_plugins:
        try:
            logger.debug("- Loading logitproc plugin entrypoint=%s target=%s",
                         entrypoint.name, entrypoint.value)
            classes.append(entrypoint.load())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LogitsProcessor plugin {entrypoint}") from e
    return classes


def _load_logitsprocs_by_fqcns(
    logits_processors: Optional[Sequence[Union[str, type[LogitsProcessor]]]]
) -> list[type[LogitsProcessor]]:
    """Load logit processor types, identifying them by fully-qualified class
    names (FQCNs).

    Effectively, a mixed list of logitproc types and FQCN strings is converted
    into a list of entirely logitproc types, by loading from the FQCNs.

    FQCN syntax is <module>:<type> i.e. x.y.z:CustomLogitProc

    Already-loaded logitproc types must be subclasses of LogitsProcessor

    Args:
      logits_processors: Potentially mixed list of logitsprocs types and FQCN
                         strings for logitproc types

    Returns:
      List of logitproc types

    """
    if not logits_processors:
        return []

    logger.debug(
        "%s additional custom logits processors specified, checking whether "
        "they need to be loaded.", len(logits_processors))

    classes: list[type[LogitsProcessor]] = []
    for ldx, logitproc in enumerate(logits_processors):
        if isinstance(logitproc, type):
            logger.debug(" - Already-loaded logit processor: %s",
                         logitproc.__name__)
            if not issubclass(logitproc, LogitsProcessor):
                raise ValueError(
                    f"{logitproc.__name__} is not a subclass of LogitsProcessor"
                )
            classes.append(logitproc)
            continue

        logger.debug("- Loading logits processor %s", logitproc)
        module_path, qualname = logitproc.split(":")

        try:
            # Load module
            module = importlib.import_module(module_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {ldx}th LogitsProcessor plugin {logitproc}"
            ) from e

        # Walk down dotted name to get logitproc class
        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if not isinstance(obj, type):
            raise ValueError("Loaded logit processor must be a type.")
        if not issubclass(obj, LogitsProcessor):
            raise ValueError(
                f"{obj.__name__} must be a subclass of LogitsProcessor")
        classes.append(obj)

    return classes


def _load_custom_logitsprocs(
    logits_processors: Optional[Sequence[Union[str, type[LogitsProcessor]]]],
) -> list[type[LogitsProcessor]]:
    """Load all custom logits processors.

    * First load all installed logitproc plugins
    * Second load custom logitsprocs pass by the user at initialization time

    Args:
      logits_processors: potentially mixed list of logitproc types and
                         logitproc type fully-qualified names (FQCNs)
                         which need to be loaded

    Returns:
      A list of all loaded logitproc types
    """
    from vllm.platforms import current_platform
    if current_platform.is_tpu():
        # No logitsprocs specified by caller
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return []

    return (_load_logitsprocs_plugins() +
            _load_logitsprocs_by_fqcns(logits_processors))


def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug("Skipping logits processor loading because pooling models"
                     " do not support logits processors.")
        return LogitsProcessors()
    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory) for ctor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes))


__all__ = [
    "LogitsProcessor", "LogitBiasLogitsProcessor", "MinPLogitsProcessor",
    "MinTokensLogitsProcessor", "BatchUpdate", "BatchUpdateBuilder",
    "MoveDirectionality", "LogitsProcessors", "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS", "LOGITSPROCS_GROUP"
]
