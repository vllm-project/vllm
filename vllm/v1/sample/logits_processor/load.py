# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import itertools
import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.impls import (LogitBiasLogitsProcessor,
                                                   MinPLogitsProcessor,
                                                   MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.state import LogitsProcessors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

LOGITSPROCS_GROUP = 'vllm.logits_processors'

_builtin_logitsprocs_classes: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


def _load_logitsprocs_by_fqcns(
    logits_processors: Optional[list[Union[str, type[LogitsProcessor]]]]
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

    logger.info(
        "%s additional custom logits processors specified, checking whether "
        "they need to be loaded.", len(logits_processors))

    classes: list[type[LogitsProcessor]] = []
    for ldx, logitproc in enumerate(logits_processors):
        if isinstance(logitproc, type):
            logger.info(" - Already loaded logit processor: %s",
                        logitproc.__name__)
            if not issubclass(logitproc, LogitsProcessor):
                raise ValueError(
                    f"{logitproc.__name__} is not a subclass of LogitsProcessor"
                )
            classes.append(logitproc)
            continue

        logger.info("- Loading logits processor %s", logitproc)
        try:
            module_path, qualname = logitproc.split(":")
            # Load module
            module = importlib.import_module(module_path)
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
        except Exception as e:
            logger.exception("Failed to load %sth logits processor %s", ldx,
                             logitproc)
            raise e

    return classes


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
    logger.info("Loading installed logitsprocs plugins (group %s):",
                LOGITSPROCS_GROUP)
    classes: list[type[LogitsProcessor]] = []
    for entrypoint in installed_logitsprocs_plugins:
        try:
            logger.info("- Loading logitproc plugin entrypoint=%s target=%s",
                        entrypoint.name, entrypoint.value)
            classes.append(entrypoint.load())
        except Exception as e:
            logger.exception("Failed to load plugin %s", entrypoint)
            raise e
    return classes


def load_custom_logitsprocs(
    logits_processors: Optional[list[Union[str, type[LogitsProcessor]]]],
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


def build_logitsprocs(vllm_config: "VllmConfig", device: torch.device,
                      is_pin_memory: bool) -> LogitsProcessors:
    custom_logitsprocs_classes = vllm_config.logits_processors or []
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory) for ctor in itertools.chain(
            _builtin_logitsprocs_classes, custom_logitsprocs_classes))
