# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import logging
from typing import Any, Callable, Optional, Union

import torch
from typing_extensions import TypedDict

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.impls import (LogitBiasLogitsProcessor,
                                                   MinPLogitsProcessor,
                                                   MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.state import LogitsProcessorManager

logger = logging.getLogger(__name__)

LOGITSPROCS_GROUP = 'vllm.logits_processors'
LogitprocCtor = Callable[[], LogitsProcessor]
logitsprocs_ctors: list[LogitprocCtor] = []
# make sure one process only loads logitsprocs once
logitsprocs_loaded = False


class LogitProcessorEntrypoint(TypedDict):
    package_name: str
    entrypoint_name: str


# Specify logitproc by qualname (str) or package and entrypoint name
LogitsProcessorsSpec = Union[str, LogitProcessorEntrypoint]


def load_logitsprocs_fqns(
        fqns: Optional[list[str]]) -> list[Callable[[], LogitsProcessor]]:
    if not fqns:
        return []

    logger.info("Attempting to load the following logits processors via FQNs:")

    constructors: list[Callable[[], LogitsProcessor]] = []
    for fqn in fqns:
        logger.info("Loading logits processor %s", fqn)
        try:
            module_path, qualname = fqn.split(":")
            # Load module
            module = importlib.import_module(module_path)
            # Walk down dotted name to get logitproc constructor
            obj = module
            for attr in qualname.split("."):
                obj = getattr(obj, attr)
            if not callable(obj):
                raise ValueError(f"{fqn} is not a Callable.")
            constructors.append(obj)
        except Exception:
            logger.exception("Failed to load logits processor %s", fqn)

    return constructors


def load_logitsprocs_entrypoints(
        entrypoints: Optional[list[str]]) -> list[Callable[[], Any]]:
    if not entrypoints:
        return []

    logger.info(
        "Attempting to load the following logits processors via "
        "entrypoints: %s", entrypoints)

    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    installed_logitsprocs_plugins = {
        plugin.name: plugin
        for plugin in entry_points(group=LOGITSPROCS_GROUP)
    }
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).",
                     LOGITSPROCS_GROUP)
        return []

    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.info
    log_level("Available logitsprocs plugins (group %s):", LOGITSPROCS_GROUP)
    for plugin in installed_logitsprocs_plugins.values():
        log_level("- %s -> %s", plugin.name, plugin.value)

    constructors: list[Callable[[], LogitsProcessor]] = []
    for entrypoint in entrypoints:
        if entrypoint not in installed_logitsprocs_plugins:
            raise ValueError(
                f"Invalid logit processor entrypoint string {entrypoint}.")
        log_level("Loading plugin %s", entrypoint)

        try:
            func = installed_logitsprocs_plugins[entrypoint].load()
            constructors.append(func)
        except Exception:
            logger.exception("Failed to load plugin %s", entrypoint)

    return constructors


def load_logitsprocs(
    logits_processors_fqns: Optional[list[str]],
    logits_processors_entrypoints: Optional[list[str]],
) -> None:
    """WARNING: logitsprocs can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global logitsprocs_loaded
    global logitsprocs_ctors
    if logitsprocs_loaded:
        # Idempotent after first load in a process
        return
    logitsprocs_loaded = True
    from vllm.platforms import current_platform
    if current_platform.is_tpu():
        # No logitsprocs specified by caller
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return

    logitsprocs_ctors = (
        load_logitsprocs_entrypoints(logits_processors_entrypoints) +
        load_logitsprocs_fqns(logits_processors_fqns))


def init_builtin_logitsprocs(pin_memory_available: bool, max_num_reqs: int,
                             device: torch.device) -> LogitsProcessorManager:
    """Construct 'builtin' vLLM logitsprocs which the engine
    loads by default.

    Args:
      pin_memory_available: pinned memory is available for use
                            for use by logitsproc
      max_num_reqs: ceiling on request count in persistent batch
      device: inference device

    Returns:
      Data structure encapsulating loaded logitsprocs
    """
    min_tokens_logitproc = MinTokensLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    logit_bias_logitproc = LogitBiasLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    min_p_logitproc = MinPLogitsProcessor(
        pin_memory=pin_memory_available,
        device=device,
        # +1 for temporary swap space
        max_num_reqs=max_num_reqs + 1)
    return LogitsProcessorManager(
        non_argmax_invariant=[
            min_tokens_logitproc,
            logit_bias_logitproc,
        ],
        argmax_invariant=[min_p_logitproc],
    )
