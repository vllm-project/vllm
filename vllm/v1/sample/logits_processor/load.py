# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Callable, Optional, Union

import torch
from typing_extensions import TypedDict

from vllm.v1.sample.logits_processor.impls import (LogitBiasLogitsProcessor,
                                                   MinPLogitsProcessor,
                                                   MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.state import LogitsProcessorManager

logger = logging.getLogger(__name__)

LOGITSPROCS_GROUP = 'vllm.logits_processors'
LogitprocCtor = Callable[[], None]
logitsprocs_ctors: list[LogitprocCtor] = []
# make sure one process only loads logitsprocs once
logitsprocs_loaded = False


class LogitsProcessorEntrypoint(TypedDict):
    package_name: str
    entrypoint_name: str


# Specify logitproc by qualname (str) or package and entrypoint name
LogitsProcessorsSpec = Union[str, LogitsProcessorEntrypoint]


def _load_logitsprocs(
    logitsprocs: list[LogitsProcessorsSpec]
) -> dict[str, Callable[[], Any]]:
    assert logitsprocs

    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    installed_logitsprocs_plugins = entry_points(group=LOGITSPROCS_GROUP)
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).", LOGITSPROCS_GROUP)
        return {}

    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.info
    log_level("Available logitsprocs plugins (group %s):", LOGITSPROCS_GROUP)
    for plugin in installed_logitsprocs_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    plugins = dict[str, Callable[[], Any]]()
    for plugin in installed_logitsprocs_plugins:
        if logitsprocs is None or plugin.name in logitsprocs:
            if logitsprocs is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)

    return plugins


def load_logitsprocs(logitsprocs: Optional[list[LogitsProcessorsSpec]]) -> None:
    """WARNING: logitsprocs can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global logitsprocs_loaded
    if logitsprocs_loaded:
        # Idempotent after first load in a process
        return
    logitsprocs_loaded = True
    from vllm.platforms import current_platform
    if not logitsprocs or current_platform.is_tpu():
        # No logitsprocs specified by caller
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return

    plugins = _load_logitsprocs(logitsprocs)
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()


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
