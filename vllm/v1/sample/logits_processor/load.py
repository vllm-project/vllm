# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import itertools
import logging
from typing import Callable, Optional

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.impls import (LogitBiasLogitsProcessor,
                                                   MinPLogitsProcessor,
                                                   MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.logits_processor.utils import LogitProcessorCtorArgs

logger = logging.getLogger(__name__)

LOGITSPROCS_GROUP = 'vllm.logits_processors'
LogitprocCtor = Callable[[LogitProcessorCtorArgs], LogitsProcessor]

_builtin_logitsprocs_ctors: list[LogitprocCtor] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


def _load_logitsprocs_ctors_by_fqns(
        fqns: Optional[list[str]]) -> list[LogitprocCtor]:
    if not fqns:
        return []

    logger.info(
        "Attempting to load the following logits processors via FQNs: %s",
        fqns)

    constructors: list[LogitprocCtor] = []
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
        except Exception as e:
            logger.exception("Failed to load logits processor %s", fqn)
            raise e

    return constructors


def _load_logitsprocs_ctors_by_entrypoints(
        entrypoints: Optional[list[str]]) -> list[LogitprocCtor]:
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

    constructors: list[LogitprocCtor] = []
    for entrypoint in entrypoints:
        if entrypoint not in installed_logitsprocs_plugins:
            raise ValueError(
                f"Invalid logit processor entrypoint string {entrypoint}.")
        log_level("Loading plugin %s", entrypoint)

        try:
            func = installed_logitsprocs_plugins[entrypoint].load()
            constructors.append(func)
        except Exception as e:
            logger.exception("Failed to load plugin %s", entrypoint)
            raise e

    return constructors


def _load_custom_logitsprocs_ctors(
    logits_processors_fqns: Optional[list[str]],
    logits_processors_entrypoints: Optional[list[str]],
) -> list[LogitprocCtor]:
    """WARNING: logitsprocs can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    from vllm.platforms import current_platform
    if current_platform.is_tpu():
        # No logitsprocs specified by caller
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return []

    return (
        _load_logitsprocs_ctors_by_entrypoints(logits_processors_entrypoints) +
        _load_logitsprocs_ctors_by_fqns(logits_processors_fqns))


def build_logitsprocs(args: LogitProcessorCtorArgs) -> LogitsProcessors:
    _custom_logitsprocs_ctors = _load_custom_logitsprocs_ctors(
        args.vllm_config.logits_processors_fqns,
        args.vllm_config.logits_processors_entrypoints,
    )
    return LogitsProcessors(
        ctor(args) for ctor in itertools.chain(_builtin_logitsprocs_ctors,
                                               _custom_logitsprocs_ctors))
