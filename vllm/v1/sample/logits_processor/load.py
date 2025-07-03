# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, TypedDict, Union

import torch

from vllm.plugins import load_plugins_by_group
from vllm.v1.sample.logits_processor.impls import (LogitBiasLogitsProcessor,
                                                   MinPLogitsProcessor,
                                                   MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.state import LogitsProcessorManager

LOGITSPROCS_GROUP = 'vllm.logits_processors'
LogitprocCtor = Callable[[], None]
logitsprocs_ctors: list[LogitprocCtor] = []
# make sure one process only loads logitsprocs once
logitsprocs_loaded = False


class LogitsProcessorEntrypoint(TypedDict):
    package_name: str
    entrypoint_name: str


# Specify logitproc by qualname (str) or package and entrypoint name
LogitsProcessorSpec = Union[str, LogitsProcessorEntrypoint]


def load_logitsprocs(allowed_logitsprocs: list[str]) -> None:
    """WARNING: logitsprocs can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global logitsprocs_loaded
    if logitsprocs_loaded:
        return
    logitsprocs_loaded = True

    # some platform-specific configurations
    from vllm.platforms import current_platform

    if current_platform.is_tpu():
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return
    plugins = load_plugins_by_group(group=LOGITSPROCS_GROUP,
                                    allowed_plugins=allowed_logitsprocs)
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
