# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import inspect
import itertools
from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import (LogitBiasLogitsProcessor,
                                                     MinPLogitsProcessor,
                                                     MinTokensLogitsProcessor,
                                                     process_dict_updates)
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


class AdapterLogitsProcessor(LogitsProcessor):
    """Wrapper for per-request logits processors
    
    To wrap a specific per-request logits processor,
    * Subclass `AdapterLogitsProcessor`
    * Implement `self.is_argmax_invariant()` base-class method
    * Implement `self.new_req_logits_processor(params)`
    
    `self.__init__(vllm_config, device, is_pin_memory)` does not need to be
    overridden in general. However, to implement custom constructor behavior -
    especially any logic which operates on or stores `vllm_config`, `device`,
    or `is_pin_memory` - `self.__init__(vllm_config, device, is_pin_memory)`
    must be overridden and the override must call
    `super().__init__(vllm_config, device, is_pin_memory)`
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        """Subclass must invoke
        `super().__init__(vllm_config, device, is_pin_memory)`.

        Subclass constructor may find it useful to utilize the `vllm_config`,
        `device` and `is_pin_memory` argument. However regardless of whether
        these arguments are used, the vLLM logits processor interface requires
        all three arguments to be present.
        """

        # Map req index -> logits processor state
        #
        # State representation is a partial[Tensor] comprising a request-level
        # logits processor with the output token ids argument and (if required)
        # the prompt token ids argument pre-populated
        #
        # Note that the partial carries a *reference* to output token ids, and
        # will thus always operate on the list as it is currently, not as it
        # was when the partial was created.
        self.req_info: dict[int, partial[torch.Tensor]] = {}

    @abstractmethod
    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """Consume request info; return a per-request logits processor.

        Return None if logits processor does not need to be applied to request

        Args:
          params: request sampling params

        Returns:
          None if logits processor should not be applied to request; otherwise
          returns a `RequestLogitsProcessor` instance
        
        """
        raise NotImplementedError

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: Optional[list[int]],
        output_ids: list[int],
    ) -> Optional[partial[torch.Tensor]]:
        """Return state representation for new request

        Returns None if logits processor is not applicable to request

        Args:
          params: request sampling params
          prompt_ids: request prompt token ids
          output_ids: decoded tokens so far for this request

        Returns:
          logits processor partial[Tensor] or None
        
        """
        if req_lp := self.new_req_logits_processor(params):
            args = [prompt_ids, output_ids] if (len(
                inspect.signature(req_lp).parameters) == 3) else [output_ids]
            return partial(req_lp, *args)
        return None

    def update_state(self, batch_update: Optional[BatchUpdate]):
        process_dict_updates(
            self.req_info,
            batch_update,
            self._new_state,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.req_info:
            # Apply per-request logits processors to corresponding rows of
            # logits tensor
            for req_idx, req_lp in self.req_info.items():
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    # Modify logits tensor row in-place if necessary
                    logits[req_idx] = new_logits
        return logits


__all__ = [
    "LogitsProcessor", "LogitBiasLogitsProcessor", "MinPLogitsProcessor",
    "MinTokensLogitsProcessor", "BatchUpdate", "BatchUpdateBuilder",
    "MoveDirectionality", "LogitsProcessors", "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS", "LOGITSPROCS_GROUP",
    "AdapterLogitsProcessor"
]
