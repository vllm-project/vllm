# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading of custom logits processor classes for the V2 model runner.

This module is deliberately import-light: the frontend process uses it to
validate per-request params, so importing it must not pull in model-runner
side modules (torch, triton, worker state). Imports needed to instantiate
processors are deferred to ``build_custom_logits_processors``.
"""

import importlib
from collections.abc import Callable, Sequence
from functools import lru_cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

# Same entry-points group as
# vllm.v1.sample.logits_processor.LOGITSPROCS_GROUP; duplicated here so this
# module stays importable without the V1 sampler machinery.
LOGITSPROCS_GROUP = "vllm.logits_processors"


def _load_v2_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    """Load installed logits processor plugins.

    Plugins registered under ``LOGITSPROCS_GROUP`` ship V1-interface
    processors today; those crash under the V2 sampler, so reject them at
    load time.
    """
    from vllm.utils.torch_utils import guard_cuda_initialization

    installed_logitsprocs_plugins = entry_points(group=LOGITSPROCS_GROUP)
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).", LOGITSPROCS_GROUP)
        return []

    logger.debug("Loading installed logitsprocs plugins (group %s):", LOGITSPROCS_GROUP)
    classes: list[type[LogitsProcessor]] = []
    for entrypoint in installed_logitsprocs_plugins:
        try:
            logger.debug(
                "- Loading logitproc plugin entrypoint=%s target=%s",
                entrypoint.name,
                entrypoint.value,
            )
            with guard_cuda_initialization():
                cls = entrypoint.load()
        except Exception as e:
            logger.error("Failed to load LogitsProcessor plugin %s: %s", entrypoint, e)
            raise RuntimeError(
                f"Failed to load LogitsProcessor plugin {entrypoint}"
            ) from e
        if not (isinstance(cls, type) and issubclass(cls, LogitsProcessor)):
            raise ValueError(
                f"LogitsProcessor plugin {entrypoint.name} is not a subclass of "
                "vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor; "
                "V1-interface plugins are not supported by the V2 model runner."
            )
        classes.append(cls)
    return classes


def _load_v2_logitsprocs_by_fqcns(
    logits_processors: Sequence[str | type],
) -> list[type[LogitsProcessor]]:
    """Resolve a mixed list of processor types and FQCN strings into types.

    FQCN syntax is <module>:<type> i.e. x.y.z:CustomLogitProc.
    """
    from vllm.utils.torch_utils import guard_cuda_initialization

    classes: list[type[LogitsProcessor]] = []
    for ldx, logitproc in enumerate(logits_processors):
        if isinstance(logitproc, type):
            logger.debug(" - Already-loaded logit processor: %s", logitproc.__name__)
            if not issubclass(logitproc, LogitsProcessor):
                raise ValueError(
                    f"{logitproc.__name__} is not a subclass of "
                    "vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor"
                )
            classes.append(logitproc)
            continue

        logger.debug("- Loading logits processor %s", logitproc)
        parts = logitproc.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid logits processor FQCN {logitproc!r}. "
                "Expected format: '<module>:<type>'"
            )
        module_path, qualname = parts

        try:
            with guard_cuda_initialization():
                module = importlib.import_module(module_path)
        except Exception as e:
            logger.error(
                "Failed to load %sth LogitsProcessor plugin %s: %s",
                ldx,
                logitproc,
                e,
            )
            raise RuntimeError(
                f"Failed to load {ldx}th LogitsProcessor plugin {logitproc}"
            ) from e

        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if not isinstance(obj, type):
            raise ValueError("Loaded logit processor must be a type.")
        if not issubclass(obj, LogitsProcessor):
            raise ValueError(
                f"{obj.__name__} is not a subclass of "
                "vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor"
            )
        classes.append(obj)

    return classes


def _load_v2_logitsprocs(
    logits_processors: Sequence[str | type] | None,
) -> list[type[LogitsProcessor]]:
    """Load all custom logits processors for the V2 model runner.

    Combines installed plugins (validated as V2 processors) with the
    user-specified list of processor types and FQCN strings.
    """
    return _load_v2_logitsprocs_plugins() + _load_v2_logitsprocs_by_fqcns(
        logits_processors or ()
    )


@lru_cache
def _cached_load_v2_logitsprocs(
    custom_logitsprocs: tuple[str | type, ...],
) -> list[type[LogitsProcessor]]:
    return _load_v2_logitsprocs(list(custom_logitsprocs))


def build_custom_logits_processors(
    vllm_config: "VllmConfig",
    req_states: "RequestState",
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type] = (),
) -> list[LogitsProcessor]:
    """Load and instantiate custom logits processors, entrypoint plugins first.

    Raises:
        ValueError: if a pooling model specifies custom processors, or a
            loaded class does not implement the V2 interface.
        RuntimeError: if an FQCN fails to import.

    """
    from vllm.v1.sample.logits_processor import STR_POOLING_REJECTS_LOGITSPROCS

    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return []

    custom_logitsprocs_classes = _load_v2_logitsprocs(custom_logitsprocs)
    lp_req_state = LogitsProcRequestState.from_request_state(req_states)
    return [ctor(vllm_config, lp_req_state) for ctor in custom_logitsprocs_classes]


def build_custom_logits_processors_params_validator(
    custom_logitsprocs: Sequence[str | type] | None,
) -> "Callable[[SamplingParams], None]":
    """Load custom processor classes once and return a params validator.

    Called from the frontend at startup. The returned callable runs each
    processor's ``validate_params`` at request admission.

    Raises:
        ValueError: if a loaded class does not implement the V2 interface.
        RuntimeError: if an FQCN fails to import.

    """
    if not custom_logitsprocs:
        return lambda _: None

    classes = _cached_load_v2_logitsprocs(tuple(custom_logitsprocs))

    def validate_params(sampling_params: "SamplingParams") -> None:
        for cls in classes:
            try:
                cls.validate_params(sampling_params)
            except ValueError as e:
                raise VLLMValidationError(str(e)) from e

    return validate_params
