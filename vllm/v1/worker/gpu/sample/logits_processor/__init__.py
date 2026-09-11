# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import guard_cuda_initialization
from vllm.v1.sample.logits_processor import STR_POOLING_REJECTS_LOGITSPROCS
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.worker.gpu.sample.logits_processor.state import LogitsProcessors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

__all__ = ["LogitsProcessor", "LogitsProcessors", "build_logitsprocs"]


def _load_v2_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    """Load installed logits processor plugins.

    Plugins registered under ``LOGITSPROCS_GROUP`` ship V1-interface
    processors today; those crash under the V2 sampler, so reject them at
    load time instead. V2-compatible plugin packages register subclasses of
    this package's ``LogitsProcessor`` and load normally.
    """
    from vllm.v1.sample.logits_processor import LOGITSPROCS_GROUP

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
    logits_processors: Sequence[str | type[LogitsProcessor]],
) -> list[type[LogitsProcessor]]:
    """Resolve a mixed list of processor types and FQCN strings into types.

    FQCN syntax is <module>:<type> i.e. x.y.z:CustomLogitProc. Loaded
    classes must subclass this package's ``LogitsProcessor``; V1-only
    processors are rejected here rather than at sampling time.
    """
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
        module_path, qualname = logitproc.split(":")

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
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """Load all custom logits processors for the V2 model runner.

    Combines installed plugins (validated as V2 processors) with the
    user-specified list of processor types and FQCN strings.
    """
    return _load_v2_logitsprocs_plugins() + _load_v2_logitsprocs_by_fqcns(
        logits_processors or ()
    )


def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return LogitsProcessors()

    # Unlike V1, custom logits processors stay active under speculative
    # decoding: the V2 rejection sampler reuses the main sampler's
    # apply_sampling_params path, so processors see the expanded draft rows.
    # The loader above validates against the V2 interface, so V1-only
    # processors are rejected at load time.
    custom_logitsprocs_classes = _load_v2_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in custom_logitsprocs_classes
    )
