# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import faulthandler
import json
import os
import sys
import threading
from types import TracebackType

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return f"'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ", ".join(
            {f"{str(k)}: {prepare_object_to_dump(v)}" for k, v in obj.items()}
        )
        return f"{{{dict_str}}}"
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return f"Tensor(shape={obj.shape}, device={obj.device},dtype={obj.dtype})"
    elif hasattr(obj, "anon_repr"):
        return obj.anon_repr()
    elif hasattr(obj, "__dict__"):
        items = obj.__dict__.items()
        dict_str = ", ".join(
            [f"{str(k)}={prepare_object_to_dump(v)}" for k, v in items]
        )
        return f"{type(obj).__name__}({dict_str})"
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    # NOTE: ensure we can log extra info without risking raises
    # unexpected errors during logging
    with contextlib.suppress(Exception):
        _dump_engine_execution_context(
            "exception", config, scheduler_output, scheduler_stats
        )


def dump_engine_execution_timeout(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    timeout_s: float,
    stage: str,
):
    with contextlib.suppress(Exception):
        logger.error(
            "V1 LLM engine stage '%s' has not completed after %.2f seconds "
            "(pid=%d). Dumping scheduler state and Python stack traces. "
            "Set VLLM_ENGINE_ITERATION_TIMEOUT_S=0 to disable this diagnostic.",
            stage,
            timeout_s,
            os.getpid(),
        )
        _dump_engine_execution_context(
            "timeout", config, scheduler_output, scheduler_stats
        )

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def _dump_engine_execution_context(
    reason: str,
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    logger.error(
        "Dumping input data for V1 LLM engine (v%s, reason=%s) with config: %s, ",
        VLLM_VERSION,
        reason,
        config,
    )
    try:
        dump_obj = prepare_object_to_dump(scheduler_output)
        logger.error("Dumping scheduler output for model execution: %s", dump_obj)
        if scheduler_stats:
            logger.error("Dumping scheduler stats: %s", scheduler_stats)
    except Exception:
        logger.exception("Error preparing object to dump")


class EngineExecutionTimeoutDumper:
    """Dumps engine state if a model execution stage exceeds a timeout."""

    def __init__(
        self,
        config: VllmConfig,
        scheduler_output: SchedulerOutput,
        scheduler_stats: SchedulerStats | None,
        timeout_s: float | None,
        stage: str,
    ) -> None:
        self.config = config
        self.scheduler_output = scheduler_output
        self.scheduler_stats = scheduler_stats
        self.timeout_s = timeout_s
        self.stage = stage
        self._timer: threading.Timer | None = None

    def __enter__(self) -> Self:
        if self.timeout_s is None or self.timeout_s <= 0:
            return self

        self._timer = threading.Timer(self.timeout_s, self._dump_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _dump_timeout(self) -> None:
        dump_engine_execution_timeout(
            self.config,
            self.scheduler_output,
            self.scheduler_stats,
            self.timeout_s or 0,
            self.stage,
        )
