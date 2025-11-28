# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import json

import torch

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
        _dump_engine_exception(config, scheduler_output, scheduler_stats)


def _dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
):
    logger.error(
        "Dumping input data for V1 LLM engine (v%s) with config: %s, ",
        VLLM_VERSION,
        config,
    )
    try:
        dump_obj = prepare_object_to_dump(scheduler_output)
        logger.error("Dumping scheduler output for model execution: %s", dump_obj)
        if scheduler_stats:
            logger.error("Dumping scheduler stats: %s", scheduler_stats)
    except Exception:
        logger.exception("Error preparing object to dump")
