# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmups owned by model packages and run by `kernel_warmup()`.

A model package registers its warmups when it is imported, so the engine does
not import concrete models. Each warmup must return early when the loaded model
has none of its layers.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

ModelWarmup: TypeAlias = Callable[["Worker"], None]

# "jit": with JIT warmup enabled, before other warmups touch the runner state.
# "full": skipped by process-local warmups, after the shared kernel warmups.
ModelWarmupPhase: TypeAlias = Literal["jit", "full"]

_MODEL_WARMUPS: dict[ModelWarmupPhase, list[ModelWarmup]] = {"jit": [], "full": []}


def register_model_warmup(
    phase: ModelWarmupPhase,
) -> Callable[[ModelWarmup], ModelWarmup]:
    def wrapper(fn: ModelWarmup) -> ModelWarmup:
        if fn not in _MODEL_WARMUPS[phase]:
            _MODEL_WARMUPS[phase].append(fn)
        return fn

    return wrapper


def run_model_warmups(worker: "Worker", phase: ModelWarmupPhase) -> None:
    for fn in _MODEL_WARMUPS[phase]:
        fn(worker)
