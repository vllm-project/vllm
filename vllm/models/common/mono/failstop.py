# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-stop and metrics.

A MonoKernel never returns tokens from a step it did not finish: the runner
copies the model's error words (``abi``) next to the sampled token ids, and
``check`` raises before that step's tokens are returned. The engine then stops
like on any other engine death, and the deployment's restart policy recovers
it. ``VLLM_MONO_UNSAFE_WARN=1`` (tests and benchmarks only, no CLI flag) logs
the incident and keeps serving instead.

Counters: steps on the kernels, steps on vLLM's path by reason, incidents.
"""

from collections import Counter

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

STEPS: Counter[tuple[str, str]] = Counter()  # (spec, "mono" or the no-go reason)
INCIDENTS: Counter[str] = Counter()  # failed ranks, "rank r"


def count_step(spec: str, why: str | None) -> None:
    STEPS[spec, why or "mono"] += 1


def check(words: torch.Tensor) -> None:
    """``words``: the output rank's error words for one step, on the host (a
    word a TP rank, nonzero once one of its in-kernel waits gave up)."""
    if not words.any():
        return
    ranks = [r for r, w in enumerate(words.tolist()) if w]
    for r in ranks:
        INCIDENTS[f"rank {r}"] += 1
    msg = (
        f"MonoKernel step failed: TP ranks {ranks} gave up an in-kernel wait; "
        "the step's tokens are discarded"
    )
    if envs.VLLM_MONO_UNSAFE_WARN:
        logger.error("%s (VLLM_MONO_UNSAFE_WARN=1: serving on)", msg)
        return
    raise RuntimeError(msg)
