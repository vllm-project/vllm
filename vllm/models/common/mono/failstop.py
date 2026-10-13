# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-stop and metrics.

A MonoKernel never returns tokens from a step it did not finish: the runner
copies the model's error words (``abi``) next to the sampled token ids, and
``check`` raises before that step's tokens are returned. The engine then stops
like on any other engine death, and the deployment's restart policy recovers
it.

Counters (``drain``, exported as ``vllm:monokernel_steps``): steps on the
kernels, and steps on vLLM's path by reason.
"""

from collections import Counter

import torch

STEPS: Counter[tuple[str, str]] = Counter()


def count_step(spec: str, why: str | None) -> None:
    """Counts one served step of ``spec``: on the kernels when ``why`` is None,
    else on vLLM's layers for that reason."""
    STEPS[spec, why or "mono"] += 1


def drain() -> dict[str, int] | None:
    """The counts since the last drain: ``"steps:<spec>:<path>"``, where path
    is "mono" or the reason the step ran vLLM's layers."""
    if not STEPS:
        return None
    out = {f"steps:{spec}:{path}": n for (spec, path), n in STEPS.items()}
    STEPS.clear()
    return out


def check(words: torch.Tensor) -> None:
    """Fails the step whose error words these are, before its tokens return.

    Args:
        words: The output rank's error words for one step, on the host (a word
            a TP rank, nonzero once one of its in-kernel waits gave up).

    Raises:
        RuntimeError: A rank gave up a wait.

    """
    if not words.any():
        return
    ranks = [r for r, w in enumerate(words.tolist()) if w]
    raise RuntimeError(
        f"MonoKernel step failed: TP ranks {ranks} gave up an in-kernel wait; "
        "the step's tokens are discarded"
    )
