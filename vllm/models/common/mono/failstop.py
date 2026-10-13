# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-stop and metrics.

A MonoKernel never returns tokens from a step it did not finish: the runner
copies the model's error words (``abi``) next to the sampled token ids, and
``check`` raises before that step's tokens are returned. The engine then stops
like on any other engine death, and the deployment's restart policy recovers
it.

Counters (``drain``, exported as ``vllm:monokernel_steps`` and
``vllm:monokernel_incidents``): steps on the kernels, steps on vLLM's path by
reason, and incidents by failed rank.
"""

from collections import Counter

import torch

STEPS: Counter[tuple[str, str]] = Counter()  # (spec, "mono" or the no-go reason)
INCIDENTS: Counter[str] = Counter()  # by failed TP rank


def count_step(spec: str, why: str | None) -> None:
    STEPS[spec, why or "mono"] += 1


def drain() -> dict[str, int] | None:
    """The counts since the last drain: ``"steps:<spec>:<path>"`` (path "mono"
    or the reason the step ran vLLM's layers) and ``"incidents:<rank>"``."""
    if not STEPS and not INCIDENTS:
        return None
    out = {f"steps:{spec}:{path}": n for (spec, path), n in STEPS.items()}
    out.update({f"incidents:{rank}": n for rank, n in INCIDENTS.items()})
    STEPS.clear()
    INCIDENTS.clear()
    return out


def check(words: torch.Tensor) -> None:
    """``words``: the output rank's error words for one step, on the host (a
    word a TP rank, nonzero once one of its in-kernel waits gave up)."""
    if not words.any():
        return
    ranks = [r for r, w in enumerate(words.tolist()) if w]
    for r in ranks:
        INCIDENTS[str(r)] += 1
    msg = (
        f"MonoKernel step failed: TP ranks {ranks} gave up an in-kernel wait; "
        "the step's tokens are discarded"
    )
    raise RuntimeError(msg)
