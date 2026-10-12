# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout every MonoKernel's error words and fault hook follow.

A rank's health words: a header whose word FAILED is set once any of its
in-kernel waits gave up (sticky: a failed rank stays failed, and its later waits
exit at once), then one record a CTA of the first wait it gave up on (flag,
region id, pair index, tag, seen tag, CTA, clock, pad).

The output rank's flags: one int32 word a TP rank in the first PEER_HEAD bytes
of the output rank's peer buffer. A rank whose wait gives up writes its word
there, drains the store, then sets FAILED, so nothing it stores later can
overtake the flag. The runner reads these words after every step.

The fault hook (tests only, ``VLLM_MONO_FAULT_INJECT=rank:epoch:us``): a
build-time stall of one CTA of rank ``rank`` for ``us`` microseconds in
every step from ``epoch`` on; no code when unset.
"""

from vllm import envs

MONO_ABI_VERSION = 1

RECORD_WORDS = 8
FAILED = 0
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of ``ctas``: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS


def fault_inject() -> tuple[int, int, int] | None:
    """(rank, epoch, microseconds) of ``VLLM_MONO_FAULT_INJECT``, or None."""
    v = envs.VLLM_MONO_FAULT_INJECT
    if not v:
        return None
    rank, epoch, us = (int(x) for x in v.split(":"))
    return rank, epoch, us
