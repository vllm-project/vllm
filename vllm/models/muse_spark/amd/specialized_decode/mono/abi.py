# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout every MonoKernel's error words follow.

A rank's health words: a header whose word FAILED is set once any of its
in-kernel waits gave up (sticky: a failed rank stays failed, and its later waits
exit at once), then one record a CTA of the first wait it gave up on (flag,
region id, pair index, tag, seen tag, CTA, clock, pad).

The output rank's flags: one int32 word a TP rank in the first PEER_HEAD bytes
of the output rank's peer buffer. A rank whose wait gives up writes its word
there, drains the store, then sets FAILED, so nothing it stores later can
overtake the flag. The runner reads these words after every step.

Muse Spark adds the header word WAIT_BUDGET: the seconds an in-kernel wait
spins before it gives up, read once a launch. The runner holds it at
STARTUP_WAIT_SECONDS from init and every KV bind until the CUDA-graph capture
ends (eager warm-ups interleave the kernels with stock kernels that may JIT for
the first time), then at WAIT_SECONDS.
"""

RECORD_WORDS = 8
FAILED = 0
WAIT_BUDGET = 1  # header word: an in-kernel wait's budget in seconds
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long while serving
STARTUP_WAIT_SECONDS = 120  # ... and until the CUDA-graph capture ends


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of ``ctas``: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS


HEALTH_WORDS = health_words(256)  # a rank's, for the K2 grid (spec.BLOCKS)

assert FAILED < WAIT_BUDGET < RECORD_WORDS
