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
"""

MONO_ABI_VERSION = 1

RECORD_WORDS = 8
FAILED = 0
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of ``ctas``: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS
