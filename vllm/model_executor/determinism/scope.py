# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which modules must be built from batch-invariant kernels."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import vllm.envs as envs

_exempt: ContextVar[bool] = ContextVar("batch_invariance_exempt", default=False)


@contextmanager
def batch_invariance_exempt() -> Iterator[None]:
    """Build modules that need not be batch invariant, e.g. a speculative
    drafter: its tokens are only proposals, and the target's verification,
    which is batch invariant, decides every output token and logprob."""
    token = _exempt.set(True)
    try:
        yield
    finally:
        _exempt.reset(token)


def requires_batch_invariant_kernels() -> bool:
    """Whether kernels chosen now must be batch invariant.

    Today only the modular MoE experts selection consults this, and only the
    model runner V2 EAGLE/MTP drafter is built inside
    :func:`batch_invariance_exempt`; other batch-invariant kernel choices still
    read ``VLLM_BATCH_INVARIANT`` directly.
    """
    return envs.VLLM_BATCH_INVARIANT and not _exempt.get()
