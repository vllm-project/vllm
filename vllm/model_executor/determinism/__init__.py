# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import vllm.envs as envs

_building_drafter: ContextVar[bool] = ContextVar("building_drafter", default=False)


@contextmanager
def building_drafter() -> Iterator[None]:
    token = _building_drafter.set(True)
    try:
        yield
    finally:
        _building_drafter.reset(token)


def requires_batch_invariant_kernels() -> bool:
    """With VLLM_BI_DRAFTER=0, the drafter may use non-batch-invariant kernels."""
    return envs.VLLM_BATCH_INVARIANT and (
        envs.VLLM_BI_DRAFTER or not _building_drafter.get()
    )
