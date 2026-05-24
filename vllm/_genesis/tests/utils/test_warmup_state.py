# SPDX-License-Identifier: Apache-2.0
"""TDD for warmup_state context manager."""
from __future__ import annotations

import asyncio

import pytest


def test_default_inactive():
    from vllm._genesis.utils.warmup_state import is_warmup_active
    # Outside any context — must be False
    assert is_warmup_active() is False


def test_context_activates_and_resets():
    from vllm._genesis.utils.warmup_state import (
        is_warmup_active, warmup_active,
    )
    assert is_warmup_active() is False
    with warmup_active():
        assert is_warmup_active() is True
    # After context — must be False again
    assert is_warmup_active() is False


def test_nested_contexts():
    from vllm._genesis.utils.warmup_state import (
        is_warmup_active, warmup_active,
    )
    with warmup_active():
        assert is_warmup_active() is True
        with warmup_active():
            assert is_warmup_active() is True
        # Inner exit restores token, outer still active
        assert is_warmup_active() is True
    assert is_warmup_active() is False


def test_exception_resets():
    from vllm._genesis.utils.warmup_state import (
        is_warmup_active, warmup_active,
    )
    with pytest.raises(RuntimeError):
        with warmup_active():
            assert is_warmup_active() is True
            raise RuntimeError("simulated kernel crash")
    # After exception — must be reset to False
    assert is_warmup_active() is False


def test_async_context_safe():
    """Verify contextvars work correctly across async tasks."""
    from vllm._genesis.utils.warmup_state import (
        is_warmup_active, warmup_active,
    )

    async def child_task():
        # Child task inherits parent's contextvar
        return is_warmup_active()

    async def runner():
        with warmup_active():
            r = await asyncio.create_task(child_task())
        return r

    result = asyncio.run(runner())
    # Inside warmup_active, child task should see True
    assert result is True
