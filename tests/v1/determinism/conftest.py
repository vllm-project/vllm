# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm.envs as envs


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


@pytest.fixture(scope="module", autouse=True)
def settle_gpu_memory_between_modules():
    """Let ROCm release VRAM before the next module allocates.

    Most of this suite builds `LLM(...)` directly rather than through the
    `vllm_runner` fixture, so it never picks up the settle that
    `tests/conftest.py` applies on that fixture's teardown. ROCm reclaims
    lazily, and the modules that start a server size themselves off *free*
    memory, so a heavy module makes the next one fail at startup rather than
    merely run slower.

    Module scope, not function scope: the settle waits for two seconds of
    stable readings, and blocks for its full timeout before failing. It
    asserts rather than warns -- the default threshold clears the allocator
    and Triton residue a kernel module leaves in-process, so tripping it means
    a whole engine is still resident and the run should say so at the module
    that caused it. No-op off ROCm.
    """
    yield

    from tests.utils import wait_for_rocm_memory_to_settle

    wait_for_rocm_memory_to_settle()
