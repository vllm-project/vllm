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
    """Let ROCm release VRAM before the next module allocates. No-op off ROCm."""
    yield

    from tests.utils import wait_for_rocm_memory_to_settle

    wait_for_rocm_memory_to_settle()
