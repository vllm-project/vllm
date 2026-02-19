# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    yield
