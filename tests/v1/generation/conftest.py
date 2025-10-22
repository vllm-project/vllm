# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode():
    """Automatically enable batch invariant kernel overrides for all tests."""
    old_value = os.environ.get("VLLM_BATCH_INVARIANT")
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    yield
    # restore original value after test
    if old_value is None:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = old_value
