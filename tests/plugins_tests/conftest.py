# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    Since this module is V0 only, set VLLM_USE_V1=0 for
    all tests in the module.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')