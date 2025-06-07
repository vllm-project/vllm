# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


# TEST V1: this should be removed. Right now V1 overrides
# all the torch compile logic. We should re-enable this
# as we add torch compile support back to V1.
@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    Since this module is V0 only, set VLLM_USE_V1=0 for
    all tests in the module.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')
