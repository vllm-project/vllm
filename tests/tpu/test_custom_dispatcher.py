# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from vllm.config import CompilationLevel

from ..utils import compare_two_settings

# --enforce-eager on TPU causes graph compilation
# this times out default Health Check in the MQLLMEngine,
# so we set the timeout here to 30s
@pytest.fixture(scope="module", autouse=True)
#All tests in this file to use VLLM_RPC_TIMEOUT = 30000
def set_vllm_rpc_timeout(monkeypatch):
    monkeypatch.setenv("VLLM_RPC_TIMEOUT", "30000")
    yield
    monkeypatch.delenv("VLLM_RPC_TIMEOUT", raising=False)


def test_custom_dispatcher():
    compare_two_settings(
        "google/gemma-2b",
        arg1=[
            "--enforce-eager",
            f"-O{CompilationLevel.DYNAMO_ONCE}",
        ],
        arg2=["--enforce-eager", f"-O{CompilationLevel.DYNAMO_AS_IS}"],
        env1={},
        env2={})
