# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.config import CompilationLevel

from ..utils import compare_two_settings

# --enforce-eager on TPU causes graph compilation
# this times out default Health Check in the MQLLMEngine,
# so we set the timeout here to 30s


def test_custom_dispatcher(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_RPC_TIMEOUT", "30000")
        compare_two_settings("Qwen/Qwen2.5-1.5B-Instruct",
                             arg1=[
                                 "--max-model-len=256",
                                 "--max-num-seqs=32",
                                 "--enforce-eager",
                                 f"-O{CompilationLevel.DYNAMO_ONCE}",
                             ],
                             arg2=[
                                 "--max-model-len=256", "--max-num-seqs=32",
                                 "--enforce-eager",
                                 f"-O{CompilationLevel.DYNAMO_AS_IS}"
                             ],
                             env1={},
                             env2={})
