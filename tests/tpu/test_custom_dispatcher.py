import os

from vllm.config import CompilationLevel

from ..utils import compare_two_settings

# --enforce-eager on TPU causes graph compilation
# this times out default Health Check in the MQLLMEngine,
# so we set the timeout here to 30s
os.environ["VLLM_RPC_TIMEOUT"] = "30000"


def test_custom_dispatcher():
    compare_two_settings(
        "google/gemma-2b",
        arg1=["--enforce-eager", "-O",
              str(CompilationLevel.DYNAMO_ONCE)],
        arg2=["--enforce-eager", "-O",
              str(CompilationLevel.DYNAMO_AS_IS)],
        env1={},
        env2={})
