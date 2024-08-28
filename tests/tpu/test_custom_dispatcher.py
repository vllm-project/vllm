from ..utils import compare_two_settings


def test_custom_dispatcher():
    compare_two_settings("google/gemma-2b",
                         arg1=["--enforce-eager"],
                         arg2=["--enforce-eager"],
                         env1={"VLLM_DYNAMO_USE_CUSTOM_DISPATCHER": "0"},
                         env2={})
