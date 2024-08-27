from ..utils import compare_two_settings


def test_custom_dispatcher():
    compare_two_settings("google/gemma-2b", [], [],
                         {"VLLM_DYNAMO_USE_CUSTOM_DISPATCHER": "0"})
