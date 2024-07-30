import pytest

from .test_stop_strings import _test_stopping

MODEL = "meta-llama/llama-2-7b-hf"
MAX_TOKENS = 200

IS_ASYNC = True


@pytest.fixture(scope="module")
def vllm_model_async(vllm_runner):
    with vllm_runner(MODEL,
                     disable_async_output_proc=False) as vllm_model_async:
        yield vllm_model_async


@pytest.mark.skip_global_cleanup
def test_stop_basic(vllm_model_async):
    _test_stopping(vllm_model_async.model.llm_engine,
                   stop=["."],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=".",
                   use_async_output_proc=IS_ASYNC)

    _test_stopping(vllm_model_async.model.llm_engine,
                   stop=["."],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization.",
                   expected_reason=".",
                   use_async_output_proc=IS_ASYNC)


@pytest.mark.skip_global_cleanup
def test_stop_multi_tokens(vllm_model_async):
    _test_stopping(
        vllm_model_async.model.llm_engine,
        stop=["group of peo", "short"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a ",
        expected_reason="group of peo",
        use_async_output_proc=IS_ASYNC)

    _test_stopping(
        vllm_model_async.model.llm_engine,
        stop=["group of peo", "short"],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of peo",
        expected_reason="group of peo",
        use_async_output_proc=IS_ASYNC)


@pytest.mark.skip_global_cleanup
def test_stop_partial_token(vllm_model_async):
    _test_stopping(vllm_model_async.model.llm_engine,
                   stop=["gani"],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer or",
                   expected_reason="gani",
                   use_async_output_proc=IS_ASYNC)

    _test_stopping(vllm_model_async.model.llm_engine,
                   stop=["gani"],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organi",
                   expected_reason="gani",
                   use_async_output_proc=IS_ASYNC)


@pytest.mark.skip_global_cleanup
def test_stop_token_id(vllm_model_async):
    # token id 13013 => " organization"

    _test_stopping(vllm_model_async.model.llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer",
                   expected_reason=13013,
                   use_async_output_proc=IS_ASYNC)

    _test_stopping(vllm_model_async.model.llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=13013,
                   use_async_output_proc=IS_ASYNC)
