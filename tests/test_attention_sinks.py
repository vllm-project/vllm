"""Test attention sinks correctness for large models (7B).

Run `pytest tests/test_attention_sinks.py`.
"""
from functools import lru_cache
import os
import pytest
from typing import List, Tuple

from vllm import LLM, SamplingParams, RequestOutput
from tests.conftest import cleanup


_ATTN_SINKS_PROMPTS_FILEPATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "attn-sinks-prompts.json"
)


@pytest.mark.parametrize(
    "model, max_model_len, test_retrieval, min_tokens, max_tokens",
    [
        # rope models
        ("meta-llama/Meta-Llama-3-8B-Instruct", 8192, True, 100, 500),
        ("mistralai/Mistral-7B-Instruct-v0.2", 32768, True, 100, 500),
        # alibi models
        ("mosaicml/mpt-7b-chat", 2048, False, 500, 1000),
        ("bigscience/bloom-7b1", 2048, False, 500, 1000)
    ]
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [5])
def test_attention_sinks(
    vllm_runner,
    model: str,
    max_model_len: int,
    test_retrieval: bool,
    min_tokens: int,
    max_tokens: int,
    dtype: str,
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch
):
    prompt = _get_prompt(model, test_retrieval=test_retrieval)
    prompts = [prompt for _ in range(batch_size)]
    params = SamplingParams(min_tokens=min_tokens, max_tokens=max_tokens)

    normal_model = vllm_runner(
        model,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True
    )
    
    # bypass context length cap for normal generation
    # to compare w/ attention sinks, which generates past context length
    monkeypatch.setattr(
        normal_model.model.llm_engine.output_processor.stop_checker,
        "use_attention_sinks",
        True
    )
    
    normal_outputs = normal_model.generate_w_cum_logprobs(prompts, params)
    del normal_model

    sink_model = vllm_runner(
        model,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        use_attention_sinks=True
    )

    sink_outputs = sink_model.generate_w_cum_logprobs(prompts, params)
    del sink_model

    if test_retrieval:
        for output_str, _ in sink_outputs:
            assert "mint green" in output_str.lower()

    max_normal_logprob = max(logprobs for _, logprobs in normal_outputs)
    min_sink_logprob = min(logprobs for _, logprobs in sink_outputs)
    
    # attn sinks should be lower perplexity (less negative cumulative logprobs)
    assert max_normal_logprob < min_sink_logprob, (
        f"max normal logprob: {max_normal_logprob}, "
        f"min sink logprob: {min_sink_logprob}"
    )


def _get_prompt(model_name: str, test_retrieval: bool) -> str:
    prompts = _get_prompts_json()
    prompt = prompts[model_name]
    # prompt is (model's context length - 100) tokens long
    
    if test_retrieval:
        return (
            "Remember: my favorite color is mint green. "
            f"Here is a Harry Potter excerpt: {prompt} "
            "First, summarize this excerpt. "
            "Then, print my favorite color AFTER the summary."
        )
    else:
        return prompt


@lru_cache
def _get_prompts_json():
    import json
    with open(_ATTN_SINKS_PROMPTS_FILEPATH, "r") as f:
        return json.load(f)
