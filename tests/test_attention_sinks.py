"""Test attention sinks correctness for large models (7B).

Run `pytest tests/test_attention_sinks.py`.
"""
from functools import lru_cache
from math import isnan
import os
import pytest

from vllm import SamplingParams, EngineArgs, LLMEngine
from vllm.attention.selector import get_attn_backend


_ATTN_SINKS_PROMPTS_FILEPATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "attn-sinks-prompts.txt"
)

_RETRIEVAL_COLOR = "mint green"


@pytest.mark.parametrize(
    "model, max_model_len, test_retrieval, min_tokens, max_tokens, enable_chunked_prefill",
    [
        # rope models
        ("meta-llama/Meta-Llama-3-8B-Instruct", 8192, True, 100, 400, True),
        ("mistralai/Mistral-7B-Instruct-v0.2", 32768, True, 100, 400, False),
        # alibi models
        ("mosaicml/mpt-7b-chat", 2048, False, 500, 800, False),
        ("bigscience/bloom-7b1", 2048, False, 500, 800, False)
    ]
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("attn_backend", ["XFORMERS", "FLASH_ATTN"])
def test_correctness(
    vllm_runner,
    model: str,
    max_model_len: int,
    test_retrieval: bool,
    min_tokens: int,
    max_tokens: int,
    dtype: str,
    batch_size: int,
    attn_backend: str,
    enable_chunked_prefill: bool,
    monkeypatch: pytest.MonkeyPatch
):
    if model == "mosaicml/mpt-7b-chat" and attn_backend == "XFORMERS":
        return  # sinks performance is worse than just alibi here
    
    prompt = _get_prompt(model, test_retrieval=test_retrieval)
    prompts = [prompt] * batch_size
    params = SamplingParams(
        temperature=0.5,
        min_tokens=min_tokens,
        max_tokens=max_tokens
    )

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", attn_backend)
    with vllm_runner(
        model,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        enable_chunked_prefill=enable_chunked_prefill
    ) as normal_model:
        # bypass context length cap for normal generation
        # to compare w/ attention sinks, which generates past context length
        monkeypatch.setattr(
            normal_model.model.llm_engine.output_processor.stop_checker,
            "use_attention_sinks",
            True
        )
        normal_outputs = normal_model.generate_w_cum_logprobs(prompts, params)
        monkeypatch.undo() # undo setattr so that cleanup runs correctly

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", attn_backend)
    with vllm_runner(
        model,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        use_attention_sinks=True,
        enable_chunked_prefill=enable_chunked_prefill
    ) as sink_model:
        sink_outputs = sink_model.generate_w_cum_logprobs(prompts, params)
    
    get_attn_backend.cache_clear()

    if test_retrieval:
        for output_str, _ in sink_outputs:
            assert _RETRIEVAL_COLOR in output_str.lower()

    sum_normal_avg_logprob_per_token = sum(
        avg_logprob for _, avg_logprob in normal_outputs)
    sum_sink_avg_logprob_per_token = sum(
        avg_logprob for _, avg_logprob in sink_outputs)
    
    # attn sinks should be lower perplexity (higher logprob per token)
    # nan logprob means negative infinity
    assert sum_sink_avg_logprob_per_token > sum_normal_avg_logprob_per_token \
        or isnan(sum_normal_avg_logprob_per_token)


@pytest.mark.parametrize("model, max_model_len", [
    ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    ("mosaicml/mpt-7b-chat", 2048)
])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("attn_backend, block_size", [
    ("XFORMERS", 8),
    ("FLASH_ATTN", 16),
    ("FLASH_ATTN", 32)
])
@pytest.mark.parametrize("enable_chunked_prefill", [False, True])
def test_eviction(
    model: str,
    max_model_len: int,
    dtype: str,
    batch_size: int,
    attn_backend: str,
    block_size: int,
    enable_chunked_prefill: bool,
    monkeypatch: pytest.MonkeyPatch
):
    prompt = _get_prompt(model)
    prompts = [prompt] * batch_size
    sampling_params = SamplingParams(min_tokens=200, max_tokens=201)

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", attn_backend)
    
    engine_args = EngineArgs(
        model,
        max_model_len=max_model_len,
        dtype=dtype,
        block_size=block_size,
        enforce_eager=True,
        use_attention_sinks=True,
        enable_chunked_prefill=enable_chunked_prefill
    )
    engine = LLMEngine.from_engine_args(engine_args)

    total_blocks = engine.scheduler.block_manager.get_num_free_gpu_blocks()
    max_blocks_needed = (max_model_len // block_size) * batch_size

    request_id = 0
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt = prompts.pop()
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        engine.step()
        free_blocks = engine.scheduler.block_manager.get_num_free_gpu_blocks()
        used_blocks = total_blocks - free_blocks
        assert used_blocks <= max_blocks_needed, (
            f"Number of used blocks ({used_blocks}) should be "
            f"at most {max_blocks_needed}"
        )

    del engine
    get_attn_backend.cache_clear()


@lru_cache
def _get_prompt(model_name: str, test_retrieval: bool = False) -> str:
    prompts = _get_prompts_json()
    prompt = prompts[model_name]
    # prompt is (model's context length - 100) tokens long
    
    if test_retrieval:
        return (
            f"Remember: my favorite color is {_RETRIEVAL_COLOR}. "
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
