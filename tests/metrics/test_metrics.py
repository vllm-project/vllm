import pytest
import vllm.engine.metrics

MODELS = [
    "facebook/opt-125m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_metrics(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model, dtype=dtype, disable_log_stats=False)
    tokenizer = vllm_model.model.get_tokenizer()
    prompt_token_counts = [len(tokenizer.encode(p)) for p in example_prompts]
    # This test needs at least 2 prompts in a batch of different lengths to verify their token count is correct despite padding.
    assert len(example_prompts) > 1, "at least 2 prompts are required"
    assert prompt_token_counts[0] != prompt_token_counts[1], (
        "prompts of different lengths are required")
    vllm_prompt_token_count = sum(prompt_token_counts)

    _ = vllm_model.generate_greedy(example_prompts, max_tokens)
    metric_count = vllm.engine.metrics.counter_prompt_tokens.get_value({})

    assert vllm_prompt_token_count == metric_count, (
        f"prompt token count: {vllm_prompt_token_count!r}\nmetric: {metric_count!r}"
    )
