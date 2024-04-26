from typing import List

import pytest

MODELS = [
    "facebook/opt-125m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_metric_counter_prompt_tokens(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.4)
    tokenizer = vllm_model.model.get_tokenizer()
    prompt_token_counts = [len(tokenizer.encode(p)) for p in example_prompts]
    # This test needs at least 2 prompts in a batch of different lengths to
    # verify their token count is correct despite padding.
    assert len(example_prompts) > 1, "at least 2 prompts are required"
    assert prompt_token_counts[0] != prompt_token_counts[1], (
        "prompts of different lengths are required")
    vllm_prompt_token_count = sum(prompt_token_counts)

    _ = vllm_model.generate_greedy(example_prompts, max_tokens)
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metric_count = stat_logger.metrics.counter_prompt_tokens.labels(
        **stat_logger.labels)._value.get()

    assert vllm_prompt_token_count == metric_count, (
        f"prompt token count: {vllm_prompt_token_count!r}\n"
        f"metric: {metric_count!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_metric_counter_generation_tokens(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.4)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    tokenizer = vllm_model.model.get_tokenizer()
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metric_count = stat_logger.metrics.counter_generation_tokens.labels(
        **stat_logger.labels)._value.get()
    vllm_generation_count = 0
    for i in range(len(example_prompts)):
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        prompt_ids = tokenizer.encode(example_prompts[i])
        # vllm_output_ids contains both prompt tokens and generation tokens.
        # We're interested only in the count of the generation tokens.
        vllm_generation_count += len(vllm_output_ids) - len(prompt_ids)

    assert vllm_generation_count == metric_count, (
        f"generation token count: {vllm_generation_count!r}\n"
        f"metric: {metric_count!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize(
    "served_model_name",
    [None, [], ["ModelName0"], ["ModelName0", "ModelName1", "ModelName2"]])
def test_metric_set_tag_model_name(vllm_runner, model: str, dtype: str,
                                   served_model_name: List[str]) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.3,
                             served_model_name=served_model_name)
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metrics_tag_content = stat_logger.labels["model_name"]

    del vllm_model

    if served_model_name is None or served_model_name == []:
        assert metrics_tag_content == model, (
            f"Metrics tag model_name is wrong! expect: {model!r}\n"
            f"actual: {metrics_tag_content!r}")
    else:
        assert metrics_tag_content == served_model_name[0], (
            f"Metrics tag model_name is wrong! expect: "
            f"{served_model_name[0]!r}\n"
            f"actual: {metrics_tag_content!r}")
