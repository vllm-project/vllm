import pytest

from vllm.worker.model_runner import _get_graph_batch_size

MODELS = ["ai21labs/Jamba-tiny-random"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [20])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    assert dtype == "float"

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [20])
def test_mamba_cache_cg_padding(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # This test is for verifying that mamba cache is padded to CG captured
    # batch size. If it's not, a torch RuntimeError will be raised because
    # tensor dimensions aren't compatible
    while len(example_prompts) == _get_graph_batch_size(len(example_prompts)):
        example_prompts.append(example_prompts[0])

    try:
        with vllm_runner(model, dtype=dtype) as vllm_model:
            vllm_model.generate_greedy(example_prompts, max_tokens)
    except RuntimeError:
        pytest.fail(
            "Couldn't run batch size which is not equal to a Cuda Graph "
            "captured batch size. "
            "Could be related to mamba cache not padded correctly")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_state_cleanup(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    # This test is for verifying that the Jamba state is cleaned up between
    # steps, If its not cleaned, an error would be expected.
    try:
        with vllm_runner(model, dtype=dtype) as vllm_model:
            for _ in range(10):
                vllm_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail("Jamba inner state wasn't cleaned up between states, "
                    "could be related to finished_requests_ids")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_model_print(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
