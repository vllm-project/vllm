"""Compare the outputs of HF and vLLM when using greedy sampling for Mamba.

Run `pytest tests/models/test_mamba.py`.
"""
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams

from ...utils import check_outputs_equal

MODELS = ["state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-tiny-dev"]


# Use lower-level interfaces to create this greedy generator, as mamba will
# choke on the model_kwarg 'attention_mask' if hf_model.generate_greedy is used.
def generate_greedy(model_name, example_prompts, max_tokens):
    # Create a text generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Generate texts from the prompts
    outputs = []
    for prompt in example_prompts:
        # Tokenize the input prompt with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(model.device)

        # Generate text using the model's generate method directly
        generated_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        generated_text = tokenizer.decode(generated_ids[0],
                                          skip_special_tokens=True)

        outputs.append((generated_ids[0].tolist(), generated_text))

    return outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_outputs = generate_greedy(model, example_prompts, max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_batching(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    for_loop_outputs = []
    with vllm_runner(model, dtype=dtype) as vllm_model:
        for prompt in example_prompts:
            for_loop_outputs.append(
                vllm_model.generate_greedy([prompt], max_tokens)[0])

        batched_outputs = vllm_model.generate_greedy(example_prompts,
                                                     max_tokens)

    check_outputs_equal(
        outputs_0_lst=for_loop_outputs,
        outputs_1_lst=batched_outputs,
        name_0="for_loop_vllm",
        name_1="batched_vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [10])
def test_chunked_prefill_with_parallel_sampling(vllm_runner, example_prompts,
                                                model: str, dtype: str,
                                                max_tokens: int) -> None:
    # Tests chunked prefill in conjunction with n>1. In this case, prefill is
    # populated with decoding tokens and we test that it doesn't fail.
    # This test might fail if cache is not allocated correctly for n > 1
    # decoding steps inside a chunked prefill forward pass (where we have both
    # prefill and decode together )
    sampling_params = SamplingParams(n=3,
                                     temperature=1,
                                     seed=0,
                                     max_tokens=max_tokens)
    with vllm_runner(
            model,
            dtype=dtype,
            enable_chunked_prefill=True,
            max_num_batched_tokens=30,
            max_num_seqs=10  # forces prefill chunks with decoding
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 16])
def test_chunked_prefill(vllm_runner, example_prompts, model: str, dtype: str,
                         max_tokens: int,
                         chunked_prefill_token_size: int) -> None:
    """
    Checks exact match decode between huggingface model and vllm runner with
    chunked prefill.
    """
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    non_chunked = generate_greedy(model, example_prompts, max_tokens)

    with vllm_runner(model,
                     dtype=dtype,
                     enable_chunked_prefill=True,
                     max_num_batched_tokens=max_num_batched_tokens,
                     max_num_seqs=max_num_seqs) as vllm_model:
        chunked = vllm_model.generate_greedy(example_prompts,
                                             max_tokens=max_tokens)

    check_outputs_equal(
        outputs_0_lst=chunked,
        outputs_1_lst=non_chunked,
        name_0="chunked",
        name_1="non_chunked",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [15])
def test_parallel_sampling(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:

    with vllm_runner(model, dtype=dtype) as vllm_model:
        for_loop_outputs = []
        for _ in range(10):
            for_loop_outputs.append(
                # using example_prompts index 1 instead of 0 since with 0 the
                # logprobs get really close and the test doesn't pass
                vllm_model.generate_greedy([example_prompts[1]], max_tokens)
                [0])
        sampling_params = SamplingParams(n=10,
                                         temperature=0.001,
                                         seed=0,
                                         max_tokens=max_tokens)
        n_lt_1_outputs = vllm_model.generate([example_prompts[1]],
                                             sampling_params)
    token_ids, texts = n_lt_1_outputs[0]
    n_lt_1_outputs = [(token_id, text)
                      for token_id, text in zip(token_ids, texts)]

    check_outputs_equal(
        outputs_0_lst=n_lt_1_outputs,
        outputs_1_lst=for_loop_outputs,
        name_0="vllm_n_lt_1_outputs",
        name_1="vllm",
    )


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
    while len(example_prompts) == VllmConfig.get_graph_batch_size(
            len(example_prompts)):
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
@pytest.mark.parametrize("max_tokens", [20])
def test_models_preemption_recompute(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # Tests that outputs are identical with and w/o preemtions (recompute)
    assert dtype == "float"

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_model.model.llm_engine.scheduler[
            0].ENABLE_ARTIFICIAL_PREEMPT = True
        preempt_vllm_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

        vllm_model.model.llm_engine.scheduler[
            0].ENABLE_ARTIFICIAL_PREEMPT = False
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=preempt_vllm_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="vllm_preepmtions",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_fail_upon_inc_requests_and_finished_requests_lt_available_blocks(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    # This test is for verifying that the Mamba inner state management doesn't
    # collapse in case where the number of incoming requests and
    # finished_requests_ids is larger than the maximum Mamba block capacity.
    # This could generally happen due to the fact that Mamba does support
    # statelessness mechanism where it can cleanup new incoming requests in
    # a single step.
    try:
        with vllm_runner(model, dtype=dtype, max_num_seqs=10) as vllm_model:
            vllm_model.generate_greedy([example_prompts[0]] * 100, 10)
    except ValueError:
        pytest.fail("Mamba inner state wasn't cleaned up properly between"
                    "steps finished requests registered unnecessarily ")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_state_cleanup(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    # This test is for verifying that the Mamba state is cleaned up between
    # steps, If its not cleaned, an error would be expected.
    try:
        with vllm_runner(model, dtype=dtype) as vllm_model:
            for _ in range(10):
                vllm_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail("Mamba inner state wasn't cleaned up between states, "
                    "could be related to finished_requests_ids")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_multistep(
    vllm_runner,
    model: str,
    dtype: str,
    example_prompts,
) -> None:
    with vllm_runner(model, num_scheduler_steps=8,
                     max_num_seqs=2) as vllm_model:
        vllm_model.generate_greedy([example_prompts[0]] * 10, 1)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
def test_multistep_correctness(vllm_runner, model: str, dtype: str,
                               max_tokens: int, example_prompts) -> None:
    with vllm_runner(model, num_scheduler_steps=8,
                     max_num_seqs=2) as vllm_model:
        vllm_outputs_multistep = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    with vllm_runner(model, num_scheduler_steps=1,
                     max_num_seqs=2) as vllm_model:
        vllm_outputs_single_step = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_outputs_multistep,
        outputs_1_lst=vllm_outputs_single_step,
        name_0="vllm_outputs_multistep",
        name_1="vllm_outputs_single_step",
    )
