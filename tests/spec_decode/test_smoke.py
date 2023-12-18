"""High-level speculative decoding tests."""
import time
import pytest
from typing import Generator
from itertools import cycle

from vllm import LLM, SamplingParams
from tests.spec_decode.utils import get_outputs, get_tokens_and_text


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("output_len", [1024])
@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("with_cuda_graph", [False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False])
@pytest.mark.parametrize("disable_shared_memory", [True])
def test_smoke_no_crash(spec_decode_llm: LLM, output_len: int,
                        temperature: float):
    """Validate that speculative decoding does not crash while generating a non-
    trivial number of tokens. This is a high-level test that validates different
    values of K and temperatures work over many generated tokens.
    """
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    print("Starting generation")
    start_time = time.time()
    outputs = spec_decode_llm.generate(prompts,
                                       sampling_params,
                                       use_tqdm=False)
    dur_ms = 1000 * (time.time() - start_time)
    num_output_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"generated {num_output_tokens} tokens in {dur_ms=:.02f}")
    print(f"ms/tok {dur_ms/num_output_tokens:.02f}")


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["meta-llama/Llama-2-7b-chat-hf"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("with_cuda_graph", [False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False])
@pytest.mark.parametrize("disable_shared_memory", [True])
def test_correctness_bs_1(spec_decode_llm_generator: Generator[LLM, None,
                                                               None],
                          non_spec_decode_llm_generator: Generator[LLM, None,
                                                                   None]):
    """High-level test that validates exact equality between normal decoding and
    speculative decoding. This is done via greedy sampling.

    Note that speculative decoding guarantees exact equality up to hardware
    numerics. The configuration tested here does not encounter numeric
    limitations, but may if ran on different hardware.
    """
    prompt = "The president of the United States is"
    output_len = 128
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    def evaluate(generator):
        for llm in generator:
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            token_ids = outputs[0].outputs[0].token_ids
            del llm
        return token_ids

    spec_token_ids = evaluate(spec_decode_llm_generator)
    non_spec_token_ids = evaluate(non_spec_decode_llm_generator)

    print(f"{len(spec_token_ids)=} {spec_token_ids=}")
    print(f"{len(non_spec_token_ids)=} {non_spec_token_ids=}")
    assert spec_token_ids == non_spec_token_ids


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["meta-llama/Llama-2-7b-chat-hf"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("with_cuda_graph", [False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False, True])
@pytest.mark.parametrize("disable_shared_memory", [True])
def test_correctness_bs_gt_1(spec_decode_llm_generator: Generator[LLM, None,
                                                                  None],
                             non_spec_decode_llm_generator: Generator[LLM,
                                                                      None,
                                                                      None]):
    """High-level test that validates exact correctness on a large batch size.
    Each sequence is compared with normal decoding and speculative decoding, and
    output tokens are compared one-by-one.

    See test_correctness_bs_1 for note on speculative decoding exact equality
    and hardware numerics.
    """
    base_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    batch_size = 64
    prompts = [
        prompt for prompt, _ in zip(cycle(base_prompts), range(batch_size))
    ]

    output_len = 32
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    spec_outputs = get_outputs(spec_decode_llm_generator, prompts,
                               sampling_params)
    non_spec_outputs = get_outputs(non_spec_decode_llm_generator, prompts,
                                   sampling_params)

    spec_text_outputs, spec_output_token_ids = get_tokens_and_text(
        spec_outputs)
    non_spec_text_outputs, non_spec_output_token_ids = get_tokens_and_text(
        non_spec_outputs)

    for i, (prompt, spec_text, spec_token_ids, non_spec_text,
            non_spec_token_ids) in enumerate(
                zip(prompts, spec_text_outputs, spec_output_token_ids,
                    non_spec_text_outputs, non_spec_output_token_ids)):
        print(f"{i=} {prompt=}")
        print(f"    {spec_text=}")
        print(f"{non_spec_text=}")
        print(f"{spec_token_ids=}")
        print(f"{non_spec_token_ids=}")

    for i, (prompt, spec_text, spec_token_ids, non_spec_text,
            non_spec_token_ids) in enumerate(
                zip(prompts, spec_text_outputs, spec_output_token_ids,
                    non_spec_text_outputs, non_spec_output_token_ids)):
        assert spec_token_ids == non_spec_token_ids, f"{i=}"


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("with_cuda_graph", [False])
@pytest.mark.parametrize("disable_shared_memory", [False])
@pytest.mark.parametrize("max_model_len", [200])
def test_correctness_model_truncation(
        max_model_len_spec_decode_generator: Generator[LLM, None, None],
        max_model_len_llm_generator: Generator[LLM, None,
                                               None], max_model_len: int):
    """Test correct generation when output must be truncated by max model len.
    """

    sampling_params = SamplingParams(
        max_tokens=max_model_len + 50,
        ignore_eos=True,
        temperature=0.0,
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    print("Starting generation")
    spec_outputs = get_outputs(max_model_len_spec_decode_generator, prompts,
                               sampling_params)
    spec_output_text, spec_output_token_ids = get_tokens_and_text(spec_outputs)

    non_spec_outputs = get_outputs(max_model_len_llm_generator, prompts,
                                   sampling_params)
    non_spec_output_text, non_spec_output_token_ids = get_tokens_and_text(
        non_spec_outputs)

    for i, prompt in enumerate(prompts):
        non_spec_text = non_spec_output_text[i]
        non_spec_token_ids = non_spec_output_token_ids[i]

        spec_text = spec_output_text[i]
        spec_token_ids = spec_output_token_ids[i]

        print(f"{i=} {prompt=}")
        print(f"{i=} {non_spec_text=}")
        print(f"{i=} {spec_text=}")
        print(f"{i=} {non_spec_token_ids=}")
        print(f"{i=} {spec_token_ids=}")

    for i, prompt in enumerate(prompts):
        non_spec_text = non_spec_output_text[i]
        non_spec_token_ids = non_spec_output_token_ids[i]

        spec_text = spec_output_text[i]
        spec_token_ids = spec_output_token_ids[i]

        assert non_spec_token_ids == spec_token_ids, f"{i=}"


def test_large_enough_cuda_graph_input():
    """Verify no crash when using CUDA graphs, particularly when the number of
    decode tokens is configured to exceed the nominal batch size. This happens
    when many tokens are accepted; the draft model must process up to bs*(k+1)
    tokens, the target model must process up to bs*2*(k+1).
    """
    draft_padding_size = 8
    target_padding_size = draft_padding_size

    batch_size = draft_padding_size * 3

    llm = LLM(
        # By setting the draft model and target model to the same model, we
        # should get a 100% acceptance rate.
        # This will maximize the number of decode tokens in the draft model
        # and target model forward passes.
        # Since the batch size is set to a multiple of the padding size, this
        # guarantees that we'll exceed the cuda graph input size unless it
        # accounts for extra speculative decode tokens.
        model="JackFram/llama-68m",
        speculative_model="JackFram/llama-68m",
        tensor_parallel_size=1,
        num_speculative_tokens=3,
        worker_use_ray=True,
        enable_cuda_graph=True,
        max_num_seqs=batch_size,
        target_model_input_padding_size=target_padding_size,
        draft_model_input_padding_size=draft_padding_size,
    )

    base_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [
        prompt for prompt, _ in zip(cycle(base_prompts), range(batch_size))
    ]

    output_len = 32
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )
    llm.generate(prompts, sampling_params, use_tqdm=False)
