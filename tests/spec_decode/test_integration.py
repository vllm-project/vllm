
import pytest
from typing import Generator
import torch

from vllm import LLM, SamplingParams
from tests.spec_decode.utils import (get_outputs, get_tokens_and_text,
                                     wait_for_gpu_memory_to_clear)
from tests.anyscale.utils import cleanup


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("output_len", [128])
@pytest.mark.parametrize("temperature", [1.0])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("with_cuda_graph", [True, False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False])
@pytest.mark.parametrize("disable_shared_memory", [False])
def test_integration_tp_and_cuda_graph(
        spec_decode_llm_generator: Generator[LLM, None, None], output_len: int,
        temperature: float, tensor_parallel_size: int):
    """Test integration with cuda graphs and different TP degrees.
    """
    run_test(spec_decode_llm_generator, output_len, temperature,
             tensor_parallel_size)


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("output_len", [128])
@pytest.mark.parametrize("temperature", [1.0])
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("with_cuda_graph", [True])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False])
@pytest.mark.parametrize("disable_shared_memory", [False, True])
def test_integration_shm(spec_decode_llm_generator: Generator[LLM, None, None],
                         output_len: int, temperature: float,
                         tensor_parallel_size: int):
    """Test integration with cuda graphs and shared memory.
    """
    run_test(spec_decode_llm_generator, output_len, temperature,
             tensor_parallel_size)


@pytest.mark.parametrize("draft_model", ["JackFram/llama-68m"])
@pytest.mark.parametrize("target_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("output_len", [128])
@pytest.mark.parametrize("temperature", [1.0])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("with_cuda_graph", [True, False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [True])
@pytest.mark.parametrize("disable_shared_memory", [False])
def test_integration_draft_tp1(spec_decode_llm_generator: Generator[LLM, None,
                                                                    None],
                               output_len: int, temperature: float,
                               tensor_parallel_size: int):
    """Test integration with different draft/target TP degrees
    """
    run_test(spec_decode_llm_generator, output_len, temperature,
             tensor_parallel_size)


def run_test(spec_decode_llm_generator: Generator[LLM, None, None],
             output_len: int, temperature: float, tensor_parallel_size: int):
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(f"Expected {tensor_parallel_size=} devices")

    print("waiting for free memory before test start")
    wait_for_gpu_memory_to_clear(list(range(torch.cuda.device_count())),
                                 1000 * 2**20)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    output_len = 128
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=output_len,
        ignore_eos=True,
    )

    spec_outputs = get_outputs(spec_decode_llm_generator, prompts,
                               sampling_params)
    cleanup()
    _, spec_output_token_ids = get_tokens_and_text(spec_outputs)

    # Assert enough expected tokens were returned.
    for token_ids in spec_output_token_ids:
        assert len(token_ids) == output_len


@pytest.mark.parametrize("draft_model", ["meta-llama/Llama-2-7b-chat-hf"])
@pytest.mark.parametrize("target_model", ["meta-llama/Llama-2-7b-chat-hf"])
@pytest.mark.parametrize("num_speculative_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", [4])
@pytest.mark.parametrize("with_cuda_graph", [False])
@pytest.mark.parametrize("disable_shared_memory", [False])
@pytest.mark.parametrize("speculative_model_uses_tp_1", [False])
def test_truncates_after_eos(spec_decode_llm_generator: Generator[LLM, None,
                                                                  None],
                             non_spec_decode_llm_generator: Generator[LLM,
                                                                      None,
                                                                      None],
                             tensor_parallel_size: int):
    """Since speculative decoding generates tokens in blocks, we verify that
    the engine truncates any tokens after EOS when EOS is to be respected.
    """
    # This test requires 26GB GPU memory (7B+7B), so it lives with the
    # distributed integration tests despite being a unit test.
    #
    # I couldn't reproduce the error case with any publicly available model
    # under 7B params.

    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(f"Expected {tensor_parallel_size=} devices")

    print("waiting for free memory before test start")
    wait_for_gpu_memory_to_clear(list(range(torch.cuda.device_count())),
                                 1000 * 2**20)

    sampling_params = SamplingParams(
        max_tokens=100,
        ignore_eos=False,
        temperature=0.0,
    )
    prompts = [
        ("[INST] <<SYS>>\nYou repeat the prompt exactly. You do not add "
         "additional thoughts or words\n<</SYS>>\n\n Repeat this exactly: "
         "'Hello world.'[/INST]"),
    ]

    print("Starting generation")
    spec_outputs = get_outputs(spec_decode_llm_generator, prompts,
                               sampling_params)
    spec_output_text, spec_output_token_ids = get_tokens_and_text(spec_outputs)

    non_spec_outputs = get_outputs(non_spec_decode_llm_generator, prompts,
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
        non_spec_token_ids = non_spec_output_token_ids[i]
        spec_token_ids = spec_output_token_ids[i]
        assert non_spec_token_ids == spec_token_ids, f"{i=}"
