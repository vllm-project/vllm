"""Compare the outputs of HF and vLLM when using beam search.

Run `pytest tests/samplers/test_beam_search.py`.
"""

import pytest

from vllm.sequence import Sequence, SequenceData

# FIXME(zhuohan): The test can not pass if we:
#   1. Increase max_tokens to 256.
#   2. Increase beam_width to 8.
#   3. Use the model "huggyllama/llama-7b".
MAX_TOKENS = [128]
BEAM_WIDTHS = [4]
MODELS = ["facebook/opt-125m"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_single_input(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(example_prompts,
                                                       beam_width, max_tokens)

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        vllm_output_ids, _ = vllm_outputs[i]
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"vLLM: {vllm_output_ids}")


def test_get_beam_search_score():
    # Create a Sequence object with dummy prompt and output tokens
    prompt_tokens = [1, 2, 3, 4]
    output_tokens = [5, 6, 7]
    seq = Sequence(seq_id=0,
                   block_size=32,
                   inputs={
                       "prompt": "dummy prompt",
                       "prompt_token_ids": prompt_tokens
                   })
    seq.data = SequenceData(prompt_tokens, output_tokens)
    # Set a dummy cumulative logprob
    seq.data.cumulative_logprob = -10.0

    expected_score = seq.data.cumulative_logprob / len(output_tokens)
    actual_score = seq.get_beam_search_score()

    assert actual_score == pytest.approx(expected_score)
