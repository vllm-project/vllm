from typing import Any, Dict, Generator, List, Optional

import pytest
from transformers import AutoTokenizer

from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup
from vllm.transformers_utils.detokenizer import (Detokenizer,
                                                 detokenize_incrementally)
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

TRUTH = [
    "Hello here, this is a simple test",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",  # noqa
    "我很感谢你的热情",
    # Burmese text triggers an edge-case for Mistral's V3-Tekken tokenizer (eg.
    # for mistralai/Pixtral-12B-2409) where tokens may map to bytes with
    # incomplete UTF-8 characters
    # see https://github.com/vllm-project/vllm/pull/9625
    "ပုံပြင်လေးပြောပြပါ်",
]
TOKENIZERS = [
    "facebook/opt-125m",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-2-7b-hf",
    "codellama/CodeLlama-7b-hf",
    "mistralai/Pixtral-12B-2409",
]


def _run_incremental_decode(tokenizer, all_input_ids,
                            skip_special_tokens: bool, starting_index: int):
    decoded_text = ""
    offset = 0
    token_offset = 0
    prev_tokens = None
    for i in range(starting_index, len(all_input_ids)):
        new_tokens, text, offset, token_offset = detokenize_incrementally(
            tokenizer,
            all_input_ids[:i + 1],
            prev_tokens,
            offset,
            token_offset,
            skip_special_tokens=skip_special_tokens)
        decoded_text += text
        if prev_tokens is None:
            prev_tokens = new_tokens
        else:
            prev_tokens += new_tokens
    return decoded_text


@pytest.fixture
def tokenizer(tokenizer_name):
    return (MistralTokenizer.from_pretrained(tokenizer_name)
            if "mistral" in tokenizer_name else
            AutoTokenizer.from_pretrained(tokenizer_name))


@pytest.mark.parametrize("tokenizer_name", ["mistralai/Pixtral-12B-2409"])
@pytest.mark.parametrize(
    "truth",
    [
        # Burmese text triggers an edge-case where tokens may map to bytes with
        # incomplete UTF-8 characters
        "ပုံပြင်လေးပြောပြပါ",
        # Using "URGENCY" since "CY" has token id 130282
        "URGENCY🌶️",
    ])
def test_mistral_edge_case(tokenizer, truth):
    """Test for a specific edge cases with V3-Tekken MistralTokenizer.

    See https://github.com/vllm-project/vllm/pull/9625
    """
    starting_index = 0
    all_input_ids = tokenizer(truth, add_special_tokens=False).input_ids

    decoded_text = _run_incremental_decode(tokenizer,
                                           all_input_ids,
                                           skip_special_tokens=True,
                                           starting_index=starting_index)
    assert decoded_text == truth


@pytest.fixture
def skip_special_tokens(request, tokenizer_name) -> Generator[bool, Any, None]:
    if "mistral" in tokenizer_name:
        yield (
            bool(True) if request.param else
            pytest.skip("mistral doesn't support skip_special_tokens=False"))
    else:
        yield bool(True) if request.param else bool(False)


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("with_prompt", [True, False])
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False), indirect=True)
def test_decode_streaming(tokenizer, truth, with_prompt, skip_special_tokens):
    if with_prompt:
        truth_tokens = tokenizer(truth, add_special_tokens=False).input_ids
        prompt_input_ids = truth_tokens[:len(truth) // 2]
        generated_input_ids = truth_tokens[len(truth) // 2:]
        all_input_ids = prompt_input_ids + generated_input_ids
        starting_index = len(prompt_input_ids)
        prompt = tokenizer.decode(prompt_input_ids,
                                  skip_special_tokens=skip_special_tokens)
        generated = truth[len(prompt):]
    else:
        generated = truth
        starting_index = 0
        all_input_ids = tokenizer(truth, add_special_tokens=False).input_ids
    if skip_special_tokens:
        if tokenizer.bos_token_id is not None:
            all_input_ids = [tokenizer.bos_token_id] + all_input_ids
            starting_index += 1
        all_input_ids = all_input_ids + [tokenizer.eos_token_id]

    decoded_text = _run_incremental_decode(
        tokenizer,
        all_input_ids,
        skip_special_tokens=skip_special_tokens,
        starting_index=starting_index)

    assert decoded_text == generated

    decoded_text = _run_incremental_decode(
        tokenizer, [len(tokenizer)],
        skip_special_tokens=skip_special_tokens,
        starting_index=starting_index)

    assert decoded_text == ''


@pytest.fixture
def detokenizer(tokenizer_name: str) -> Detokenizer:
    init_kwargs = dict(
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="mistral" if "mistral" in tokenizer_name else "auto",
        trust_remote_code=False,
        revision=None,
    )

    tokenizer_group = get_tokenizer_group(
        None,
        **init_kwargs,
    )

    return Detokenizer(tokenizer_group)


@pytest.fixture(name="complete_sequence_token_ids")
def create_complete_sequence_token_ids(complete_sequence: str,
                                       tokenizer) -> List[int]:
    complete_sequence_token_ids = tokenizer(complete_sequence).input_ids
    return complete_sequence_token_ids


def create_sequence(prompt_token_ids=None):
    prompt_token_ids = prompt_token_ids or [1]
    return Sequence(
        seq_id=0,
        inputs={
            "prompt": "<s>",
            "prompt_token_ids": prompt_token_ids,
        },
        block_size=16,
    )


def create_dummy_logprobs(
        complete_sequence_token_ids: List[int]) -> List[Dict[int, Logprob]]:
    return [{
        token_id: Logprob(logprob=0.0),
        token_id + 1: Logprob(logprob=0.1)
    } for token_id in complete_sequence_token_ids]


def create_dummy_prompt_logprobs(
        complete_sequence_token_ids: List[int]
) -> List[Optional[Dict[int, Any]]]:
    # logprob for the first prompt token is None.
    logprobs: List[Optional[Dict[int, Any]]] = [None]
    logprobs.extend(create_dummy_logprobs(complete_sequence_token_ids)[1:])
    return logprobs


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", [True, False], indirect=True)
def test_decode_sequence_logprobs(complete_sequence: str,
                                  complete_sequence_token_ids: List[int],
                                  detokenizer: Detokenizer,
                                  skip_special_tokens: bool):
    """Verify Detokenizer decodes logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens,
                                     logprobs=2)

    # Run sequentially.
    seq = create_sequence()
    dummy_logprobs = create_dummy_logprobs(complete_sequence_token_ids)
    sequential_logprobs_text_chosen_token: List[str] = []
    sequential_logprobs_text_other_token: List[str] = []
    for new_token, logprobs in zip(complete_sequence_token_ids,
                                   dummy_logprobs):
        seq.append_token_id(new_token, logprobs)
        detokenizer.decode_sequence_inplace(seq, sampling_params)
        sequential_logprobs_text_chosen_token.append(
            seq.output_logprobs[-1][new_token].decoded_token)
        sequential_logprobs_text_other_token.append(
            seq.output_logprobs[-1][new_token + 1].decoded_token)
    sequential_result = seq.output_text

    assert sequential_result == "".join(sequential_logprobs_text_chosen_token)
    assert sequential_result != "".join(sequential_logprobs_text_other_token)

    if skip_special_tokens:
        # Text for logprobs for the chosen token should be the same as the
        # generated text. Note that this will only be true if we skip
        # special tokens.
        assert sequential_result == complete_sequence


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
def test_decode_prompt_logprobs(complete_sequence_token_ids: List[int],
                                detokenizer: Detokenizer):
    """Verify Detokenizer decodes prompt logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=True,
                                     prompt_logprobs=1)

    # Run sequentially.
    seq = create_sequence(complete_sequence_token_ids)
    seq_group = SequenceGroup(request_id="1",
                              seqs=[seq],
                              sampling_params=sampling_params,
                              arrival_time=0.0)
    dummy_logprobs = create_dummy_prompt_logprobs(complete_sequence_token_ids)
    detokenizer.decode_prompt_logprobs_inplace(seq_group,
                                               dummy_logprobs,
                                               position_offset=0)
    # First logprob is None.
    decoded_prompt_logprobs: List[Dict[int, Any]] = dummy_logprobs[
        1:]  # type: ignore

    # decoded_prompt_logprobs doesn't contain the first token.
    token_ids = complete_sequence_token_ids
    tokenizer = detokenizer.get_tokenizer_for_seq(seq)
    text_full = tokenizer.decode(token_ids, skip_special_tokens=True)
    text_first = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    text = text_full[len(text_first):]

    # Text for logprobs for the chosen token should be the same as the
    # prompt text. Note that the first logprob is None.
    assert text == "".join([
        logprobs[token_id].decoded_token
        for token_id, logprobs in zip(token_ids[1:], decoded_prompt_logprobs)
    ])
    assert text != "".join([
        logprobs[token_id + 1].decoded_token
        for token_id, logprobs in zip(token_ids[1:], decoded_prompt_logprobs)
    ])


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 7, 16, -1])
def test_decode_prompt_logprobs_chunked_prefill(
    vllm_runner,
    model,
    chunked_prefill_token_size: int,
    example_prompts,
):
    max_num_seqs = 256
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_seqs = min(chunked_prefill_token_size, max_num_seqs)
        max_num_batched_tokens = chunked_prefill_token_size

    with vllm_runner(model,
                     dtype="half",
                     max_logprobs=5,
                     gpu_memory_utilization=0.5,
                     enable_chunked_prefill=enable_chunked_prefill,
                     max_num_batched_tokens=max_num_batched_tokens,
                     max_num_seqs=max_num_seqs) as vllm_model:

        vllm_sampling_params = SamplingParams(max_tokens=10,
                                              logprobs=5,
                                              prompt_logprobs=5,
                                              temperature=0.0)
        vllm_results = vllm_model.model.generate(
            example_prompts, sampling_params=vllm_sampling_params)

        for idx, result in enumerate(vllm_results):
            assert result.prompt_logprobs is not None
            assert result.prompt_logprobs[0] is None

            # Compared detokenized prompts ids to original prompt.
            generated_string = ""
            for (prompt_token,
                 prompt_logprobs) in zip(result.prompt_token_ids[1:],
                                         result.prompt_logprobs[1:]):
                # prompt_logprobs is a dict of the token_id: logprob
                # We select the token_id corresponding to the actual prompt
                # Decoded token in the detokenized string corresponding to this
                # prompt token.
                generated_string += prompt_logprobs[prompt_token].decoded_token

            assert generated_string == example_prompts[idx], (
                "Detokenized prompt logprobs do not match original prompt")
