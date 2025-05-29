# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from typing import Any, Optional

import pytest
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.inputs import token_inputs
from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (FastIncrementalDetokenizer,
                                        IncrementalDetokenizer,
                                        SlowIncrementalDetokenizer)

SPECIAL_TOKS_TRUTH = [
    "Some text with adjacent special tokens                <|padding|><|padding|><fim_prefix><fim_middle><fim_suffix>other text<fim_pad>",  # noqa
]

TRUTH = [
    "Hello here, this is a simple test",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",  # noqa
    "我很感谢你的热情",
    # Burmese text triggers an edge-case for Mistral's V3-Tekken tokenizer (eg.
    # for mistralai/Pixtral-12B-2409) where tokens may map to bytes with
    # incomplete UTF-8 characters
    # see https://github.com/vllm-project/vllm/pull/9625
    "ပုံပြင်လေးပြောပြပါ်",
] + SPECIAL_TOKS_TRUTH

TOKENIZERS = [
    "facebook/opt-125m",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-3.2-1B-Instruct",
    "codellama/CodeLlama-7b-hf",
    "mistralai/Pixtral-12B-2409",
]


def _run_incremental_decode(tokenizer,
                            all_input_ids,
                            skip_special_tokens: bool,
                            starting_index: int,
                            spaces_between_special_tokens: bool = True,
                            fast: Optional[bool] = None):

    prompt_token_ids = all_input_ids[:starting_index]

    params = SamplingParams(
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=spaces_between_special_tokens,
    )
    request = EngineCoreRequest("",
                                prompt_token_ids,
                                None,
                                None,
                                None,
                                params,
                                None,
                                0.0,
                                None,
                                cache_salt=None)

    if fast is None:
        detokenizer = IncrementalDetokenizer.from_new_request(
            tokenizer, request)
    elif fast:
        detokenizer = FastIncrementalDetokenizer(tokenizer, request)
    else:
        detokenizer = SlowIncrementalDetokenizer(tokenizer, request)

    output_text = ""
    for i, token_id in enumerate(all_input_ids[starting_index:]):
        detokenizer.update([token_id], False)
        finished = i == len(all_input_ids) - 1
        output_text += detokenizer.get_next_output_text(finished, delta=True)

    return output_text, detokenizer.output_token_ids


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

    decoded_text, out_ids = _run_incremental_decode(
        tokenizer,
        all_input_ids,
        skip_special_tokens=True,
        starting_index=starting_index)
    assert decoded_text == truth
    assert out_ids == all_input_ids[starting_index:]


@pytest.fixture
def skip_special_tokens(request, tokenizer_name) -> Generator[bool, Any, None]:
    if "mistral" in tokenizer_name:
        yield (
            True if request.param else
            pytest.skip("mistral doesn't support skip_special_tokens=False"))
    else:
        yield bool(request.param)


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("with_prompt", [True, False])
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False), indirect=True)
@pytest.mark.parametrize("spaces_between_special_tokens", (True, False))
@pytest.mark.parametrize("fast", (True, False))
def test_decode_streaming(tokenizer, truth, with_prompt, skip_special_tokens,
                          spaces_between_special_tokens, fast):
    if fast and not isinstance(tokenizer, PreTrainedTokenizerFast):
        pytest.skip()

    if skip_special_tokens and not spaces_between_special_tokens:
        pytest.skip()

    if not fast and isinstance(tokenizer, PreTrainedTokenizerFast):
        # Fix up inconsistency in fast/slow tokenizer behaviour.
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                at for at in
                tokenizer._tokenizer.get_added_tokens_decoder().values()
                if at.special
            ]
        })

    extra_decode_args = {} if not isinstance(tokenizer,  PreTrainedTokenizer) \
        else {"spaces_between_special_tokens": spaces_between_special_tokens}

    truth_tokens = tokenizer(truth, add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        truth_tokens.insert(0, tokenizer.bos_token_id)
    truth_tokens.append(tokenizer.eos_token_id)

    new_truth = tokenizer.decode(truth_tokens,
                                 skip_special_tokens=skip_special_tokens,
                                 **extra_decode_args)

    if with_prompt:
        num_prompt_tokens = len(
            tokenizer(truth[:len(truth) // 2],
                      add_special_tokens=False).input_ids)
        if tokenizer.bos_token_id is not None:
            num_prompt_tokens += 1

        prompt_input_ids = truth_tokens[:num_prompt_tokens]
        generated_input_ids = truth_tokens[num_prompt_tokens:]
        all_input_ids = prompt_input_ids + generated_input_ids
        starting_index = len(prompt_input_ids)
        prompt = tokenizer.decode(prompt_input_ids,
                                  skip_special_tokens=skip_special_tokens,
                                  **extra_decode_args)

        generated = new_truth[len(prompt):]
    else:
        generated = new_truth
        starting_index = 0
        all_input_ids = truth_tokens

    decoded_text, out_ids = _run_incremental_decode(
        tokenizer,
        all_input_ids,
        skip_special_tokens=skip_special_tokens,
        starting_index=starting_index,
        spaces_between_special_tokens=spaces_between_special_tokens,
        fast=fast)

    assert decoded_text == generated
    assert out_ids == all_input_ids[starting_index:]


@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("fast", (True, False))
def test_oov_decode(tokenizer, fast):
    if fast and not isinstance(tokenizer, PreTrainedTokenizerFast):
        pytest.skip()

    decoded_text, out_ids = _run_incremental_decode(
        tokenizer, [len(tokenizer)],
        skip_special_tokens=True,
        starting_index=0,
        spaces_between_special_tokens=True,
        fast=fast)

    assert decoded_text == ''
    assert out_ids == [len(tokenizer)]


@pytest.fixture
def detokenizer(tokenizer_name: str) -> Detokenizer:
    tokenizer_group = TokenizerGroup(
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="mistral" if "mistral" in tokenizer_name else "auto",
        trust_remote_code=False,
        revision=None,
    )

    return Detokenizer(tokenizer_group)


@pytest.fixture(name="complete_sequence_token_ids")
def create_complete_sequence_token_ids(complete_sequence: str,
                                       tokenizer) -> list[int]:
    return tokenizer(complete_sequence, add_special_tokens=False).input_ids


def create_sequence(prompt_token_ids=None):
    prompt_token_ids = prompt_token_ids or []
    return Sequence(
        seq_id=0,
        inputs=token_inputs(prompt_token_ids),
        block_size=16,
    )


def create_dummy_logprobs(
        complete_sequence_token_ids: list[int]) -> list[dict[int, Logprob]]:
    return [{
        token_id: Logprob(logprob=0.0),
        token_id + 1: Logprob(logprob=0.1)
    } for token_id in complete_sequence_token_ids]


def create_dummy_prompt_logprobs(
        complete_sequence_token_ids: list[int]
) -> list[Optional[dict[int, Any]]]:
    # logprob for the first prompt token is None.
    logprobs: list[Optional[dict[int, Any]]] = [None]
    logprobs.extend(create_dummy_logprobs(complete_sequence_token_ids)[1:])
    return logprobs


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", [True, False], indirect=True)
def test_decode_sequence_logprobs(complete_sequence: str,
                                  complete_sequence_token_ids: list[int],
                                  detokenizer: Detokenizer,
                                  skip_special_tokens: bool):
    """Verify Detokenizer decodes logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens,
                                     logprobs=2)

    # Run sequentially.
    seq = create_sequence()
    dummy_logprobs = create_dummy_logprobs(complete_sequence_token_ids)
    sequential_logprobs_text_chosen_token: list[str] = []
    sequential_logprobs_text_other_token: list[str] = []
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

    if not skip_special_tokens:
        # Text for logprobs for the chosen token should be the same as the
        # generated text. Note that this will only be true if we skip
        # special tokens.
        assert sequential_result == complete_sequence


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
def test_decode_prompt_logprobs(complete_sequence: str,
                                complete_sequence_token_ids: list[int],
                                detokenizer: Detokenizer):

    # We want to use skip_special_tokens=False here but Mistral tokenizers
    # don't support that.
    if complete_sequence not in SPECIAL_TOKS_TRUTH:
        skip_special_tokens = True
    elif not isinstance(detokenizer.tokenizer_group.get_lora_tokenizer(None),
                        MistralTokenizer):
        skip_special_tokens = False
    else:
        pytest.skip("MistralTokenizers don't support "
                    "skip_special_tokens=False")
        return
    """Verify Detokenizer decodes prompt logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens,
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
    decoded_prompt_logprobs: list[dict[int, Any]] = dummy_logprobs[
        1:]  # type: ignore

    # decoded_prompt_logprobs doesn't contain the first token.
    token_ids = complete_sequence_token_ids
    tokenizer = detokenizer.get_tokenizer_for_seq(seq)
    text_full = tokenizer.decode(token_ids,
                                 skip_special_tokens=skip_special_tokens)
    text_first = tokenizer.decode(token_ids[0],
                                  skip_special_tokens=skip_special_tokens)
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
    monkeypatch,
):
    # VLLM V1 does not use incremental detokenization for
    # prompt logprobs, so this test strategy is irrelevant.
    monkeypatch.setenv("VLLM_USE_V1", "0")

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
