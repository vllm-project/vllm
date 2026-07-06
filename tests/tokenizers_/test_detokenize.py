# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)

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
    # FIXME: mosaicml/mpt-7b has been deleted
    # "mosaicml/mpt-7b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-3.2-1B-Instruct",
    "codellama/CodeLlama-7b-hf",
    "mistralai/Pixtral-12B-2409",
]


class MockLogprobsTokenizer:
    def __init__(self, raw_tokens: dict[int, str], decoded_tokens: dict[int, str]):
        self.raw_tokens = raw_tokens
        self.decoded_tokens = decoded_tokens

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        del skip_special_tokens
        return [self.raw_tokens[token_id] for token_id in ids]

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return "".join(self.decoded_tokens[token_id] for token_id in ids)


class MockSentencePieceBackend:
    def __init__(self, raw_tokens: dict[int, str]):
        self.raw_tokens = raw_tokens

    def id_to_piece(self, token_id: int) -> str:
        return self.raw_tokens[token_id]


class MockSentencePieceLogprobsTokenizer(MockLogprobsTokenizer):
    is_spm = True

    def __init__(
        self,
        raw_tokens: dict[int, str],
        decoded_tokens: dict[int, str],
        sentencepiece_pieces: dict[int, str],
    ):
        super().__init__(raw_tokens, decoded_tokens)
        self.tokenizer = MockSentencePieceBackend(sentencepiece_pieces)
        self.convert_ids_to_tokens_call_count = 0

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        self.convert_ids_to_tokens_call_count += 1
        return super().convert_ids_to_tokens(ids, skip_special_tokens)


def _make_top_logprobs(
    tokenizer: MockLogprobsTokenizer,
    token_ids: list[int],
    logprobs: list[float],
    requested_logprobs: int,
) -> dict[str, float]:
    requested_token_ids = token_ids[:requested_logprobs]
    decoded_tokens = convert_ids_list_to_tokens(tokenizer, requested_token_ids)
    return dict(zip(decoded_tokens, logprobs[:requested_logprobs]))


def _run_incremental_decode(
    tokenizer,
    all_input_ids,
    skip_special_tokens: bool,
    starting_index: int,
    spaces_between_special_tokens: bool = True,
    fast: bool | None = None,
):
    prompt_token_ids = all_input_ids[:starting_index]

    params = SamplingParams(
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=spaces_between_special_tokens,
    )
    request = EngineCoreRequest(
        request_id="",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    if fast is None:
        detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
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
    return (
        MistralTokenizer.from_pretrained(tokenizer_name)
        if "mistral" in tokenizer_name
        else AutoTokenizer.from_pretrained(tokenizer_name)
    )


@pytest.mark.parametrize("tokenizer_name", ["mistralai/Pixtral-12B-2409"])
@pytest.mark.parametrize(
    "truth",
    [
        # Burmese text triggers an edge-case where tokens may map to bytes with
        # incomplete UTF-8 characters
        "ပုံပြင်လေးပြောပြပါ",
        # Using "URGENCY" since "CY" has token id 130282
        "URGENCY🌶️",
    ],
)
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
        starting_index=starting_index,
    )
    assert decoded_text == truth
    assert out_ids == all_input_ids[starting_index:]


@pytest.fixture
def skip_special_tokens(request, tokenizer_name) -> Generator[bool, Any, None]:
    if "mistral" in tokenizer_name:
        yield (
            True
            if request.param
            else pytest.skip("mistral doesn't support skip_special_tokens=False")
        )
    else:
        yield bool(request.param)


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("with_prompt", [True, False])
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False), indirect=True)
@pytest.mark.parametrize("spaces_between_special_tokens", (True, False))
@pytest.mark.parametrize("fast", (True, False))
def test_decode_streaming(
    tokenizer,
    truth,
    with_prompt,
    skip_special_tokens,
    spaces_between_special_tokens,
    fast,
):
    if fast and not isinstance(tokenizer, PreTrainedTokenizerFast):
        pytest.skip()

    if skip_special_tokens and not spaces_between_special_tokens:
        pytest.skip()

    if not fast and isinstance(tokenizer, PreTrainedTokenizerFast):
        # Fix up inconsistency in fast/slow tokenizer behaviour.
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    at
                    for at in tokenizer._tokenizer.get_added_tokens_decoder().values()
                    if at.special
                ]
            }
        )

    extra_decode_args = (
        {}
        if not isinstance(tokenizer, PreTrainedTokenizer)
        else {"spaces_between_special_tokens": spaces_between_special_tokens}
    )

    truth_tokens = tokenizer(truth, add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        truth_tokens.insert(0, tokenizer.bos_token_id)
    truth_tokens.append(tokenizer.eos_token_id)

    new_truth = tokenizer.decode(
        truth_tokens, skip_special_tokens=skip_special_tokens, **extra_decode_args
    )

    if with_prompt:
        num_prompt_tokens = len(
            tokenizer(truth[: len(truth) // 2], add_special_tokens=False).input_ids
        )
        if tokenizer.bos_token_id is not None:
            num_prompt_tokens += 1

        prompt_input_ids = truth_tokens[:num_prompt_tokens]
        generated_input_ids = truth_tokens[num_prompt_tokens:]
        all_input_ids = prompt_input_ids + generated_input_ids
        starting_index = len(prompt_input_ids)
        prompt = tokenizer.decode(
            prompt_input_ids,
            skip_special_tokens=skip_special_tokens,
            **extra_decode_args,
        )

        generated = new_truth[len(prompt) :]
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
        fast=fast,
    )

    assert decoded_text == generated
    assert out_ids == all_input_ids[starting_index:]


@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("fast", (True, False))
def test_oov_decode(tokenizer, fast):
    if fast and not isinstance(tokenizer, PreTrainedTokenizerFast):
        pytest.skip()

    decoded_text, out_ids = _run_incremental_decode(
        tokenizer,
        [len(tokenizer)],
        skip_special_tokens=True,
        starting_index=0,
        spaces_between_special_tokens=True,
        fast=fast,
    )

    assert decoded_text == ""
    assert out_ids == [len(tokenizer)]


def test_convert_ids_list_to_tokens_preserves_sentencepiece_boundaries():
    tokenizer = MockLogprobsTokenizer(
        raw_tokens={0: "▁true", 1: "true", 2: "▁false", 3: "false"},
        decoded_tokens={0: "true", 1: "true", 2: "false", 3: "false"},
    )
    token_ids = [0, 1, 2, 3]

    decoded_tokens = convert_ids_list_to_tokens(tokenizer, token_ids)

    assert decoded_tokens == [" true", "true", " false", "false"]
    top_logprobs = _make_top_logprobs(
        tokenizer, token_ids, [-0.1, -0.2, -0.3, -0.4], requested_logprobs=4
    )
    assert len(top_logprobs) == 4


def test_convert_ids_list_to_tokens_prefers_sentencepiece_backend_pieces():
    tokenizer = MockSentencePieceLogprobsTokenizer(
        raw_tokens={0: "true", 1: "true"},
        decoded_tokens={0: "true", 1: "true"},
        sentencepiece_pieces={0: "▁true", 1: "true"},
    )

    assert convert_ids_list_to_tokens(tokenizer, [0, 1]) == [" true", "true"]
    assert tokenizer.convert_ids_to_tokens_call_count == 0


def test_convert_ids_list_to_tokens_preserves_readable_space_tokens():
    byte_level_tokenizer = MockLogprobsTokenizer(
        raw_tokens={0: "Ġword", 1: "word", 2: "Ġ", 3: "Ċ"},
        decoded_tokens={0: " word", 1: "word", 2: " ", 3: "\n"},
    )
    sentencepiece_tokenizer = MockLogprobsTokenizer(
        raw_tokens={0: "▁", 1: "▁▁"},
        decoded_tokens={0: "", 1: " "},
    )

    assert convert_ids_list_to_tokens(byte_level_tokenizer, [0, 1, 2, 3]) == [
        " word",
        "word",
        " ",
        "\n",
    ]
    assert convert_ids_list_to_tokens(sentencepiece_tokenizer, [0, 1]) == [
        " ",
        "  ",
    ]


def test_convert_ids_list_to_tokens_keeps_logprobs_count_and_values_stable():
    tokenizer = MockLogprobsTokenizer(
        raw_tokens={
            0: "▁true",
            1: "alpha",
            2: "beta",
            3: "gamma",
            4: "true",
            5: "delta",
            6: "epsilon",
            7: "zeta",
            8: "eta",
            9: "theta",
        },
        decoded_tokens={
            0: "true",
            1: "alpha",
            2: "beta",
            3: "gamma",
            4: "true",
            5: "delta",
            6: "epsilon",
            7: "zeta",
            8: "eta",
            9: "theta",
        },
    )
    token_ids = list(range(10))
    logprobs = [-0.1 * (idx + 1) for idx in range(10)]

    top4 = _make_top_logprobs(tokenizer, token_ids, logprobs, requested_logprobs=4)
    top10 = _make_top_logprobs(tokenizer, token_ids, logprobs, requested_logprobs=10)

    assert len(top4) == 4
    assert len(top10) == 10
    assert top4[" true"] == top10[" true"] == logprobs[0]
    assert top10["true"] == logprobs[4]
