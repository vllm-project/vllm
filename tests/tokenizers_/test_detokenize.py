# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from typing import Any

import pytest
from transformers import AutoTokenizer, PythonBackend, TokenizersBackend

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
    "openai-community/gpt2",
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
    if fast and not isinstance(tokenizer, TokenizersBackend):
        pytest.skip()

    if skip_special_tokens and not spaces_between_special_tokens:
        pytest.skip()

    if not fast and isinstance(tokenizer, TokenizersBackend):
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
        if not isinstance(tokenizer, PythonBackend)
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
    if fast and not isinstance(tokenizer, TokenizersBackend):
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


# ---------- convert_ids_list_to_tokens collision tests ----------


class _MockBackend:
    """Fake backend_tokenizer that exposes pre_tokenizer config."""

    def __init__(self, pre_tokenizer_type, replacement=None):
        import json

        pre: dict = {"type": pre_tokenizer_type}
        if replacement is not None:
            pre["replacement"] = replacement
        self._config = json.dumps({"pre_tokenizer": pre})

    def to_str(self):
        return self._config


class _MockTokenizer:
    """Minimal tokenizer mock for testing convert_ids_list_to_tokens."""

    def __init__(
        self,
        raw_tokens: dict[int, str],
        decoded_tokens: dict[int, str],
        pre_tokenizer_type: str = "Metaspace",
        replacement: str | None = "▁",
    ):
        self._raw = raw_tokens
        self._decoded = decoded_tokens
        self.backend_tokenizer = _MockBackend(pre_tokenizer_type, replacement)

    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]:
        return [self._raw[tid] for tid in ids]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self._decoded[tid] for tid in ids)


def test_sentencepiece_leading_space_preserved():
    """▁true and true must produce distinct strings."""
    tok = _MockTokenizer(
        raw_tokens={0: "▁true", 1: "true", 2: "▁false", 3: "false"},
        decoded_tokens={0: "true", 1: "true", 2: "false", 3: "false"},
    )
    result = convert_ids_list_to_tokens(tok, [0, 1, 2, 3])
    assert result == [" true", "true", " false", "false"]

    # No dict collision when used as top_logprobs keys
    logprobs = dict(zip(result, [-0.1, -0.2, -0.3, -0.4]))
    assert len(logprobs) == 4


def test_whitespace_run_tokens_stay_distinct():
    """▁, ▁▁, ▁▁▁ must produce different-length space strings."""
    tok = _MockTokenizer(
        raw_tokens={0: "▁", 1: "▁▁", 2: "▁▁▁"},
        decoded_tokens={0: "", 1: " ", 2: "  "},
    )
    result = convert_ids_list_to_tokens(tok, [0, 1, 2])
    assert result == [" ", "  ", "   "]


def test_bpe_leading_space_already_preserved():
    """GPT-2 BPE: Ġtrue already decodes to ' true', no fix needed."""
    tok = _MockTokenizer(
        raw_tokens={0: "Ġtrue", 1: "true"},
        decoded_tokens={0: " true", 1: "true"},
        pre_tokenizer_type="ByteLevel",
        replacement=None,
    )
    result = convert_ids_list_to_tokens(tok, [0, 1])
    assert result == [" true", "true"]


def test_logprobs_count_stable_across_k():
    """logprobs=4 and logprobs=10 must return 4 and 10 entries."""
    tok = _MockTokenizer(
        raw_tokens={
            0: "▁true",
            1: "a",
            2: "b",
            3: "c",
            4: "true",
            5: "d",
            6: "e",
            7: "f",
            8: "g",
            9: "h",
        },
        decoded_tokens={
            0: "true",
            1: "a",
            2: "b",
            3: "c",
            4: "true",
            5: "d",
            6: "e",
            7: "f",
            8: "g",
            9: "h",
        },
    )
    ids = list(range(10))
    lps = [-0.1 * (i + 1) for i in range(10)]

    tokens4 = convert_ids_list_to_tokens(tok, ids[:4])
    top4 = dict(zip(tokens4, lps[:4]))

    tokens10 = convert_ids_list_to_tokens(tok, ids)
    top10 = dict(zip(tokens10, lps))

    assert len(top4) == 4
    assert len(top10) == 10
    assert top4[" true"] == top10[" true"]
