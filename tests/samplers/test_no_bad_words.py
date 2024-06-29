"""Make sure bad_words_ids works.

Run `pytest tests/samplers/test_no_bad_words.py`.

"""
import gc
from typing import List, Optional, Tuple

import pytest
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def _generate(
    model: LLM,
    prompt: str,
    num_prompt_tokens: int,
    temperature: float = 0,
    bad_words_ids: Optional[List[List[int]]] = None,
) -> List[int]:
    sampling_params = SamplingParams(
        temperature=temperature,
        bad_words_ids=bad_words_ids,
    )
    # output = model.generate(prompt, sampling_params=sampling_params)
    # output_token_ids = output[0].outputs[0].token_ids
    output = model.generate([prompt], sampling_params=sampling_params)
    # [([output_token_ids, ], [output_text, ]), ]
    # first (and only) request
    # token_ids
    # first output
    output_token_ids = output[0][0][0][num_prompt_tokens:]

    print(f'!!! {output}')
    print(f'!!! {output_token_ids}')
    print(f'!!! {num_prompt_tokens}')

    return output_token_ids


class _TestOneTokenBadWord:
    MODEL = "openai-community/gpt2"

    PROMPT = "Hi! How are"
    TARGET_TOKEN = "you"

    def setup_method(self, method):
        self.llm = LLM(self.MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, add_prefix_space=True)

        self.target_token_id = self.tokenizer(self.TARGET_TOKEN).input_ids[0]

    def teardown_method(self, method):
        del self.llm

        gc.collect()
        torch.cuda.empty_cache()

    def test_one_token_bad_word(self):
        output_token_ids = self._generate()
        assert output_token_ids[0] == self.target_token_id

        output_token_ids = self._generate(bad_words_ids=[[self.target_token_id]])
        assert self.target_token_id not in output_token_ids

    def _generate(self, bad_words_ids: Optional[List[List[int]]] = None) -> List[int]:
        return _generate(
            model=self.llm, prompt=self.PROMPT, bad_words_ids=bad_words_ids,
        )


class TestOneTokenBadWord:
    MODEL = "openai-community/gpt2"

    PROMPT = "Hi! How are"
    TARGET_TOKEN = "you"

    def setup_method(self, method):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, add_prefix_space=True)
        self.num_prompt_tokens = len(self.tokenizer(self.PROMPT).input_ids)
        self.target_token_id = self.tokenizer(self.TARGET_TOKEN).input_ids[0]

    def test_one_token_bad_word(self, vllm_runner):
        with vllm_runner(self.MODEL) as llm:
            output_token_ids = self._generate(llm)
            assert output_token_ids[0] == self.target_token_id

            output_token_ids = self._generate(llm, bad_words_ids=[[self.target_token_id]])
            assert self.target_token_id not in output_token_ids

    def _generate(self, model: LLM, bad_words_ids: Optional[List[List[int]]] = None) -> List[int]:
        return _generate(
            model=model,
            prompt=self.PROMPT,
            num_prompt_tokens=self.num_prompt_tokens,
            bad_words_ids=bad_words_ids,
        )


class _TestTwoTokenBadWord:
    MODEL = "openai-community/gpt2"

    PROMPT = "How old are you? I am 10"
    TARGET_TOKEN1 = "years"
    TARGET_TOKEN2 = "old"
    NEIGHBOUR_TOKEN2 = "older"

    def setup_method(self, method):
        self.llm = LLM(self.MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, add_prefix_space=True)

        self.target_token_id1 = self.tokenizer(self.TARGET_TOKEN1).input_ids[0]
        self.target_token_id2 = self.tokenizer(self.TARGET_TOKEN2).input_ids[0]
        self.neighbour_token_id2 = self.tokenizer(self.NEIGHBOUR_TOKEN2).input_ids[0]

    def teardown_method(self, method):
        del self.llm

        gc.collect()
        torch.cuda.empty_cache()

    def test_two_token_bad_word(self):
        output_token_ids = self._generate()
        assert output_token_ids[:2] == [self.target_token_id1, self.target_token_id2]

        output_token_ids = self._generate(bad_words_ids=[[self.target_token_id1]])
        assert self.target_token_id1 not in output_token_ids

        output_token_ids = self._generate(bad_words_ids=[[self.target_token_id2]])
        assert output_token_ids[0] == self.target_token_id1
        assert self.target_token_id2 not in output_token_ids

        output_token_ids = self._generate(bad_words_ids=[[self.target_token_id1, self.target_token_id2]])
        assert output_token_ids[0] == self.target_token_id1
        assert output_token_ids[:2] != [self.target_token_id1, self.target_token_id2]
        assert output_token_ids[:2] == [self.target_token_id1, self.neighbour_token_id2]

        output_token_ids = self._generate(
            bad_words_ids=[[self.target_token_id1, self.target_token_id2],
                           [self.target_token_id1, self.neighbour_token_id2]]
        )
        assert output_token_ids[0] == self.target_token_id1
        assert output_token_ids[:2] != [self.target_token_id1, self.target_token_id2]
        assert output_token_ids[:2] != [self.target_token_id1, self.neighbour_token_id2]
        assert (self.target_token_id2 in output_token_ids) or (self.neighbour_token_id2 in output_token_ids)

    def _generate(self, bad_words_ids: Optional[List[List[int]]] = None) -> List[int]:
        return _generate(
            model=self.llm, prompt=self.PROMPT, bad_words_ids=bad_words_ids,
        )


class TestTwoTokenBadWord:
    MODEL = "openai-community/gpt2"

    PROMPT = "How old are you? I am 10"
    TARGET_TOKEN1 = "years"
    TARGET_TOKEN2 = "old"
    NEIGHBOUR_TOKEN2 = "older"

    def setup_method(self, method):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, add_prefix_space=True)

        self.num_prompt_tokens = len(self.tokenizer(self.PROMPT).input_ids)
        self.target_token_id1 = self.tokenizer(self.TARGET_TOKEN1).input_ids[0]
        self.target_token_id2 = self.tokenizer(self.TARGET_TOKEN2).input_ids[0]
        self.neighbour_token_id2 = self.tokenizer(self.NEIGHBOUR_TOKEN2).input_ids[0]

    def test_two_token_bad_word(self, vllm_runner):
        with vllm_runner(self.MODEL) as llm:
            output_token_ids = self._generate(llm)
            assert output_token_ids[:2] == [self.target_token_id1, self.target_token_id2]

            output_token_ids = self._generate(llm, bad_words_ids=[[self.target_token_id1]])
            assert self.target_token_id1 not in output_token_ids

            output_token_ids = self._generate(llm, bad_words_ids=[[self.target_token_id2]])
            assert output_token_ids[0] == self.target_token_id1
            assert self.target_token_id2 not in output_token_ids

            output_token_ids = self._generate(llm, bad_words_ids=[[self.target_token_id1, self.target_token_id2]])
            assert output_token_ids[0] == self.target_token_id1
            assert output_token_ids[:2] != [self.target_token_id1, self.target_token_id2]
            assert output_token_ids[:2] == [self.target_token_id1, self.neighbour_token_id2]

            output_token_ids = self._generate(
                llm, bad_words_ids=[[self.target_token_id1, self.target_token_id2],
                                    [self.target_token_id1, self.neighbour_token_id2]]
            )
            assert output_token_ids[0] == self.target_token_id1
            assert output_token_ids[:2] != [self.target_token_id1, self.target_token_id2]
            assert output_token_ids[:2] != [self.target_token_id1, self.neighbour_token_id2]
            assert (self.target_token_id2 in output_token_ids) or (self.neighbour_token_id2 in output_token_ids)

    def _generate(self, model: LLM, bad_words_ids: Optional[List[List[int]]] = None) -> List[int]:
        return _generate(
            model=model,
            prompt=self.PROMPT,
            num_prompt_tokens=self.num_prompt_tokens,
            bad_words_ids=bad_words_ids,
        )
