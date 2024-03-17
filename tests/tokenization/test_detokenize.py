import pytest

from functools import partial
from transformers import AutoTokenizer
from typing import Callable, List, Dict
from unittest.mock import MagicMock

from vllm.engine.llm_engine import LLMEngine
from vllm.sequence import Sequence, Logprob
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group
from vllm.transformers_utils.tokenizer import detokenize_incrementally

TRUTH = [
    "Hello here, this is a simple test",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",  # noqa
    "我很感谢你的热情"
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


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("with_prompt", [True, False])
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False))
def test_decode_streaming(tokenizer_id, truth, with_prompt,
                          skip_special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if with_prompt:
        truth_tokens = tokenizer(truth, add_special_tokens=False)["input_ids"]
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
        all_input_ids = tokenizer(truth, add_special_tokens=False)["input_ids"]
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


@pytest.fixture(name="decode_sequence")
def create_decode_sequence(tokenizer_name: str) -> Callable:
    init_kwargs = dict(
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        revision=None,
    )

    self = MagicMock()
    self.tokenizer = get_tokenizer_group(
        None,
        **init_kwargs,
    )

    decode_sequence = partial(LLMEngine._decode_sequence, self)  # pylint: disable=protected-access
    return decode_sequence


@pytest.fixture(name="complete_sequence_token_ids")
def create_complete_sequence_token_ids(complete_sequence: str,
                                       tokenizer_name: str) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    complete_sequence_token_ids = tokenizer(complete_sequence)["input_ids"]
    return complete_sequence_token_ids


def create_empty_sequence():
    return Sequence(
        seq_id=0,
        prompt="<s>",
        prompt_token_ids=[1],
        block_size=16,
    )


def create_dummy_logprobs(
        complete_sequence_token_ids: List[int]) -> List[Dict[int, float]]:
    return [{
        token_id: Logprob(logprob=0.0)
    } for token_id in complete_sequence_token_ids]
