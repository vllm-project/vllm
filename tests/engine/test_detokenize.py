
import os
import pytest

from functools import partial
from transformers import AutoTokenizer
from typing import Callable, List, Dict
from unittest.mock import MagicMock

from vllm.anyscale.tokenization import TransformersTokenizer
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence
from vllm.transformers_utils.tokenizer import detokenize_incrementally

TRUTH = [
    # pylint: disable=line-too-long
    "Hello here, this is a simple test",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",
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
                            skip_special_tokens: bool):
    decoded_text = ""
    offset = 0
    token_offset = 0
    prev_tokens = None
    for i in range(len(all_input_ids)):
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


@pytest.mark.skipif("HUGGING_FACE_HUB_TOKEN" not in os.environ,
                    reason="requires HF token")
@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False))
def test_decode_streaming(tokenizer_id, truth, skip_special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    all_input_ids = tokenizer(truth, add_special_tokens=False)["input_ids"]
    if skip_special_tokens:
        all_input_ids = ([tokenizer.bos_token_id]
                         if tokenizer.bos_token_id is not None else
                         []) + all_input_ids + [tokenizer.eos_token_id]

    decoded_text = _run_incremental_decode(
        tokenizer, all_input_ids, skip_special_tokens=skip_special_tokens)

    assert decoded_text == truth


@pytest.mark.skipif("HUGGING_FACE_HUB_TOKEN" not in os.environ,
                    reason="requires HF token")
@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_decode_sequence_works_with_multiple_tokens(
        complete_sequence_token_ids: List[int],
        dummy_logprobs: List[Dict[int, float]], decode_sequence: Callable,
        skip_special_tokens: bool):
    """Verify LLMEngine can decode sequences with >1 new tokens per step.
    """
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens)

    # Run sequentially.
    seq = create_empty_sequence()
    for new_token, logprob in zip(complete_sequence_token_ids, dummy_logprobs):
        seq.append_token_ids([new_token], [logprob])
        decode_sequence(seq, sampling_params)
    sequential_result = seq.output_text

    # Run in batch.
    seq = create_empty_sequence()
    seq.append_token_ids(complete_sequence_token_ids, dummy_logprobs)
    decode_sequence(seq, sampling_params)
    batch_result = seq.output_text

    assert sequential_result == batch_result


@pytest.fixture(name="dummy_logprobs")
def create_dummy_logprobs(
        complete_sequence_token_ids: List[int]) -> List[Dict[int, float]]:
    return list({token_id: 0.0} for token_id in complete_sequence_token_ids)


@pytest.fixture(name="decode_sequence")
def create_decode_sequence(tokenizer_name: str) -> Callable:
    init_kwargs = dict(
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        revision=None,
    )

    self = MagicMock()
    self.tokenizer = TransformersTokenizer(
        tokenizer_name,
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
        prompt="",
        prompt_token_ids=[],
        block_size=16,
    )
