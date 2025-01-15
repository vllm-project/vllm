"""Engine test fixtures"""
import pytest
from transformers import AutoTokenizer

from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from tests.v1.engine.utils import (
    DummyOutputProcessorTestVectors,
    generate_dummy_sample_logprobs,
    generate_dummy_prompt_logprobs,
    TOKENIZER_NAME,
    FULL_STRINGS,
    PROMPT_LEN,
    NUM_SAMPLE_LOGPROBS,
    NUM_PROMPT_LOGPROBS,
)

@pytest.fixture
def dummy_test_vectors() -> DummyOutputProcessorTestVectors:
    """Generate dummy test vectors for detokenizer tests.
    
    Returns:
      DummyTestVectors instance
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vllm_config = EngineArgs(model=TOKENIZER_NAME).create_engine_config()
    # Tokenize prompts under test & create dummy generated tokens
    prompt_tokens = [
        tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS
    ]
    generation_tokens = [
        tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
    ]
    # Generate prompt strings
    prompt_strings = [
        tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        for prompt_tokens in prompt_tokens
    ]
    prompt_strings_len = [
        len(prompt_string) for prompt_string in prompt_strings
    ]
    return DummyOutputProcessorTestVectors(
        tokenizer=tokenizer,
        tokenizer_group=init_tokenizer_from_configs(
            vllm_config.model_config, vllm_config.scheduler_config,
            vllm_config.parallel_config, vllm_config.lora_config),
        vllm_config=vllm_config,
        full_tokens=[tokenizer(text).input_ids for text in FULL_STRINGS],
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_strings=prompt_strings,
        prompt_strings_len=prompt_strings_len,
        generation_strings=[
            text[prompt_len:]
            for text, prompt_len in zip(FULL_STRINGS, prompt_strings_len)
        ],
        prompt_logprobs=[],
        generation_logprobs=[])


@pytest.fixture
def dummy_test_vectors_with_logprobs() -> DummyOutputProcessorTestVectors:
    """Generate dummy test vectors for detokenizer tests.
    
    Returns:
      DummyTestVectors instance
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vllm_config = EngineArgs(model=TOKENIZER_NAME).create_engine_config()
    # Tokenize prompts under test & create dummy generated tokens
    prompt_tokens = [
        tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS
    ]
    generation_tokens = [
        tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
    ]
    # Generate prompt strings
    prompt_strings = [
        tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        for prompt_tokens in prompt_tokens
    ]
    prompt_strings_len = [
        len(prompt_string) for prompt_string in prompt_strings
    ]
    return DummyOutputProcessorTestVectors(
        tokenizer=tokenizer,
        tokenizer_group=init_tokenizer_from_configs(
            vllm_config.model_config, vllm_config.scheduler_config,
            vllm_config.parallel_config, vllm_config.lora_config),
        vllm_config=vllm_config,
        full_tokens=[tokenizer(text).input_ids for text in FULL_STRINGS],
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_strings=prompt_strings,
        prompt_strings_len=prompt_strings_len,
        generation_strings=[
            text[prompt_len:]
            for text, prompt_len in zip(FULL_STRINGS, prompt_strings_len)
        ],
        prompt_logprobs=[
            generate_dummy_prompt_logprobs(prompt_tokens_list=tokens_list,
                                           num_logprobs=NUM_PROMPT_LOGPROBS,
                                           tokenizer=tokenizer)
            for tokens_list in prompt_tokens
        ],
        generation_logprobs=[
            generate_dummy_sample_logprobs(sampled_tokens_list=tokens_list,
                                           num_logprobs=NUM_SAMPLE_LOGPROBS,
                                           tokenizer=tokenizer)
            for tokens_list in generation_tokens
        ])