"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import random
from typing import List, Optional

import pytest

from tests.kernels.utils import override_backend_env_variable
from vllm import SamplingParams, TokensPrompt
from vllm.transformers_utils.tokenizer import get_tokenizer

from ..models.utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
]

UNSTABLE_PROMPT_SEQUENCE = [
    ([0] * 588) + ([1] * 1332) + ([2] * 30) + ([3] * 1),
    ([0] * 588) + ([1] * 1332) + ([4] * 3) + ([5] * 50),
    ([0] * 588) + ([1] * 1332) + ([2] * 30) + ([6] * 95),
    ([0] * 588) + ([1] * 1332) + ([4] * 3) + ([7] * 174),
    ([0] * 588) + ([8] * 1539),
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER", "XFORMERS"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("cached_position", [0, 1])
@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
@pytest.mark.parametrize("block_size", [16])
def test_mixed_requests(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    cached_position: int,
    enable_chunked_prefill: bool,
    block_size: int,
    monkeypatch,
) -> None:
    """
    Test the case when some sequences have the prefix cache hit
    and the others don't. The cached position determines where
    the sequence is at among the batch of prefills.
    """
    override_backend_env_variable(monkeypatch, backend)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    cached_prompt = example_prompts[cached_position]
    with vllm_runner(
            model,
            dtype=dtype,
            enable_prefix_caching=True,
            enable_chunked_prefill=enable_chunked_prefill,
            block_size=block_size,
    ) as vllm_model:
        # Run the first prompt so the cache is populated
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens)

        # Run all the promopts
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        req_outputs = vllm_model.model.generate(example_prompts, greedy_params)

        # Verify number of cached tokens
        for i in range(len(req_outputs)):
            if i == cached_position:
                expected_num_cached_tokens = (
                    len(req_outputs[i].prompt_token_ids) //
                    block_size) * block_size
            else:
                expected_num_cached_tokens = 0
            assert req_outputs[
                i].num_cached_tokens == expected_num_cached_tokens

        vllm_outputs = [
            (output.prompt_token_ids + list(output.outputs[0].token_ids),
             output.prompt + output.outputs[0].text) for output in req_outputs
        ]

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("chunk_size", [None, 64])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prefix_ratio", [0.0, 0.5, 1.0])
def test_partially_cached_prefills(
    vllm_runner,
    model: str,
    chunk_size: Optional[int],
    prefix_ratio: float,
) -> None:
    """ """
    # Prompts with various lengths.
    prompt_lens = [16, 32, 63, 64, 65, 127, 128, 256, 512]

    # NOTE(rickyx): We could potentially run against HF here and with a longer
    # decode len, but seems there's quite some invariance in the generated
    # tokens, i.e. decoded tokens might mismatch indeterministically across
    # runs. There's even mismatch between non-prefix-cached enabled and HF
    # ones.
    max_tokens = 1
    max_model_len = 2 * max(prompt_lens)
    if chunk_size is None:
        max_num_batched_tokens = max_model_len
        max_num_seqs = len(prompt_lens)
    else:
        max_num_batched_tokens = chunk_size
        max_num_seqs = chunk_size

    tokenizer = get_tokenizer(model, trust_remote_code=True)
    random.seed(0)
    vocab = list(tokenizer.get_vocab().values())

    batch_prompts: List[List[str]] = []
    for prompt_len in prompt_lens:
        num_prefix_tokens = int(prefix_ratio * prompt_len)
        num_unique_tokens = prompt_len - num_prefix_tokens

        prefix_token_ids = random.choices(vocab, k=num_prefix_tokens)
        unique_token_ids = random.choices(vocab, k=num_unique_tokens)

        prompt_token_ids = prefix_token_ids + unique_token_ids
        prompt_str = tokenizer.decode(prompt_token_ids)

        # First 2 prompts might not have prefix cache hit even with shared
        # prefix cache since not yet computed.
        batch_prompts.append([prompt_str] * 2)

        # Next prompts should be fully cached, guaranteed to have prefix cache
        # hit.
        batch_prompts.append([prompt_str])

    dtype = "half"
    outputs = {}  # type: ignore
    for enable_prefix_caching in (True, False):
        with vllm_runner(
                model,
                dtype=dtype,
                enable_prefix_caching=enable_prefix_caching,
                enable_chunked_prefill=chunk_size is not None,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                enforce_eager=True,
        ) as vllm_model:
            outputs[enable_prefix_caching] = []
            for prompts in batch_prompts:
                outputs[enable_prefix_caching] += vllm_model.generate_greedy(
                    prompts, max_tokens=max_tokens)

    check_outputs_equal(
        outputs_0_lst=outputs[False],
        outputs_1_lst=outputs[True],
        name_0="vllm_no_apc",
        name_1="vllm",
    )


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER", "XFORMERS"])
def test_unstable_prompt_sequence(
    vllm_runner,
    backend: str,
    monkeypatch,
) -> None:
    override_backend_env_variable(monkeypatch, backend)

    with vllm_runner(
            "Qwen/Qwen2.5-0.5B-Instruct",
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_model_len=4096,
    ) as vllm_model:
        for prompt in UNSTABLE_PROMPT_SEQUENCE:
            vllm_model.generate(TokensPrompt(prompt_token_ids=prompt),
                                SamplingParams(max_tokens=1))
