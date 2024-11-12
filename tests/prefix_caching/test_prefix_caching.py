"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

from tests.kernels.utils import override_backend_env_variable
from vllm import SamplingParams, TokensPrompt

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
