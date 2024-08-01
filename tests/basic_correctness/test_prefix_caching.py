"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests prefix caching. Chunked prefill can be enabled by
enable_prefix_caching=True.

Run `pytest tests/basic_correctness/test_prefix_caching.py`.
"""
import pytest

from tests.kernels.utils import override_backend_env_variable

from ..models.utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("use_v2_block_manager", [False, True])
def test_mixed_requests(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    use_v2_block_manager: bool,
    monkeypatch,
) -> None:
    """
    Test the case when some sequences have the prefix cache hit
    and the others don't.
    """
    override_backend_env_variable(monkeypatch, backend)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    cached_prompt = example_prompts[0]
    with vllm_runner(
            model,
            dtype=dtype,
            enable_prefix_caching=True,
            use_v2_block_manager=use_v2_block_manager,
    ) as vllm_model:
        # Run the first prompt so the cache is populated
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens)

        # Run all the promopts
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
