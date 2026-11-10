# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import random

import pytest
import torch
from utils import (
    _extract_step_logprobs,
    _random_prompt,
    skip_if_not_cuda_alike,
)

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.model_executor.kernels.linear import HummingLinearKernel
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

# FLASH_ATTN is CUDA-only.
E2E_BACKENDS = ["TRITON_ATTN"] if current_platform.is_rocm() else ["FLASH_ATTN"]

GPTQ_TEST_MODEL = os.getenv(
    "VLLM_TEST_GPTQ_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
)


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=GPTQ_TEST_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_GPTQ_TEST_GPU_MEMORY_UTILIZATION", "0.15")
        ),
        max_model_len=int(os.getenv("VLLM_GPTQ_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@skip_if_not_cuda_alike
@pytest.mark.parametrize("backend", E2E_BACKENDS)
def test_dense_gptq_generation_is_deterministic_across_batch_sizes_e2e(
    monkeypatch, backend
):
    """GPTQ-Int4 generation is batch-invariant end to end.

    On SM90 the mixed-precision oracle currently ranks Machete first for
    4-bit weights, and Machete is not batch-invariant, so batch-invariant
    mode keeps the dequantize fallback. Machete is disabled here so the
    test exercises the Humming batch-invariant path; elsewhere selection
    already avoids Machete and disabling it is a no-op. The env var can be
    dropped once Humming outranks Machete (#63265).
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "MacheteLinearKernel")
    monkeypatch.setattr(envs, "VLLM_DISABLED_KERNELS", ["MacheteLinearKernel"])

    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_GPTQ_NEEDLE_TRIALS", "2"))
    max_batch_size = int(os.getenv("VLLM_GPTQ_NEEDLE_BATCH_SIZE", "8"))
    min_random_prompt = int(os.getenv("VLLM_GPTQ_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_GPTQ_MAX_PROMPT", "96"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to test invariance."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_GPTQ_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_GPTQ_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_GPTQ_NEEDLE_MAX_TOKENS", "16")),
        seed=20240919,
        logprobs=5,
    )
    # The model continues this prompt for the full 16 tokens; a prompt it
    # answers with an immediate EOS would reduce the check to a single
    # decode step.
    needle_prompt = "On the surface of Mars, we found"

    llm = None
    baseline_completion = None
    baseline_logprobs = None
    try:
        llm = _make_llm(max_num_seqs=max_batch_size, backend=backend)
        if current_platform.has_device_capability(90) and has_humming():
            # The Humming path is the point of this test; on SM90 with
            # humming installed it must not silently degrade to the fallback.
            def check_kernel(model):
                method = model.model.layers[0].self_attn.qkv_proj.quant_method
                assert isinstance(method.kernel, HummingLinearKernel), (
                    "Expected HummingLinearKernel under VLLM_BATCH_INVARIANT "
                    f"on SM90, got {method.kernel!r}"
                )

            llm.apply_model(check_kernel)
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None
        assert baseline_token_ids is not None
        for _ in range(num_trials):
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            prompts: list[str] = []
            for idx in range(batch_size):
                if idx == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(_random_prompt(min_random_prompt, max_random_prompt))

            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            needle_output = outputs[needle_pos]
            needle_completion = needle_output.outputs[0]
            needle_logprobs, needle_token_ids = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None
            assert needle_token_ids is not None

            assert needle_output.prompt == needle_prompt
            assert baseline_completion is not None
            assert baseline_logprobs is not None
            assert needle_completion.token_ids == baseline_completion.token_ids
            assert needle_completion.text == baseline_completion.text
            # The batch-invariance contract is exact: identical logprobs, no
            # tolerance.
            torch.testing.assert_close(
                needle_logprobs, baseline_logprobs, rtol=0, atol=0
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
