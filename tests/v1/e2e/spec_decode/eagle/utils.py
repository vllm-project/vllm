# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import wait_for_rocm_memory_to_settle
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


def _run_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
):
    """
    Compare the outputs of an original LLM and a speculative LLM
    which should be the same when using eagle speculative decoding.
    """
    if model_impl == "transformers":
        import transformers
        from packaging.version import Version

        installed = Version(transformers.__version__)
        required = Version("5.0.0")
        if installed < required:
            pytest.skip(
                "Eagle3 with the Transformers modeling backend requires "
                f"transformers>={required}, but got {installed}"
            )

    test_prompts = get_test_prompts(mm_enabled)

    if "Llama-4-Scout" in model_setup[1] and attn_backend == "FLASH_ATTN":
        if current_platform.is_rocm():
            print(
                "FLASH_ATTN for spec_decode not supported on "
                "ROCm currently. Changing to FLEX_ATTENTION backend."
            )
            attention_config = {"backend": "FLEX_ATTENTION"}
        else:
            attention_config = None
    else:
        attention_config = {"backend": attn_backend}

    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
            if "deepseek" in model_setup[1].lower():
                m.setenv("VLLM_ROCM_USE_AITER", "1")
                m.delenv("VLLM_MLA_DISABLE", raising=False)
                attention_config = {"backend": "ROCM_AITER_MLA"}
            else:
                m.setenv("VLLM_ROCM_USE_AITER", "1")

        method, model_name, spec_model_name, tp_size = model_setup
        _skip_if_insufficient_gpus_for_tp(tp_size)

        max_model_len = 2048
        max_num_batched_tokens = 128 if enable_chunked_prefill else max_model_len

        ref_llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
            attention_config=attention_config,
        )
        evaluate_llm_for_gsm8k(
            ref_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
        # ROCm frees VRAM lazily; wait so the spec engine started right after
        # does not OOM on its startup memory guard.
        wait_for_rocm_memory_to_settle()

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 3,
                "max_model_len": max_model_len,
            },
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            model_impl=model_impl,
            attention_config=attention_config,
        )
        # EAGLE/EAGLE3 supports async scheduling; assert it is active by default.
        assert spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        evaluate_llm_for_gsm8k(
            spec_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        assert matches > int(0.6 * len(ref_outputs))
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
        # ROCm frees VRAM lazily; wait so the next parametrization's engine does
        # not OOM on its startup memory guard.
        wait_for_rocm_memory_to_settle()
