# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False, 0.5),  # ref: 65%-70%
        pytest.param(
            ("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["mimo", "deepseek", "qwen3_5-hybrid", "gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int] | tuple[str, str, int, str],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
):
    """
    Compare the outputs of a original LLM and a speculative LLM
    which should be the same when using MTP speculative decoding. Due to some variance
    in the engine, it is possible for some outputs to differ, so we expect that at least
    6/10 output tokens match exactly, and that the GSM8k accuracy is above a precomputed
    reference threshold for each model.
    """
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if len(model_setup) == 4:
            method, model_name, tp_size, draft_model = model_setup
        else:
            method, model_name, tp_size = model_setup
            draft_model = None
        _skip_if_insufficient_gpus_for_tp(tp_size)

        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"

        # Skip multimodal profiling for models that don't need it in this test.
        extra_kwargs: dict[str, Any] = {}
        if "Qwen3.5" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        elif "gemma-4" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

        if draft_model is not None and "gemma-4" in draft_model:
            import transformers
            from packaging.version import Version

            if Version(transformers.__version__) < Version("5.8.0"):
                pytest.skip(
                    "Gemma4 MTP assistant requires transformers>=5.8.0, "
                    f"got {transformers.__version__}"
                )

        ref_llm = LLM(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        evaluate_llm_for_gsm8k(
            ref_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        del ref_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

        speculative_config: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": 1,
            "max_model_len": 2048,
        }
        if draft_model is not None:
            speculative_config["model"] = draft_model
            speculative_config["num_speculative_tokens"] = 2

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config=speculative_config,
            max_model_len=2048,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        # MTP supports async scheduling; assert it is active by default.
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

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.8 * len(ref_outputs))
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@single_gpu_only
def test_qwen3_5_mtp_prefix_cache_reuses_last_safe_block(
    monkeypatch: pytest.MonkeyPatch,
):
    """Offloading uses the successor-proven MTP boundary without accuracy loss."""
    aligned_page_size = 544
    prompt_len = 2 * aligned_page_size + 1
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        logprobs=5,
    )

    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")
        llm = LLM(
            model="Qwen/Qwen3.5-0.8B-Base",
            trust_remote_code=True,
            enable_prefix_caching=True,
            max_model_len=2048,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
                "max_model_len": 2048,
            },
            kv_transfer_config={
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": 512 << 20,
                },
            },
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        assert llm.llm_engine.vllm_config.cache_config.prefix_match_unit is None

        tokenizer = llm.get_tokenizer()
        source = (
            "The following context is intentionally repeated to exercise prefix "
            "caching. " * 300
        ) + "Explain why deterministic inference should be reproducible."
        prompt_token_ids = tokenizer.encode(source)[-prompt_len:]
        assert len(prompt_token_ids) == prompt_len
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        cold_output = llm.generate([prompt], sampling_params)[0]
        warm_output = llm.generate([prompt], sampling_params)[0]

        assert cold_output.num_cached_tokens == 0
        max_safe_hit = 2 * aligned_page_size
        # An unconditional one-page EAGLE drop would report only 544 tokens.
        assert warm_output.num_cached_tokens == max_safe_hit

        cold_completion = cold_output.outputs[0]
        warm_completion = warm_output.outputs[0]
        assert cold_completion.token_ids == warm_completion.token_ids
        check_logprobs_close(
            outputs_0_lst=[
                (
                    list(cold_completion.token_ids),
                    cold_completion.text,
                    cold_completion.logprobs,
                )
            ],
            outputs_1_lst=[
                (
                    list(warm_completion.token_ids),
                    warm_completion.text,
                    warm_completion.logprobs,
                )
            ],
            name_0="cold_mtp",
            name_1="warm_mtp",
            always_check_logprobs=True,
        )

        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
