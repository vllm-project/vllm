# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    assert_request_outputs_match,
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)

PLACEHOLDER_REGRESSION_MODEL = "Qwen/Qwen3.5-0.8B-Base"
PLACEHOLDER_REGRESSION_PROMPT = "The capital of France is"
PLACEHOLDER_REGRESSION_MAX_TOKENS = 16
PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID = 42
PLACEHOLDER_REGRESSION_NUM_SPEC_TOKENS = 3


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE],
    ids=["piecewise", "full-and-piecewise"],
)
@single_gpu_only
def test_mtp_rejected_placeholders_match_greedy_token_ids(
    cudagraph_mode: CUDAGraphMode,
    vllm_runner,
):
    """Rejected MTP padding must not alter the non-speculative token stream."""
    common_kwargs: dict[str, Any] = {
        "block_size": None,
        "max_model_len": 256,
        "max_num_seqs": 1,
        "trust_remote_code": True,
        "enable_chunked_prefill": None,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "attention_backend": ("TRITON_ATTN" if current_platform.is_rocm() else "auto"),
    }
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=PLACEHOLDER_REGRESSION_MAX_TOKENS,
        ignore_eos=True,
        allowed_token_ids=[PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID],
        logprobs=1,
    )

    def generate_token_ids(runner) -> tuple[int, ...]:
        outputs = runner.llm.generate([PLACEHOLDER_REGRESSION_PROMPT], sampling_params)
        assert outputs and outputs[0].outputs, "Expected one completion"
        token_ids = tuple(outputs[0].outputs[0].token_ids)
        assert len(token_ids) == PLACEHOLDER_REGRESSION_MAX_TOKENS
        return token_ids

    def compilation_config() -> CompilationConfig:
        return CompilationConfig(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=[1, PLACEHOLDER_REGRESSION_NUM_SPEC_TOKENS + 1],
        )

    with vllm_runner(
        PLACEHOLDER_REGRESSION_MODEL,
        compilation_config=compilation_config(),
        **common_kwargs,
    ) as ref_runner:
        ref_mode = (
            ref_runner.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
        )
        assert ref_mode == cudagraph_mode
        ref_token_ids = generate_token_ids(ref_runner)
        expected_token_ids = (PLACEHOLDER_REGRESSION_ALLOWED_TOKEN_ID,) * (
            PLACEHOLDER_REGRESSION_MAX_TOKENS
        )
        assert ref_token_ids == expected_token_ids

    with vllm_runner(
        PLACEHOLDER_REGRESSION_MODEL,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": PLACEHOLDER_REGRESSION_NUM_SPEC_TOKENS,
            "max_model_len": 256,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [0.0]
            * PLACEHOLDER_REGRESSION_NUM_SPEC_TOKENS,
        },
        disable_log_stats=False,
        compilation_config=compilation_config(),
        **common_kwargs,
    ) as spec_runner:
        spec_mode = (
            spec_runner.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
        )
        assert spec_mode == cudagraph_mode
        spec_token_ids = generate_token_ids(spec_runner)
        metrics = spec_runner.llm.get_metrics()
        num_drafts = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_drafts"
        )
        num_draft_tokens = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_draft_tokens"
        )
        num_accepted_tokens = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_accepted_tokens"
        )

    assert num_drafts > 0, "MTP drafter did not run"
    assert num_draft_tokens > 0, "MTP drafter did not produce draft tokens"
    assert num_accepted_tokens == 0, "Synthetic sampler accepted a draft token"
    assert spec_token_ids == ref_token_ids, (
        f"MTP changed greedy token ids under {cudagraph_mode.name}:\n"
        f"  reference={ref_token_ids}\n"
        f"  speculative={spec_token_ids}"
    )


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        # Reference accuracy: 65%-70%.
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1, None), False, 0.5),
        pytest.param(
            ("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1, None),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
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
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    """
    Compare the outputs of a original LLM and a speculative LLM
    which should be the same when using MTP speculative decoding. Due to some variance
    in the engine, it is possible for some outputs to differ, so we expect that at least
    6/10 output tokens match exactly, and that the GSM8k accuracy is above a precomputed
    reference threshold for each model.
    """
    method, model_name, tp_size, draft_model = model_setup
    _skip_if_insufficient_gpus_for_tp(tp_size)

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

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

        with vllm_runner(
            model_name,
            block_size=None,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            attention_backend=attn_backend,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        ) as ref_runner:
            ref_outputs = ref_runner.llm.chat(test_prompts, sampling_config)
            evaluate_llm_for_gsm8k(
                ref_runner.llm,
                expected_accuracy_threshold=expected_accuracy_threshold,
            )

        speculative_config: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": 1,
            "max_model_len": 2048,
        }
        if draft_model is not None:
            speculative_config["model"] = draft_model
            speculative_config["num_speculative_tokens"] = 2

        with vllm_runner(
            model_name,
            block_size=None,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config=speculative_config,
            max_model_len=2048,
            attention_backend=attn_backend,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        ) as spec_runner:
            # MTP supports async scheduling by default.
            has_async = (
                spec_runner.llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            )
            assert has_async, (
                f"Expected async scheduling for {method}: target={model_name}, "
                f"draft={draft_model}; got {has_async}"
            )
            evaluate_llm_for_gsm8k(
                spec_runner.llm,
                expected_accuracy_threshold=expected_accuracy_threshold,
            )
            spec_outputs = spec_runner.llm.chat(test_prompts, sampling_config)

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert_request_outputs_match(
            ref_outputs,
            spec_outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context=f"{method} target={model_name}, draft={draft_model}",
        )
