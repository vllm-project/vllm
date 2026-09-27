# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform

from ...utils import get_spec_decode_metric_value

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
