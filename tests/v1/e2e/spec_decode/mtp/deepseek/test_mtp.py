# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import assert_request_outputs_match, get_test_prompts
from .._correctness import check_mtp_correctness

DEEPSEEK_MTP_MODEL = "vllm-project/DeepSeek-V3-4layers-MTP-FP8"  # dummy model
skip_on_blackwell = pytest.mark.skipif(
    current_platform.is_device_capability_family(100),
    reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
)


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        pytest.param(
            ("mtp", DEEPSEEK_MTP_MODEL, 1, None),
            False,
            0.0,
            marks=skip_on_blackwell,
        ),
    ],
    ids=["deepseek"],
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
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@skip_on_blackwell
@single_gpu_only
def test_mtp_local_argmax_reduction(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The DeepSeek MTP drafter starts with local-argmax drafting and keeps the
    greedy outputs of full-vocab drafting."""
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    prompts = get_test_prompts(mm_enabled=False)
    outputs = {}
    for use_local_argmax_reduction in (False, True):
        with vllm_runner(
            DEEPSEEK_MTP_MODEL,
            block_size=None,
            max_model_len=2048,
            trust_remote_code=True,
            attention_backend="TRITON_ATTN" if current_platform.is_rocm() else "auto",
            enable_chunked_prefill=None,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
                "use_local_argmax_reduction": use_local_argmax_reduction,
            },
        ) as runner:
            outputs[use_local_argmax_reduction] = runner.llm.chat(
                prompts, sampling_config
            )

    assert_request_outputs_match(
        outputs[False],
        outputs[True],
        required_matches=int(0.8 * len(outputs[False])) + 1,
        context="use_local_argmax_reduction=True",
    )
