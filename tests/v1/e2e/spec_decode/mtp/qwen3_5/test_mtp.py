# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
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


@single_gpu_only
@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(121)),
    reason="draft_confidence_threshold is SM121 only",
)
def test_mtp_draft_confidence_stop(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The stop only shortens drafting: greedy outputs match fixed-depth MTP,
    a deeper capped chain keeps at least 90% of the fixed chain's AL, and the
    draft counters count only verified drafts. One request at a time, so the
    stop rather than the fallback is used."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False, num_prompts=32)
    results = {}
    drafts_per_round = {}
    for num_spec, threshold in ((3, None), (4, 0.6)):
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=2048,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": num_spec,
                "draft_confidence_threshold": threshold,
                "draft_confidence_fallback_depth": 3 if threshold else None,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.chat(prompts, sampling_config)
            metrics = runner.llm.get_metrics()
            num_drafts, num_draft_tokens = (
                get_spec_decode_metric_value(metrics, f"vllm:spec_decode_{name}")
                for name in ("num_drafts", "num_draft_tokens")
            )
            results[threshold] = (outputs, compute_acceptance_len(metrics))
            drafts_per_round[threshold] = num_draft_tokens / num_drafts

    (ref_outputs, ref_al), (outputs, al) = results[None], results[0.6]
    assert_request_outputs_match(
        ref_outputs,
        outputs,
        required_matches=int(0.8 * len(ref_outputs)) + 1,
        context="draft_confidence_threshold=0.6",
    )
    assert al >= 0.9 * ref_al, f"acceptance length {al:.3f} vs {ref_al:.3f}"
    assert drafts_per_round[None] == 3
    assert 1 <= drafts_per_round[0.6] < 4, drafts_per_round
