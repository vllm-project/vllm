# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.v1.metrics.reader import Vector

from ...utils import get_spec_decode_metric_value, get_test_prompts


@pytest.mark.parametrize("use_mrv2", [False, True], ids=["mrv1", "mrv2"])
@single_gpu_only
def test_glm_mtp_drafts_on_both_model_runners(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    use_mrv2: bool,
    vllm_runner,
):
    """GLM MTP returns (logits_hidden, feedback_hidden); both model runners must
    unpack it and feed the feedback hidden into the next draft step."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_mrv2 else "0")
    with vllm_runner(
        "zai-org/GLM-OCR",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0},
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        disable_log_stats=False,
    ) as runner:
        runner.llm.chat(get_test_prompts(mm_enabled=False), sampling_config)
        metrics = runner.llm.get_metrics()

    num_drafts = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
    accepted_per_pos = next(
        m.values
        for m in metrics
        if isinstance(m, Vector)
        and m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    # Measured on MI355X: ~[0.60, 0.28] on both runners.
    step_1, step_2 = (accepted / num_drafts for accepted in accepted_per_pos)
    assert step_1 > 0.4, f"step 1 acceptance {step_1:.3f}"
    assert step_2 > 0.15, f"step 2 acceptance {step_2:.3f}"
