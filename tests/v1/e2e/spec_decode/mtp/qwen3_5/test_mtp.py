# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams

from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "expected_accuracy_threshold",
        "extra_spec_config",
        "spec_llm_kwargs",
    ],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
            None,
            None,
        ),  # hybrid + MTP, ref: ~34%-35%
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.0,  # GSM8K is covered above; one request at a time is slow
            {"num_speculative_tokens": 3, "ngram_lookup": True},
            # The lookup only runs for single-request batches.
            {"max_num_seqs": 1},
        ),
    ],
    ids=["qwen3_5-hybrid", "qwen3_5-hybrid-ngram-lookup"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    extra_spec_config: dict[str, Any] | None,
    spec_llm_kwargs: dict[str, Any] | None,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
        extra_speculative_config=extra_spec_config,
        spec_llm_kwargs=spec_llm_kwargs,
    )
