# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ..utils import run_acceptance_length_eval


@pytest.mark.parametrize("use_mrv2", [False, True])
def test_gemma4_mtp_acceptance_lengths(
    monkeypatch: pytest.MonkeyPatch,
    use_mrv2: bool,
):
    run_acceptance_length_eval(
        monkeypatch,
        spec_config={
            "model": "google/gemma-4-E4B-it",
            "trust_remote_code": True,
            "speculative_config": {
                "method": "mtp",
                "model": "google/gemma-4-E4B-it-assistant",
                "num_speculative_tokens": 2,
                "max_model_len": 32768,
            },
            "max_model_len": 32768,
            "limit_mm_per_prompt": {"image": 0, "audio": 0},
            "disable_log_stats": False,
        },
        expected_acceptance_lengths={
            "mt-bench": 2.28,
            "humaneval": 2.68,
            "gsm8k": 2.67 * 0.975,
        },
        chat_template_kwargs={},
        use_mrv2=use_mrv2,
    )
