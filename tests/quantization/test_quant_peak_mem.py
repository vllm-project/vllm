# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for peak memory usage during online quantization weight loading."""

import logging

import pytest
import regex

from tests.quantization.utils import is_quant_method_supported


@pytest.mark.parametrize(
    "quantization,expected_model_memory_gib",
    [
        pytest.param(
            "fp8",
            6.7,
            marks=pytest.mark.skipif(
                not is_quant_method_supported("fp8"),
                reason="FP8 is not supported on this GPU type.",
            ),
        ),
        pytest.param(
            "experts_int8",
            7.0,
        ),
    ],
)
def test_online_quant_peak_mem(
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch,
    quantization: str,
    expected_model_memory_gib: float,
) -> None:
    # Note: `allenai/OLMoE-1B-7B-0125-Instruct` was selected because:
    # 1. it covers both Linear and MoE paths
    # 2. it is already used by other tests in CI, so adding it here
    #    does not increase disk space for CI runners
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"

    # Force spawn to ensure caplog_mp_spawn works consistently
    # (it relies on VLLM_LOGGING_CONFIG_PATH which spawn reads but fork ignores)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    with (
        caplog_mp_spawn(logging.DEBUG) as log_holder,
        vllm_runner(
            model_name,
            quantization=quantization,
            enforce_eager=True,
        ) as llm,
    ):
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])

    log_text = log_holder.text

    # Parse memory usage from captured logs
    model_memory_gib = None
    peak_memory_gib = None
    for line in log_text.splitlines():
        if model_memory_gib is None:
            match = regex.search(r"Model loading took ([\d.]+) GiB memory", line)
            if match:
                model_memory_gib = float(match.group(1))
        if peak_memory_gib is None:
            match = regex.search(
                r"Peak GPU memory after loading weights: ([\d.]+) GiB", line
            )
            if match:
                peak_memory_gib = float(match.group(1))

    assert model_memory_gib is not None, "Could not find model loading memory log"
    assert peak_memory_gib is not None, "Could not find peak memory log"
    print(f"GPU memory used after loading weights: {model_memory_gib} GiB")
    print(f"Peak GPU memory usage while loading weights: {peak_memory_gib} GiB")

    # Peak should be somewhat higher than final model memory due to
    # streaming quantization (bf16 + quantized weights alive simultaneously)
    expected_peak_memory_gib = expected_model_memory_gib * 1.4

    assert model_memory_gib < expected_model_memory_gib, (
        f"{model_memory_gib=} higher than {expected_model_memory_gib}"
    )
    assert peak_memory_gib < expected_peak_memory_gib, (
        f"{peak_memory_gib=} higher than {expected_peak_memory_gib}"
    )
