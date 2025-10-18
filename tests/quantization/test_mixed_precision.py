# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test quark-quantized {MXFP4, FP8} mixed precision models.

Run `pytest tests/quantization/test_mixed_precision.py`.

"""

import importlib
import importlib.metadata
from dataclasses import dataclass

import lm_eval
import pytest
from packaging import version

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")


@dataclass
class ModelCase:
    model_id: str
    tp: int


@dataclass
class EvaluationConfig:
    model_name: str

    def get_model_args(self) -> str:
        return (
            f"pretrained={self.model_name},"
            "tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=False"
        )


TEST_CONFIGS = {
    # Mixed-precision (AMP) model
    # - Demonstrates end-to-end pipeline functionality
    "amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8": {"arc_challenge": 0.52, "mmlu": 0.72},
    # Non-mixed-precision (PTQ) model
    # - Reference for pipeline compatibility verification -> No conflicts or breakings
    "amd/Llama-2-70b-chat-hf-FP8-MLPerf-fp8_attn_quark_format": {
        "arc_challenge": 0.53,
        "mmlu": 0.61,
    },
}


@pytest.mark.parametrize("model_name, accuracy_numbers", TEST_CONFIGS.items())
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
def test_mixed_precision_model_accuracies(model_name: str, accuracy_numbers: dict):
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=EvaluationConfig(model_name).get_model_args(),
        tasks=list(accuracy_numbers.keys()),
        batch_size=8,
    )

    rtol = 0.05

    for task, expect_accuracy in accuracy_numbers.items():
        measured_accuracy = results["results"][task]["acc,none"]
        assert (
            measured_accuracy - rtol < expect_accuracy
            and measured_accuracy + rtol > expect_accuracy
        ), f"Expected: {expect_accuracy} |  Measured: {measured_accuracy}"
