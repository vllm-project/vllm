# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test quark-quantized {MXFP4, FP8} mixed precision models.

Run `pytest tests/quantization/test_mixed_precision.py`.

"""

import importlib
import importlib.metadata
import os
from dataclasses import dataclass

import huggingface_hub
import lm_eval
import pytest
from packaging import version


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module relies on V0 internals, so set VLLM_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


QUARK_MXFP4_AVAILABLE = importlib.util.find_spec(
    "quark") is not None and version.parse(
        importlib.metadata.version("amd-quark")) >= version.parse('0.8.99')

try:
    huggingface_hub.list_repo_refs(
        "amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8")
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


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
    "amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8": {
        "arc_challenge": 0.53,
        "mmlu": 0.60
    },
    "amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8": {
        "arc_challenge": 0.60,
        "mmlu": 0.66
    },
    "amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8": {
        "arc_challenge": 0.53,
        "mmlu": 0.70
    },
}


@pytest.mark.parametrize("model_name, accuracy_numbers", TEST_CONFIGS.items())
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.")
def test_mixed_precision_model_accuracies(model_name: str,
                                          accuracy_numbers: dict, monkeypatch):
    monkeypatch.setenv("VLLM_QUARK_EMU_MEM_OPT", "1")

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=EvaluationConfig(model_name).get_model_args(),
        tasks=list(accuracy_numbers.keys()),
        batch_size="auto" if "Qwen3-8B" not in model_name else 8)

    rtol = 0.05

    for task, expect_accuracy in accuracy_numbers.items():
        measured_accuracy = results["results"][task]["acc,none"]
        assert (
            measured_accuracy - rtol < expect_accuracy
            and measured_accuracy + rtol > expect_accuracy
        ), f"Expected: {expect_accuracy} |  Measured: {measured_accuracy}"

    del os.environ["VLLM_QUARK_EMU_MEM_OPT"]
