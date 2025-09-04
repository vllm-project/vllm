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

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec(
    "quark") is not None and version.parse(
        importlib.metadata.version("amd-quark")) >= version.parse('0.8.99')

try:
    huggingface_hub.list_repo_refs(
        "amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8"
    )
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module relies on V0 internals, so set VLLM_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


@dataclass
class ModelCase:
    model_id: str
    tp: int


@dataclass
class EvaluationConfig:
    model_name: str
    excepted_value: float

    def get_model_args(self) -> str:
        return (
            f"pretrained={self.model_name},"
            "tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=False"
        )


ACCURACY_CONFIGS = [
    # Private model.
    EvaluationConfig(
        model_name=
        "amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8",
        excepted_value=0.53),
    EvaluationConfig(
        model_name=
        "amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8",
        excepted_value=0.60),
]
TASKS = ["arc_challenge"]


@pytest.mark.parametrize("config", ACCURACY_CONFIGS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.")
def test_mixed_precision_model_accuracies(config: EvaluationConfig, task: str):
    os.environ["VLLM_QUARK_EMU_MEM_OPT"] = "1"

    results = lm_eval.simple_evaluate(model="vllm",
                                      model_args=config.get_model_args(),
                                      tasks=task,
                                      batch_size="auto")

    rtol = 0.05

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["acc,none"]
    assert (measured_value - rtol < EXPECTED_VALUE
            and measured_value + rtol > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"

    del os.environ["VLLM_QUARK_EMU_MEM_OPT"]
