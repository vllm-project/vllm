# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test attention quantization of gpt-oss model.
The qkv_proj and o_proj in self_attention can be either quantized or excluded.

Run `pytest tests/models/quantization/test_gpt_oss.py`.

"""

import importlib
import importlib.metadata
from dataclasses import dataclass

import huggingface_hub
import lm_eval
import pytest
from packaging import version

MODEL_NAMES = [
    "amd/gpt-oss-20b-MoE-WMXFP4-AMXFP4-Attention-WFP8-AFP8-KV-FP8",
]

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")


def has_huggingface_access(repo):
    try:
        huggingface_hub.list_repo_refs(repo)
        return True
    except huggingface_hub.errors.RepositoryNotFoundError:
        return False


HF_HUB_AMD_ORG_ACCESS = all(
    [has_huggingface_access(model_name) for model_name in MODEL_NAMES]
)


@dataclass
class ModelCase:
    model_id: str
    tp: int


@dataclass
class EvaluationConfig:
    model_name: str

    def get_model_args(self):
        return {
            "pretrained": self.model_name,
            "chat_template_args": {"reasoning_effort": "low"},
            "enable_thinking": True,
            "think_end_token": "200008",
            "tensor_parallel_size": 1,
            "dtype": "auto",
            "gpu_memory_utilization": 0.95,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "enforce_eager": False,
        }


EXPECTED_ACCURACIES = {"gsm8k_platinum": 0.89}


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("task_name, expected_accuracy", EXPECTED_ACCURACIES.items())
def test_gpt_oss_attention_quantization(
    model_name: str, task_name: str, expected_accuracy: float
):
    model_args = EvaluationConfig(model_name).get_model_args()

    extra_run_kwargs = {
        "gen_kwargs": {"max_gen_toks": 8000},
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    }

    measured_accuracy = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_name,
        batch_size="auto",
        **extra_run_kwargs,
    )["results"][task_name]["acc,none"]

    rtol = 0.05
    assert (
        measured_accuracy - rtol < expected_accuracy
        and measured_accuracy + rtol > expected_accuracy
    ), f"Expected: {expected_accuracy} |  Measured: {measured_accuracy}"
