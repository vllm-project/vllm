# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end accuracy test for GPT-OSS model quantization.

Config:
    Task:   gsm8k_platinum
    Filter: flexible-extract
    n-shot: 5
    Metric: exact_match

Run: pytest tests/models/quantization/test_gpt_oss.py
"""

import importlib
import importlib.metadata
from dataclasses import dataclass

import huggingface_hub
import lm_eval
import pytest
from packaging import version

MODEL_ACCURACIES = {
    # Full quantization: attention linears and MoE linears
    "amd/gpt-oss-20b-WFP8-AFP8-KVFP8": 0.89,
    # MoE linears only quantization
    "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8": 0.89,
    # MoE linears only quantization
    # "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-MXFP4-KV-FP8": 0.90,
}

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.9.0")


def has_huggingface_access(repo):
    try:
        huggingface_hub.list_repo_refs(repo)
        return True
    except huggingface_hub.errors.RepositoryNotFoundError:
        return False


HF_HUB_AMD_ORG_ACCESS = all(
    [has_huggingface_access(model_name) for model_name in MODEL_ACCURACIES]
)


@dataclass
class ModelCase:
    model_id: str
    tp: int


@dataclass
class EvaluationConfig:
    model_name: str

    def get_model_args(self, tp_size: int):
        return {
            "pretrained": self.model_name,
            "chat_template_args": {"reasoning_effort": "low"},
            "enable_thinking": True,
            "think_end_token": "200008",
            "tensor_parallel_size": tp_size,
            "dtype": "auto",
            "gpu_memory_utilization": 0.95,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "enforce_eager": False,
        }


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("model_name, expected_accuracy", MODEL_ACCURACIES.items())
def test_gpt_oss_attention_quantization(
    model_name: str, tp_size: int, expected_accuracy: float
):
    model_args = EvaluationConfig(model_name).get_model_args(tp_size)

    extra_run_kwargs = {
        "gen_kwargs": {"max_gen_toks": 8000},
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    }

    lm_eval_out = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k_platinum",
        batch_size="auto",
        **extra_run_kwargs,
    )
    measured_accuracy = float(
        lm_eval_out["results"]["gsm8k_platinum"]["exact_match,flexible-extract"]
    )

    rtol = 0.02
    assert (
        measured_accuracy - rtol < expected_accuracy
        and measured_accuracy + rtol > expected_accuracy
    ), f"Expected: {expected_accuracy} |  Measured: {measured_accuracy}"
