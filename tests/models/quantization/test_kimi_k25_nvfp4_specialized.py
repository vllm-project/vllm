# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for the Kimi-K2.5 NVFP4 specialized model path."""

from __future__ import annotations

import json

import openai
import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

if (
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
):
    pytest.skip(
        "Kimi-K2.5 NVFP4 specialization requires Blackwell CUDA.",
        allow_module_level=True,
    )

if current_platform.device_count() < 2:
    pytest.skip(
        "Kimi-K2.5 NVFP4 specialization integration test requires 2 GPUs.",
        allow_module_level=True,
    )

pytest.importorskip("cutlass")
flashinfer_comm = pytest.importorskip("flashinfer.comm")
if not has_flashinfer_trtllm_fused_moe():
    pytest.skip(
        "FlashInfer TRTLLM fused NVFP4 MoE is unavailable.",
        allow_module_level=True,
    )
if not hasattr(flashinfer_comm, "trtllm_allreduce_fusion") or not hasattr(
    flashinfer_comm,
    "trtllm_moe_finalize_allreduce_fusion",
):
    pytest.skip(
        "FlashInfer TRTLLM all-reduce fusion kernels are unavailable.",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.distributed(num_gpus=2), pytest.mark.quant_model]

MODEL = "nvidia/Kimi-K2.5-NVFP4"
PROMPT = "Hello, my name is"

HF_OVERRIDES = {
    "text_config": {
        "num_layers": 2,
        "num_hidden_layers": 2,
    },
    "vision_config": {
        "num_hidden_layers": 1,
    },
}

SERVER_ARGS = [
    "--language-model-only",
    "--load-format",
    "dummy",
    "--dtype",
    "bfloat16",
    "--kv-cache-dtype",
    "fp8",
    "--tensor-parallel-size",
    "2",
    "--attention-backend",
    "FLASHINFER_MLA",
    "--moe-backend",
    "flashinfer_trtllm",
    "--max-model-len",
    "2048",
    "--max-num-batched-tokens",
    "256",
    "--max-num-seqs",
    "4",
    "--trust-remote-code",
    "--limit-mm-per-prompt",
    json.dumps({"image": 0, "video": 0, "vision_chunk": 0}),
]

COMMON_ENV = {
    "TOKENIZERS_PARALLELISM": "true",
    "VLLM_HAS_FLASHINFER_CUBIN": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_USE_FLASHINFER_MOE_FP4": "1",
    "VLLM_FLASHINFER_MOE_BACKEND": "latency",
    "FLASHINFER_NVCC_THREADS": "16",
}


def _completion_result(client: openai.OpenAI) -> dict[str, object]:
    completion = client.completions.create(
        model=MODEL,
        prompt=PROMPT,
        temperature=0.0,
        max_tokens=1,
        logprobs=5,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    top_logprobs = choice.logprobs.top_logprobs[0]
    return {
        "text": choice.text,
        "top_logprobs": {
            token: round(float(logprob), 2)
            for token, logprob in top_logprobs.items()
        },
    }


def _run_with_specialization(enabled: bool) -> dict[str, object]:
    env = {
        **COMMON_ENV,
        "VLLM_USE_SPECIALIZED_MODELS": "1" if enabled else "0",
    }
    with RemoteOpenAIServer(
        MODEL,
        SERVER_ARGS,
        env_dict=env,
        max_wait_seconds=1500,
        override_hf_configs=HF_OVERRIDES,
    ) as server:
        return _completion_result(server.get_client())


def test_kimi_k25_nvfp4_specialized_matches_generic() -> None:
    generic = _run_with_specialization(enabled=False)
    specialized = _run_with_specialization(enabled=True)
    assert specialized == generic
