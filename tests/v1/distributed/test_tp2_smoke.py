# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Smoke test for TP=2 on Hopper/Blackwell with default config.

Starts a vLLM server with TP=2 and default settings, verifies it can
serve requests across dense, FP8, and MoE models.
"""

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
    "microsoft/Phi-mini-MoE-instruct",
]


@pytest.mark.parametrize("model_id", MODELS)
def test_tp2_smoke(model_id: str, num_gpus_available: int):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    if num_gpus_available < 2:
        pytest.skip("Need at least 2 GPUs")

    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
    ]

    with RemoteOpenAIServer(model_id, args) as server:
        client = server.get_client()
        completion = client.completions.create(
            model=model_id,
            prompt="Hello",
            max_tokens=16,
            temperature=0,
        )
        assert len(completion.choices[0].text) > 0
