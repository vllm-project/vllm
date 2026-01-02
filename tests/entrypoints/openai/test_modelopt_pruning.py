# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer

MODEL_ID = "CedricHwang/qwen2.5-0.5b-modelopt-pruning-gradnas"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for modelopt pruning server test.",
)
def test_modelopt_pruning_openai_server():
    hf_token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        token=hf_token,
    )

    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enable-modelopt-pruning",
    ]
    with RemoteOpenAIServer(MODEL_ID, server_args) as server:
        client = server.get_client()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=32,
            temperature=0.0,
        )

        assert response.choices, "No choices returned from vLLM server."
        content = response.choices[0].message.content or ""
        assert content.strip(), "Empty completion returned from vLLM server."
