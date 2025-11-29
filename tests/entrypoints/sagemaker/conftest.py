# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared fixtures and utilities for SageMaker tests."""

import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# Model name constants used across tests
MODEL_NAME_SMOLLM = "HuggingFaceTB/SmolLM2-135M-Instruct"
LORA_ADAPTER_NAME_SMOLLM = "jekunz/smollm-135m-lora-fineweb-faroese"

# SageMaker header constants
HEADER_SAGEMAKER_CLOSED_SESSION_ID = "X-Amzn-SageMaker-Closed-Session-Id"
HEADER_SAGEMAKER_SESSION_ID = "X-Amzn-SageMaker-Session-Id"
HEADER_SAGEMAKER_NEW_SESSION_ID = "X-Amzn-SageMaker-New-Session-Id"


@pytest.fixture(scope="session")
def smollm2_lora_files():
    """Download LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=LORA_ADAPTER_NAME_SMOLLM)


@pytest.fixture(scope="module")
def basic_server_with_lora(smollm2_lora_files):
    """Basic server fixture with standard configuration."""
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--max-lora-rank",
        "256",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "64",
    ]

    envs = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    with RemoteOpenAIServer(MODEL_NAME_SMOLLM, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def async_client(basic_server_with_lora: RemoteOpenAIServer):
    """Async OpenAI client fixture for use with basic_server."""
    async with basic_server_with_lora.get_async_client() as async_client:
        yield async_client
