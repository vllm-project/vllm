# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from vllm.plugins.lora_resolvers.hf_hub_resolver import HfHubResolver

MODEL_NAME = "ibm-granite/granite-3.3-8b-instruct"
LORA_LIB = "ibm-granite/granite-3.3-8b-rag-agent-lib"
LORA_NAME = "answerability_prediction_lora"
LIB_DOWNLOAD_DIR = os.path.join(
    HF_HUB_CACHE, "models--ibm-granite--granite-3.3-8b-rag-agent-lib")


@pytest.mark.asyncio
async def test_hf_resolver():
    hf_resolver = HfHubResolver(LORA_LIB)
    assert hf_resolver is not None

    lora_request = await hf_resolver.resolve_lora(MODEL_NAME, LORA_NAME)
    assert lora_request is not None
    assert lora_request.lora_name == LORA_NAME
    assert LIB_DOWNLOAD_DIR in lora_request.lora_path
    assert "adapter_config.json" in os.listdir(lora_request.lora_path)


@pytest.mark.asyncio
async def test_missing_adapter():
    hf_resolver = HfHubResolver(LORA_LIB)
    assert hf_resolver is not None

    missing_lora_request = await hf_resolver.resolve_lora(MODEL_NAME, "foobar")
    assert missing_lora_request is None


@pytest.mark.asyncio
async def test_nonlora_adapter():
    hf_resolver = HfHubResolver(LORA_LIB)
    assert hf_resolver is not None

    readme_request = await hf_resolver.resolve_lora(MODEL_NAME, "README.md")
    assert readme_request is None
