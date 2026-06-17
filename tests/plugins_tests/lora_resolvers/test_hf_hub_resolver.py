# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from vllm.plugins.lora_resolvers.hf_hub_resolver import HfHubResolver

LORA_LIB_MODEL_NAME = "ibm-granite/granite-3.3-8b-instruct"
# Repo with multiple LoRAs contained in it
LORA_LIB = "ibm-granite/granite-3.3-8b-rag-agent-lib"
LORA_NAME = "ibm-granite/granite-3.3-8b-rag-agent-lib/answerability_prediction_lora"  # noqa: E501
NON_LORA_SUBPATH = "ibm-granite/granite-3.3-8b-rag-agent-lib/README.md"
LIB_DOWNLOAD_DIR = os.path.join(
    HF_HUB_CACHE, "models--ibm-granite--granite-3.3-8b-rag-agent-lib"
)
INVALID_REPO_NAME = "thisrepodoesnotexist"

# Repo with only one LoRA in the root dir
LORA_REPO_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LORA_REPO = "yard1/llama-2-7b-sql-lora-test"
REPO_DOWNLOAD_DIR = os.path.join(
    HF_HUB_CACHE, "models--yard1--llama-2-7b-sql-lora-test"
)


@pytest.mark.asyncio
async def test_hf_resolver_with_direct_path():
    hf_resolver = HfHubResolver([LORA_REPO])
    assert hf_resolver is not None

    lora_request = await hf_resolver.resolve_lora(LORA_REPO_MODEL_NAME, LORA_REPO)
    assert lora_request.lora_name == LORA_REPO
    assert REPO_DOWNLOAD_DIR in lora_request.lora_path
    assert "adapter_config.json" in os.listdir(lora_request.lora_path)


@pytest.mark.asyncio
async def test_hf_resolver_with_nested_paths():
    hf_resolver = HfHubResolver([LORA_LIB])
    assert hf_resolver is not None

    lora_request = await hf_resolver.resolve_lora(LORA_LIB_MODEL_NAME, LORA_NAME)
    assert lora_request is not None
    assert lora_request.lora_name == LORA_NAME
    assert LIB_DOWNLOAD_DIR in lora_request.lora_path
    assert "adapter_config.json" in os.listdir(lora_request.lora_path)


@pytest.mark.asyncio
async def test_hf_resolver_with_multiple_repos():
    hf_resolver = HfHubResolver([LORA_LIB, LORA_REPO])
    assert hf_resolver is not None

    lora_request = await hf_resolver.resolve_lora(LORA_LIB_MODEL_NAME, LORA_NAME)
    assert lora_request is not None
    assert lora_request.lora_name == LORA_NAME
    assert LIB_DOWNLOAD_DIR in lora_request.lora_path
    assert "adapter_config.json" in os.listdir(lora_request.lora_path)


@pytest.mark.asyncio
async def test_missing_adapter():
    hf_resolver = HfHubResolver([LORA_LIB])
    assert hf_resolver is not None

    missing_lora_request = await hf_resolver.resolve_lora(LORA_LIB_MODEL_NAME, "foobar")
    assert missing_lora_request is None


@pytest.mark.asyncio
async def test_nonlora_adapter():
    hf_resolver = HfHubResolver([LORA_LIB])
    assert hf_resolver is not None

    readme_request = await hf_resolver.resolve_lora(
        LORA_LIB_MODEL_NAME, NON_LORA_SUBPATH
    )
    assert readme_request is None


@pytest.mark.asyncio
async def test_invalid_repo():
    hf_resolver = HfHubResolver([LORA_LIB])
    assert hf_resolver is not None

    invalid_repo_req = await hf_resolver.resolve_lora(
        INVALID_REPO_NAME,
        f"{INVALID_REPO_NAME}/foo",
    )
    assert invalid_repo_req is None


@pytest.mark.asyncio
async def test_trailing_slash():
    hf_resolver = HfHubResolver([LORA_LIB])
    assert hf_resolver is not None

    lora_request = await hf_resolver.resolve_lora(
        LORA_LIB_MODEL_NAME,
        f"{LORA_NAME}/",
    )
    assert lora_request is not None
    assert lora_request.lora_name == f"{LORA_NAME}/"
    assert LIB_DOWNLOAD_DIR in lora_request.lora_path
    assert "adapter_config.json" in os.listdir(lora_request.lora_path)
