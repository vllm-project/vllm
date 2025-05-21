# SPDX-License-Identifier: Apache-2.0
import os
import shutil

import pytest
from huggingface_hub import snapshot_download

from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_NAME = "typeof/zephyr-7b-beta-lora"
PA_NAME = "swapnilbp/llama_tweet_ptune"


@pytest.fixture(scope='module')
def adapter_cache(request, tmpdir_factory):
    # Create dir that mimics the structure of the adapter cache
    adapter_cache = tmpdir_factory.mktemp(
        request.module.__name__) / "adapter_cache"
    return adapter_cache


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def pa_files():
    return snapshot_download(repo_id=PA_NAME)


@pytest.mark.asyncio
async def test_filesystem_resolver(adapter_cache, zephyr_lora_files):
    model_files = adapter_cache / LORA_NAME
    shutil.copytree(zephyr_lora_files, model_files)

    fs_resolver = FilesystemResolver(adapter_cache)
    assert fs_resolver is not None

    lora_request = await fs_resolver.resolve_lora(MODEL_NAME, LORA_NAME)
    assert lora_request is not None
    assert lora_request.lora_name == LORA_NAME
    assert lora_request.lora_path == os.path.join(adapter_cache, LORA_NAME)


@pytest.mark.asyncio
async def test_missing_adapter(adapter_cache):
    fs_resolver = FilesystemResolver(adapter_cache)
    assert fs_resolver is not None

    missing_lora_request = await fs_resolver.resolve_lora(MODEL_NAME, "foobar")
    assert missing_lora_request is None


@pytest.mark.asyncio
async def test_nonlora_adapter(adapter_cache, pa_files):
    model_files = adapter_cache / PA_NAME
    shutil.copytree(pa_files, model_files)

    fs_resolver = FilesystemResolver(adapter_cache)
    assert fs_resolver is not None

    pa_request = await fs_resolver.resolve_lora(MODEL_NAME, PA_NAME)
    assert pa_request is None
