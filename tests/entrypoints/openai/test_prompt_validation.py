# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import hypothesis
import hypothesis_torch
# imports for guided decoding tests
import openai
import pybase64
import pytest
import regex as re
import torch
from hypothesis import strategies as st

from vllm.entrypoints.openai.serving_engine import OpenAIServing

from ...utils import RemoteOpenAIServer


@pytest.fixture(scope="function", autouse=True)
def use_v1_only(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V1', '1')


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match="decoder prompt cannot be empty"):
            await client.completions.create(model=model_name,
                                            prompt="",
                                            max_tokens=5,
                                            temperature=0.0)


@pytest.mark.asyncio
async def test_out_of_vocab_token_ids():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match=re.compile('.*out of vocabulary.*').pattern):
            await client.completions.create(model=model_name,
                                            prompt=[999999],
                                            max_tokens=5,
                                            temperature=0.0)


@hypothesis.given(tensor=hypothesis_torch.tensor_strategy(
    dtype=hypothesis_torch.dtype_strategy(
        [torch.float32, torch.bfloat16, torch.float16]),
    shape=st.tuples(st.integers(min_value=2, max_value=10),
                    st.integers(min_value=2, max_value=10)),
    device=hypothesis_torch.device_strategy(),
    layout=hypothesis_torch.layout_strategy()))
def test_load_prompt_embeds(tensor: torch.Tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    encoded_tensor = pybase64.b64encode(buffer.getvalue())
    assert tensor.layout == torch.sparse_coo

    loaded_prompt_embeds = OpenAIServing._load_prompt_embeds(encoded_tensor)
    assert len(loaded_prompt_embeds) == 1
    loaded_tensor = loaded_prompt_embeds[0]["prompt_embeds"]
    assert loaded_tensor.device.type == "cpu"
    assert loaded_tensor.layout == torch.strided
    torch.testing.assert_close(loaded_tensor,
                               tensor.to("cpu").to_dense(),
                               equal_nan=True)
