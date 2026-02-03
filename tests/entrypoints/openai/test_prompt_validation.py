# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from unittest.mock import Mock

# imports for structured outputs tests
import openai
import pybase64
import pytest
import regex as re
import torch

from vllm.config import ModelConfig
from vllm.renderers.embed_utils import safe_load_prompt_embeds

from ...utils import RemoteOpenAIServer


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(
            openai.BadRequestError,
            match="Either prompt or prompt_embeds must be provided and non-empty.",
        ):
            await client.completions.create(
                model=model_name,
                prompt=None,
                max_tokens=5,
                temperature=0.0,
                extra_body={"prompt_embeds": []},
            )


@pytest.mark.asyncio
async def test_out_of_vocab_token_ids():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(
            openai.BadRequestError, match=re.compile(".*out of vocabulary.*").pattern
        ):
            await client.completions.create(
                model=model_name, prompt=[999999], max_tokens=5, temperature=0.0
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout", [torch.strided, torch.sparse_coo, torch.sparse_csc, torch.sparse_csr]
)
@pytest.mark.parametrize("seq_len", [2, 10])
@pytest.mark.parametrize("hidden_size", [2, 10])
def test_load_prompt_embeds(
    dtype: torch.dtype, layout: torch.layout, seq_len: int, hidden_size: int
):
    model_config = Mock(spec=ModelConfig)
    model_config.enable_prompt_embeds = True

    # construct arbitrary tensors of various dtypes, layouts, and sizes.
    # We need to check against different layouts to make sure that if a user
    # uses sparse tensors to reduce the transmission size of prompt embeddings,
    # we must cast them to dense/strided before passing them into the engine.
    # We don't use non-CPU tensors in this test to avoid preemptively
    # initializing cuda and break other tests in the suite that fork processes.
    # We also need to make sure that we only use devices that are actually
    # available in the environment the test is running on. For simplicity,
    # we just test against CPU.
    tensor = torch.randn((seq_len, hidden_size), dtype=dtype)
    if layout == torch.strided:
        tensor = tensor.contiguous()
    elif layout == torch.sparse_coo:
        tensor = tensor.to_sparse_coo()
    elif layout == torch.sparse_csc:
        tensor = tensor.to_sparse_csc()
    elif layout == torch.sparse_csr:
        tensor = tensor.to_sparse_csr()

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    encoded_tensor = pybase64.b64encode(buffer.getvalue())

    loaded_tensor = safe_load_prompt_embeds(model_config, encoded_tensor)
    assert loaded_tensor.device.type == "cpu"
    assert loaded_tensor.layout == torch.strided
    torch.testing.assert_close(
        loaded_tensor, tensor.to("cpu").to_dense(), equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("hidden_size", [2])
def test_disable_prompt_embeds(dtype: torch.dtype, seq_len: int, hidden_size: int):
    model_config = Mock(spec=ModelConfig)
    model_config.enable_prompt_embeds = False

    tensor = torch.randn((seq_len, hidden_size), dtype=dtype)

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    encoded_tensor = pybase64.b64encode(buffer.getvalue())

    with pytest.raises(ValueError, match="--enable-prompt-embeds"):
        safe_load_prompt_embeds(model_config, encoded_tensor)
