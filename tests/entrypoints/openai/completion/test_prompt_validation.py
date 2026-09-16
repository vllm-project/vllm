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

from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers.embed_utils import safe_load_prompt_embeds


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "openai-community/gpt2"
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
    model_name = "openai-community/gpt2"
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
    model_config.get_hidden_size.return_value = hidden_size
    model_config.dtype = dtype

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

    with pytest.raises(VLLMValidationError, match="--enable-prompt-embeds"):
        safe_load_prompt_embeds(model_config, encoded_tensor)


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"n": 2},
        {"use_beam_search": True},
    ],
)
def test_inline_hidden_states_rejects_unsupported_http_shapes(endpoint, options):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    cls, prompt = (
        (CompletionRequest, {"prompt": "hello"})
        if endpoint == "completion"
        else (
            ChatCompletionRequest,
            {"messages": [{"role": "user", "content": "hello"}]},
        )
    )
    with pytest.raises(VLLMValidationError, match="return_inline"):
        cls(
            model="test",
            max_tokens=1,
            kv_transfer_params={"return_inline": True},
            **prompt,
            **options,
        )
    ordinary = cls(model="test", max_tokens=128, **prompt, **options)
    assert ordinary.max_tokens == 128


@pytest.mark.parametrize("prompt", [["a", "b"], [[1], [2]]])
def test_inline_hidden_states_rejects_multiple_completion_prompts(prompt):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    with pytest.raises(VLLMValidationError, match="single prompt"):
        CompletionRequest(
            prompt=prompt, max_tokens=1, kv_transfer_params={"return_inline": True}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_frontend", [False, True], ids=["python", "rust"])
async def test_inline_hidden_states_with_concurrent_streaming(
    tmp_path, rust_frontend, monkeypatch
):
    """A single generation server remains usable before/during/after extraction."""
    import asyncio
    import shutil

    from transformers import AutoConfig

    import vllm.envs as envs

    server_env = {
        "VLLM_USE_RUST_FRONTEND": "1" if rust_frontend else "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    if rust_frontend:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_RUST_FRONTEND", "1")
            try:
                binary_path = envs.VLLM_RUST_FRONTEND_PATH
            except FileNotFoundError:
                binary_path = None
        binary = shutil.which(binary_path) if binary_path else None
        if binary is not None:
            server_env["VLLM_RUST_FRONTEND_PATH"] = binary
        else:
            pytest.skip("Build vllm-rs and set VLLM_RUST_FRONTEND_PATH")

    model = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    config = AutoConfig.from_pretrained(model, revision=revision).get_text_config()
    args = [
        "--revision",
        revision,
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]
    if not rust_frontend:
        args.extend(["--tokenizer-revision", revision])
    with RemoteOpenAIServer(model, args, env_dict=server_env) as server:
        async with server.get_async_client() as client:

            async def normal_stream():
                stream = await client.completions.create(
                    model=model,
                    prompt="A forest is",
                    max_tokens=32,
                    stream=True,
                    temperature=0,
                    extra_body={"ignore_eos": True},
                )
                chunks = [chunk async for chunk in stream]
                assert chunks[-1].choices[0].finish_reason == "length"
                assert all(
                    not getattr(chunk, "kv_transfer_params", None) for chunk in chunks
                )
                return "".join(chunk.choices[0].text for chunk in chunks)

            before = await normal_stream()
            normal, inline = await asyncio.gather(
                normal_stream(),
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Describe a forest."}],
                    max_tokens=1,
                    extra_body={"kv_transfer_params": {"return_inline": True}},
                ),
            )
            payload = inline.kv_transfer_params
            assert len(payload["hidden_states"]) == config.hidden_size
            assert payload["representation"] == "post_final_norm"
            assert all(torch.isfinite(torch.tensor(payload["hidden_states"])))
            with pytest.raises(openai.BadRequestError, match="return_inline"):
                await client.completions.create(
                    model=model,
                    prompt="hello",
                    max_tokens=2,
                    extra_body={"kv_transfer_params": {"return_inline": True}},
                )
            completion = await client.completions.create(
                model=model,
                prompt="A forest is",
                max_tokens=1,
                extra_body={"kv_transfer_params": {"return_inline": True}},
            )
            assert (
                len(completion.kv_transfer_params["hidden_states"])
                == config.hidden_size
            )
            assert completion.kv_transfer_params["layer_id"] == config.num_hidden_layers
            assert before == normal == await normal_stream()
            assert not list(tmp_path.iterdir())
