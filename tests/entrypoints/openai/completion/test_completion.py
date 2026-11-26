# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from typing import cast

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import regex as re
from openai import BadRequestError

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer

# any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen3-0.6B"

UNICODE_SERVER_ARGS = [
    "--dtype",
    "float16",
    "--max-model-len",
    "128",
    "--max-num-seqs",
    "4",
    "--attention-backend",
    "TRITON_ATTN",
    "--gpu-memory-utilization",
    "0.15",
    "--kv-cache-memory-bytes",
    "67108864",
]


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--dtype",
        "float32",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--enable-prompt-tokens-details",
        "--no-enable-prefix-caching",
    ]


@pytest.fixture(scope="module")
def server(default_server_args, request):
    params: (
        tuple[str, dict[str, str] | None]
        | tuple[str, dict[str, str] | None, list[str] | str]
    ) = getattr(request, "param", (MODEL_NAME, None))
    model_name, env_dict = params[:2]
    extra_args = params[2] if len(params) > 2 else []
    if isinstance(extra_args, str):
        extra_args = request.getfixturevalue(extra_args)
    args = [*default_server_args, *extra_args]
    with RemoteOpenAIServer(model_name, args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def _get_echo_logprobs(client, model, prompt, stream, max_tokens, extra_body):
    reference = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max(1, max_tokens),
        temperature=0,
        logprobs=0,
        extra_body={**extra_body, "return_token_ids": True},
    )
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        echo=True,
        max_tokens=max_tokens,
        temperature=0,
        logprobs=0,
        stream=stream,
        extra_body=extra_body,
    )
    choices = (
        [chunk.choices[0] async for chunk in response] if stream else response.choices
    )
    text = "".join(c.text for c in choices)
    offsets = [o for c in choices for o in c.logprobs.text_offset]
    tokens = [t for c in choices for t in c.logprobs.tokens]
    return reference.choices[0], text, offsets, tokens


@pytest.fixture(scope="module")
def unicode_eos_server_args(tmp_path_factory):
    from tokenizers import Tokenizer, processors
    from transformers import TokenizersBackend

    tokenizer = cast(
        TokenizersBackend, get_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")
    )
    backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.bos_token_id),
            ("</s>", tokenizer.eos_token_id),
        ],
    )
    path = tmp_path_factory.mktemp("unicode-eos-tokenizer")
    TokenizersBackend(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>"
    ).save_pretrained(path)
    return [*UNICODE_SERVER_ARGS, "--tokenizer", str(path)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            {"VLLM_USE_RUST_FRONTEND": "0", "VLLM_BATCH_INVARIANT": "1"},
            "unicode_eos_server_args",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize("prompt", [" 𐍈x", " 𐍈x </s>", " 𐍈x  "])
async def test_echo_unicode_offsets_with_added_eos(
    client, server, stream, max_tokens, prompt
):
    model = (await client.models.list()).data[0].id
    reference, text, offsets, tokens = await _get_echo_logprobs(
        client, model, prompt, stream, max_tokens, {}
    )
    count = len(reference.prompt_token_ids)
    assert text == prompt + (reference.text if max_tokens else "")
    assert tokens[0] == "<s>" and offsets[0] == 0
    assert tokens[count - 1] == "</s>" and offsets[count - 1] == len(prompt)
    assert offsets[:count] == sorted(offsets[:count])
    if "</s>" in prompt:
        assert tokens[count - 2] == "</s>"
        assert offsets[count - 2] == prompt.index("</s>")
    if max_tokens:
        assert offsets[count:] == [
            len(prompt) + o for o in reference.logprobs.text_offset
        ]
    else:
        assert len(offsets) == count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        (
            model,
            {"VLLM_USE_RUST_FRONTEND": "0"},
            [
                *UNICODE_SERVER_ARGS,
                "--tokenizer-mode",
                mode,
            ],
        )
        for model, mode in [
            ("facebook/opt-125m", "auto"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "auto"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ]
    ],
    indirect=True,
    ids=["bytelevel", "hf-fallback", "native-fallback"],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize("prompt", ["𐍈x", " 𐍈x", "a�x", "a�x suffix"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
async def test_echo_unicode_offsets(
    client, server, stream, max_tokens, prompt, add_special_tokens
):
    """Prompt offsets refer to echoed characters, including split UTF-8."""
    model = (await client.models.list()).data[0].id
    extra_body = {"add_special_tokens": add_special_tokens}
    truncated = prompt.endswith(" suffix")
    if truncated:
        # Retain the complete Unicode prefix, including any added BOS in the
        # model inputs. The echo must exclude that added BOS after truncation.
        prefix = prompt.removesuffix(" suffix")
        prefix_response = await client.completions.create(
            model=model,
            prompt=prefix,
            max_tokens=1,
            temperature=0,
            extra_body={**extra_body, "return_token_ids": True},
        )
        extra_body.update(
            truncate_prompt_tokens=len(prefix_response.choices[0].prompt_token_ids),
            truncation_side="right",
        )
    reference, text, offsets, tokens = await _get_echo_logprobs(
        client, model, prompt, stream, max_tokens, extra_body
    )
    prompt_ids = reference.prompt_token_ids
    assert prompt_ids is not None
    if truncated:
        assert prompt_ids == prefix_response.choices[0].prompt_token_ids
        prompt = prefix
    count = len(prompt_ids) - int(truncated and add_special_tokens)
    assert text == prompt + (reference.text if max_tokens else "")
    assert offsets[count - 1] == len(prompt) - 1
    assert tokens[count - 1] == "x"
    if prompt == "𐍈x":
        assert offsets[: count - 1] == [0] * (count - 1)
    elif prompt == "a�x":
        assert offsets[:count] == [0] * (count - 2) + [1, 2]
    if max_tokens:
        assert offsets[count:] == [
            len(prompt) + o for o in reference.logprobs.text_offset
        ]
        assert tokens[count:] == reference.logprobs.tokens
    else:
        assert len(offsets) == count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_single_completion(client: openai.AsyncOpenAI, model_name: str) -> None:
    completion = await client.completions.create(
        model=model_name, prompt="Hello, my name is", max_tokens=5, temperature=0.0
    )

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 5
    assert completion.usage.prompt_tokens == 5
    assert completion.usage.total_tokens == 10
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cached_tokens == 0

    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_truncation_side_controls_prompt_truncation(
    client: openai.AsyncOpenAI, model_name: str
) -> None:
    prompt_token_ids = list(range(8))

    right_completion = await client.completions.create(
        model=model_name,
        prompt=prompt_token_ids,
        max_tokens=1,
        temperature=0.0,
        extra_body={
            "return_token_ids": True,
            "truncate_prompt_tokens": 4,
            "truncation_side": "right",
        },
    )
    assert right_completion.choices[0].prompt_token_ids == prompt_token_ids[:4]

    left_completion = await client.completions.create(
        model=model_name,
        prompt=prompt_token_ids,
        max_tokens=1,
        temperature=0.0,
        extra_body={
            "return_token_ids": True,
            "truncate_prompt_tokens": 4,
            "truncation_side": "left",
        },
    )
    assert left_completion.choices[0].prompt_token_ids == prompt_token_ids[-4:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    # Exercise the Python renderer even when this module runs in Rust frontend CI.
    [("facebook/opt-125m", {"VLLM_USE_RUST_FRONTEND": "0"})],
    indirect=True,
    ids=["python-opt"],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize(
    "prompt,keep,expected",
    [
        ("Hello, my name is", 4, "Hello, my"),
        ("</s>Hello, my name is", 4, "</s>Hello,"),
        ("Hello, my name is", 1, ""),
    ],
)
async def test_truncated_echo_excludes_only_added_special_tokens(
    client: openai.AsyncOpenAI,
    prompt: str,
    keep: int,
    expected: str,
    stream: bool,
    max_tokens: int,
) -> None:
    extra_body = {"truncate_prompt_tokens": keep, "truncation_side": "right"}
    generated = await client.completions.create(
        model="facebook/opt-125m",
        prompt=prompt,
        max_tokens=max(1, max_tokens),
        temperature=0.0,
        extra_body={**extra_body, "return_token_ids": True},
    )
    tokenizer = get_tokenizer("facebook/opt-125m")
    assert generated.choices[0].prompt_token_ids == tokenizer.encode(prompt)[:keep]
    assert generated.usage is not None
    assert generated.usage.prompt_tokens == keep
    response = await client.completions.create(
        model="facebook/opt-125m",
        prompt=prompt,
        echo=True,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=0,
        stream=stream,
        extra_body=extra_body,
    )
    choices = (
        [chunk.choices[0] async for chunk in response] if stream else response.choices
    )
    if not stream:
        assert response.usage is not None
        assert response.usage.prompt_tokens == keep
    text = "".join(choice.text for choice in choices)
    generated_text = generated.choices[0].text if max_tokens else ""
    assert text == expected + generated_text
    offset = 0
    for choice in choices:
        assert choice.logprobs is not None
        assert len(choice.logprobs.tokens) == len(choice.logprobs.token_logprobs)
        assert len(choice.logprobs.tokens) == len(choice.logprobs.top_logprobs)
        for token, text_offset in zip(
            choice.logprobs.tokens, choice.logprobs.text_offset, strict=True
        ):
            assert text_offset == offset
            assert text[text_offset : text_offset + len(token)] == token
            offset += len(token)
    assert offset == len(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_no_logprobs(client: openai.AsyncOpenAI, model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=None,
    )
    choice = completion.choices[0]
    assert choice.logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_zero_logprobs(client: openai.AsyncOpenAI, model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=0,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert len(choice.logprobs.top_logprobs[0]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_some_logprobs(client: openai.AsyncOpenAI, model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=5,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert 5 <= len(choice.logprobs.top_logprobs[0]) <= 6


@pytest.mark.asyncio
async def test_logprob_token_ids(client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
        logprobs=5,
        extra_body={
            "logprob_token_ids": [5000],
            "allowed_token_ids": [42],
            "return_tokens_as_token_ids": True,
        },
    )

    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.tokens == ["token_id:42"]
    assert choice.logprobs.top_logprobs is not None
    assert set(choice.logprobs.top_logprobs[0]) == {
        "token_id:42",
        "token_id:5000",
    }


@pytest.mark.asyncio
async def test_logprob_token_ids_stream(client: openai.AsyncOpenAI):
    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
        logprobs=5,
        stream=True,
        extra_body={
            "logprob_token_ids": [5000],
            "allowed_token_ids": [42],
            "return_tokens_as_token_ids": True,
        },
    )

    returned_top_logprobs: list[dict[str, float]] = []
    async for chunk in stream:
        logprobs = chunk.choices[0].logprobs
        if logprobs is not None and logprobs.top_logprobs is not None:
            returned_top_logprobs.extend(
                top_logprobs
                for top_logprobs in logprobs.top_logprobs
                if top_logprobs is not None
            )

    assert len(returned_top_logprobs) == 1
    assert set(returned_top_logprobs[0]) == {"token_id:42", "token_id:5000"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_too_many_completion_logprobs(
    client: openai.AsyncOpenAI, model_name: str
) -> None:
    with pytest.raises(
        (openai.BadRequestError, openai.APIError)
    ):  # test using token IDs
        await client.completions.create(
            model=model_name,
            prompt=[0, 0, 0, 0, 0],
            max_tokens=5,
            temperature=0.0,
            # vLLM has higher default max_logprobs (20 instead of 5) to support
            # both Completion API and Chat Completion API
            logprobs=21,
        )
        ...
    with pytest.raises(
        (openai.BadRequestError, openai.APIError)
    ):  # test using token IDs
        stream = await client.completions.create(
            model=model_name,
            prompt=[0, 0, 0, 0, 0],
            max_tokens=5,
            temperature=0.0,
            # vLLM has higher default max_logprobs (20 instead of 5) to support
            # both Completion API and Chat Completion API
            logprobs=30,
            stream=True,
        )
        async for _ in stream:
            ...

    # the server should still work afterwards
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, prompt_logprobs",
    [(MODEL_NAME, -1), (MODEL_NAME, 0), (MODEL_NAME, 1), (MODEL_NAME, None)],
)
async def test_prompt_logprobs_completion(
    client: openai.AsyncOpenAI, model_name: str, prompt_logprobs: int | None
):
    params: dict = {
        "prompt": ["A robot may not injure another robot", "My name is"],
        "model": model_name,
    }
    if prompt_logprobs is not None:
        params["extra_body"] = {"prompt_logprobs": prompt_logprobs}

    if prompt_logprobs is not None and prompt_logprobs < 0:
        with pytest.raises(BadRequestError):
            await client.completions.create(**params)
    else:
        completion = await client.completions.create(**params)
        if prompt_logprobs is not None:
            assert completion.choices[0].prompt_logprobs is not None
            assert len(completion.choices[0].prompt_logprobs) > 0

            assert completion.choices[1].prompt_logprobs is not None
            assert len(completion.choices[1].prompt_logprobs) > 0

        else:
            assert completion.choices[0].prompt_logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_streaming(
    client: openai.AsyncOpenAI, model_name: str
) -> None:
    prompt = "What is an LLM?"

    single_completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    single_output = single_completion.choices[0].text
    stream = await client.completions.create(
        model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True
    )
    chunks: list[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    # The last chunk may carry only the finish_reason (Rust); the joined text is
    # checked against the non-streaming output below.
    assert "".join(chunks) == single_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_parallel_no_streaming(client: openai.AsyncOpenAI, model_name: str):
    """Parallel sampling without streaming.
    A single request output contains a list of completions.
    """
    prompt = "What is an LLM?"
    n = 3
    max_tokens = 50

    # High temperature to maximize chance of unique completions.
    completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        temperature=1.0,
        stream=False,
        logprobs=0,
        seed=42,
    )

    # Assert `n` completions
    num_completions = len(completion.choices)
    assert num_completions == n, f"Num completions {num_completions} but expected {n}."
    completion_repeats: dict[str, int] = {}
    for idx, choice in enumerate(completion.choices):
        # Assert correct completion index & some finish reason.
        assert choice.index == idx, f"Index {choice.index} but expected {idx}."
        assert choice.finish_reason is not None, "None finish_reason is invalid."
        text = choice.text
        completion_repeats[text] = completion_repeats.get(text, 0) + 1
    # Assert `n` unique completions
    num_unique = len(completion_repeats)
    if num_unique != n:
        repeats = {txt: num for (txt, num) in completion_repeats.items() if num > 1}
        raise AssertionError(
            f"Expected {n} unique completions, got {num_unique}; repeats: {repeats}."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_parallel_streaming(client: openai.AsyncOpenAI, model_name: str):
    """Streaming for parallel sampling.
    The tokens from multiple samples, are flattened into a single stream,
    with an index to indicate which sample the token belongs to.
    """
    prompt = "What is an LLM?"
    n = 3
    max_tokens = 50

    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        temperature=1.0,
        stream=True,
        seed=42,
    )
    chunks: list[list[str]] = [[] for _ in range(n)]
    finish_reason_count = 0
    async for chunk in stream:
        index = chunk.choices[0].index
        text = chunk.choices[0].text
        # The last chunk of a sample may carry only the finish_reason (Rust).
        if text or chunk.choices[0].finish_reason is None:
            chunks[index].append(text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # Assert `n` completions with correct finish reasons
    assert finish_reason_count == n, (
        f"Expected {n} completions with valid indices and finish_reason."
    )
    completion_repeats: dict[str, int] = {}
    for chunk in chunks:
        chunk_len = len(chunk)
        # Assert correct number of completion tokens
        assert chunk_len <= max_tokens, (
            f"max_tokens={max_tokens} but chunk len is {chunk_len}."
        )
        text = "".join(chunk)
        completion_repeats[text] = completion_repeats.get(text, 0) + 1
        print(text)
    # Assert `n` unique completions
    num_unique = len(completion_repeats)
    if num_unique != n:
        repeats = {txt: num for (txt, num) in completion_repeats.items() if num > 1}
        raise AssertionError(
            f"{num_unique} unique completions, expected {n}; repeats: {repeats}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_stream_options(client: openai.AsyncOpenAI, model_name: str):
    prompt = "What is the capital of France?"

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": False}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": False,
            "continuous_usage_stats": False,
        },
    )

    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": True}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": False,
            "continuous_usage_stats": True,
        },
    )
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": False}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": False,
        },
    )
    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await anext(stream)
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens + final_chunk.usage.completion_tokens
            )
            assert final_chunk.choices == []

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": True}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    )
    async for chunk in stream:
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens > 0
        assert chunk.usage.completion_tokens > 0
        assert chunk.usage.total_tokens == (
            chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        )
        if chunk.choices[0].finish_reason is not None:
            final_chunk = await anext(stream)
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens + final_chunk.usage.completion_tokens
            )
            assert final_chunk.choices == []

    # Test stream=True, stream_options={}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={},
    )
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=False, stream_options=
    #     {"include_usage": None}
    # A null option sets nothing: Python rejects it, Rust accepts it as unset.
    with suppress(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None},
        )

    # Test stream=False, stream_options=
    #    {"include_usage": True}
    with pytest.raises(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True},
        )

    # Test stream=False, stream_options=
    #     {"continuous_usage_stats": None}
    # A null option sets nothing: Python rejects it, Rust accepts it as unset.
    with suppress(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": None},
        )

    # Test stream=False, stream_options=
    #    {"continuous_usage_stats": True}
    with pytest.raises(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_batch_completions(client: openai.AsyncOpenAI, model_name: str):
    # test both text and token IDs
    for prompts in (["Hello, my name is"] * 2, [[0, 0, 0, 0, 0]] * 2):
        # test simple list
        batch = await client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=5,
            temperature=0.0,
        )
        assert len(batch.choices) == 2
        assert batch.choices[0].text == batch.choices[1].text

        # test n = 2
        batch = await client.completions.create(
            model=model_name,
            prompt=prompts,
            n=2,
            max_tokens=5,
            temperature=0.0,
            extra_body=dict(
                # NOTE: this has to be true for n > 1 in vLLM, but
                # not necessary for official client.
                use_beam_search=True
            ),
        )
        assert len(batch.choices) == 4
        assert batch.choices[0].text != batch.choices[1].text, (
            "beam search should be different"
        )
        assert batch.choices[0].text == batch.choices[2].text, (
            "two copies of the same prompt should be the same"
        )
        assert batch.choices[1].text == batch.choices[3].text, (
            "two copies of the same prompt should be the same"
        )

        # test streaming
        batch = await client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=5,
            temperature=0.0,
            stream=True,
        )
        texts = [""] * 2
        async for chunk in batch:
            assert len(chunk.choices) == 1
            choice = chunk.choices[0]
            texts[choice.index] += choice.text
        assert texts[0] == texts[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
@pytest.mark.parametrize("logprobs_arg", [1, 0])
async def test_echo_logprob_completion(
    client: openai.AsyncOpenAI, model_name: str, logprobs_arg: int
):
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    # test using text and token IDs
    for prompt in ("Hello, my name is", [0, 0, 0, 0, 0]):
        completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            echo=True,
            logprobs=logprobs_arg,
        )

        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        assert re.search(r"^" + prompt_text, completion.choices[0].text)
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) > 5
        assert len(logprobs.token_logprobs) > 5 and logprobs.token_logprobs[0] is None
        assert len(logprobs.top_logprobs) > 5 and logprobs.top_logprobs[0] is None
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg, 1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) > 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(client: openai.AsyncOpenAI, model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {"title": "Brand", "type": "string"},
            "model": {"title": "Model", "type": "string"},
            "car_type": {"$ref": "#/$defs/CarType"},
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={"structured_outputs": {"json": invalid_json_schema}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = (
        "Generate an email address for Alan Turing, who works in Enigma."
        "End in .com and new line. Example result:"
        "alan.turing@enigma.com\n"
    )

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={"structured_outputs": {"regex": r"[.*"}, "stop": ["\n"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(client: openai.AsyncOpenAI, model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={
                "structured_outputs": {"grammar": invalid_simplified_sql_grammar}
            },
        )


# Unit tests for bad_words in CompletionRequest.to_sampling_params()
def test_completion_request_bad_words_to_sampling_params():
    """bad_words should be forwarded to SamplingParams (parity with chat)."""
    request = CompletionRequest(
        model="test-model",
        prompt="Hello",
        bad_words=["foo", "bar"],
        max_tokens=10,
    )

    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.bad_words == ["foo", "bar"]


def test_completion_request_bad_words_default_empty():
    """bad_words defaults to an empty list, matching the chat endpoint."""
    request = CompletionRequest(
        model="test-model",
        prompt="Hello",
        max_tokens=10,
    )

    assert request.bad_words == []
    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )
    assert sampling_params.bad_words == []


def test_completion_request_forwards_routed_experts_prompt_start():
    request = CompletionRequest(
        model="test-model",
        prompt="Hello",
        max_tokens=10,
        routed_experts_prompt_start=3,
    )

    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert sampling_params.routed_experts_prompt_start == 3
