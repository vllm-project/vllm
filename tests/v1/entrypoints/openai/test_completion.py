# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, List, Optional

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from openai import BadRequestError

from tests.utils import RemoteOpenAIServer
from vllm.transformers_utils.tokenizer import get_tokenizer

# any model with a chat template should work here
MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager"
    ]


@pytest.fixture(scope="module",
                params=[["--no-enable-prefix-caching"],
                        [
                            "--no-enable-prefix-caching",
                            "--disable-frontend-multiprocessing"
                        ]])
def server(default_server_args, request):
    if request.param:
        default_server_args.extend(request.param)
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_single_completion(client: openai.AsyncOpenAI,
                                 model_name: str) -> None:
    completion = await client.completions.create(model=model_name,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

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
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_too_many_completion_logprobs(client: openai.AsyncOpenAI,
                                            model_name: str) -> None:

    with pytest.raises(
        (openai.BadRequestError, openai.APIError)):  # test using token IDs
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
        (openai.BadRequestError, openai.APIError)):  # test using token IDs
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
        async for chunk in stream:
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
@pytest.mark.parametrize("model_name, prompt_logprobs", [(MODEL_NAME, -1),
                                                         (MODEL_NAME, 0),
                                                         (MODEL_NAME, 1),
                                                         (MODEL_NAME, None)])
async def test_prompt_logprobs_completion(client: openai.AsyncOpenAI,
                                          model_name: str,
                                          prompt_logprobs: Optional[int]):
    params: Dict = {
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
async def test_completion_streaming(client: openai.AsyncOpenAI,
                                    model_name: str) -> None:
    prompt = "What is an LLM?"

    single_completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    single_output = single_completion.choices[0].text
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True)
    chunks: List[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_parallel_no_streaming(client: openai.AsyncOpenAI,
                                     model_name: str):
    """Parallel sampling without streaming.
    A single request output contains a list of completions.
    """

    prompt = "What is an LLM?"
    n = 3
    max_tokens = 5

    # High temperature to maximize chance of unique completions.
    completion = await client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 max_tokens=max_tokens,
                                                 n=n,
                                                 temperature=0.95,
                                                 stream=False,
                                                 seed=42)

    # Assert `n` completions
    num_completions = len(completion.choices)
    assert num_completions == n, (
        f"Num completions {num_completions} but expected {n}.")
    completion_repeats: Dict[str, int] = {}
    for idx, choice in enumerate(completion.choices):
        # Assert correct completion index & some finish reason.
        assert choice.index == idx, (
            f"Index {choice.index} but expected {idx}.")
        assert choice.finish_reason is not None, (
            "None finish_reason is invalid.")
        text = choice.text
        completion_repeats[text] = completion_repeats.get(text, 0) + 1
    # Assert `n` unique completions
    num_unique = len(completion_repeats)
    if num_unique != n:
        repeats = {
            txt: num
            for (txt, num) in completion_repeats.items() if num > 1
        }
        raise AssertionError(
            f"Expected {n} unique completions, got {num_unique};"
            f" repeats: {repeats}.")


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
    max_tokens = 5

    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=max_tokens,
                                             n=n,
                                             temperature=0.95,
                                             stream=True,
                                             seed=42)
    chunks: List[List[str]] = [[] for i in range(n)]
    finish_reason_count = 0
    async for chunk in stream:
        index = chunk.choices[0].index
        text = chunk.choices[0].text
        chunks[index].append(text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # Assert `n` completions with correct finish reasons
    assert finish_reason_count == n, (
        f"Expected {n} completions with valid indices and finish_reason.")
    completion_repeats: Dict[str, int] = {}
    for chunk in chunks:
        chunk_len = len(chunk)
        # Assert correct number of completion tokens
        assert chunk_len == max_tokens, (
            f"max_tokens={max_tokens} but chunk len is {chunk_len}.")
        text = "".join(chunk)
        completion_repeats[text] = completion_repeats.get(text, 0) + 1
        print(text)
    # Assert `n` unique completions
    num_unique = len(completion_repeats)
    if num_unique != n:
        repeats = {
            txt: num
            for (txt, num) in completion_repeats.items() if num > 1
        }
        raise AssertionError(f"{num_unique} unique completions, expected {n};"
                             f" repeats: {repeats}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_stream_options(client: openai.AsyncOpenAI,
                                         model_name: str):
    prompt = "What is the capital of France?"

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": False}
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True,
                                             stream_options={
                                                 "include_usage": False,
                                                 "continuous_usage_stats":
                                                 False,
                                             })

    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": True}
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True,
                                             stream_options={
                                                 "include_usage": False,
                                                 "continuous_usage_stats":
                                                 True,
                                             })
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": False}
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True,
                                             stream_options={
                                                 "include_usage": True,
                                                 "continuous_usage_stats":
                                                 False,
                                             })
    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": True}
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True,
                                             stream_options={
                                                 "include_usage": True,
                                                 "continuous_usage_stats":
                                                 True,
                                             })
    async for chunk in stream:
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens > 0
        assert chunk.usage.completion_tokens > 0
        assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                            chunk.usage.completion_tokens)
        if chunk.choices[0].finish_reason is not None:
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=False, stream_options=
    #     {"include_usage": None}
    with pytest.raises(BadRequestError):
        await client.completions.create(model=model_name,
                                        prompt=prompt,
                                        max_tokens=5,
                                        temperature=0.0,
                                        stream=False,
                                        stream_options={"include_usage": None})

    # Test stream=False, stream_options=
    #    {"include_usage": True}
    with pytest.raises(BadRequestError):
        await client.completions.create(model=model_name,
                                        prompt=prompt,
                                        max_tokens=5,
                                        temperature=0.0,
                                        stream=False,
                                        stream_options={"include_usage": True})

    # Test stream=False, stream_options=
    #     {"continuous_usage_stats": None}
    with pytest.raises(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": None})

    # Test stream=False, stream_options=
    #    {"continuous_usage_stats": True}
    with pytest.raises(BadRequestError):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": True})


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
                use_beam_search=True),
        )
        assert len(batch.choices) == 4
        assert batch.choices[0].text != batch.choices[
            1].text, "beam search should be different"
        assert batch.choices[0].text == batch.choices[
            2].text, "two copies of the same prompt should be the same"
        assert batch.choices[1].text == batch.choices[
            3].text, "two copies of the same prompt should be the same"

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
async def test_echo_logprob_completion(client: openai.AsyncOpenAI,
                                       model_name: str, logprobs_arg: int):
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    # test using text and token IDs
    for prompt in ("Hello, my name is", [0, 0, 0, 0, 0]):
        completion = await client.completions.create(model=model_name,
                                                     prompt=prompt,
                                                     max_tokens=5,
                                                     temperature=0.0,
                                                     echo=True,
                                                     logprobs=logprobs_arg)

        prompt_text = tokenizer.decode(prompt) if isinstance(prompt,
                                                             list) else prompt
        assert re.search(r"^" + prompt_text, completion.choices[0].text)
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) > 5
        assert (len(logprobs.token_logprobs) > 5
                and logprobs.token_logprobs[0] is None)
        assert (len(logprobs.top_logprobs) > 5
                and logprobs.top_logprobs[0] is None)
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg,
                       1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) > 5
