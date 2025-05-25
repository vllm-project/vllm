# SPDX-License-Identifier: Apache-2.0

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from openai import BadRequestError

from tests.utils import RemoteOpenAIServer

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


class TestCompletionStreaming:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_name",
        [MODEL_NAME],
    )
    async def test_too_many_completion_logprobs(self,
                                                client: openai.AsyncOpenAI,
                                                model_name: str) -> None:

        with pytest.raises(
            (openai.BadRequestError, openai.APIError)):  # test using token IDs
            await client.completions.create(
                model=model_name,
                prompt=[0, 0, 0, 0, 0],
                max_tokens=5,
                temperature=0.0,
                # vLLM has higher default max_logprobs (20 instead of 5)
                # to support both Completion API and Chat Completion API
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
                # vLLM has higher default max_logprobs (20 instead of 5)
                # to support both Completion API and Chat Completion API
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
    @pytest.mark.parametrize(
        "model_name",
        [MODEL_NAME],
    )
    async def test_completion_streaming(self, client: openai.AsyncOpenAI,
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
        chunks: list[str] = []
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
    async def test_parallel_streaming(self, client: openai.AsyncOpenAI,
                                      model_name: str):
        """Streaming for parallel sampling.
        The tokens from multiple samples, are flattened into a single stream,
        with an index to indicate which sample the token belongs to.
        """

        prompt = "What is an LLM?"
        n = 3
        max_tokens = 50  # we want some to finish earlier than others

        stream = await client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 max_tokens=max_tokens,
                                                 n=n,
                                                 temperature=1.0,
                                                 stream=True,
                                                 seed=42)
        chunks: list[list[str]] = [[] for _ in range(n)]
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
        completion_repeats: dict[str, int] = {}
        chunk_lengths = set()
        for chunk in chunks:
            chunk_len = len(chunk)
            # Assert correct number of completion tokens
            chunk_lengths.add(chunk_len)
            assert chunk_len <= max_tokens, (
                f"max_tokens={max_tokens} but chunk len is {chunk_len}.")
            text = "".join(chunk)
            completion_repeats[text] = completion_repeats.get(text, 0) + 1
            print(text)
        # Assert subrequests finished at different times
        assert len(chunk_lengths) > 1
        # Assert `n` unique completions
        num_unique = len(completion_repeats)
        if num_unique != n:
            repeats = {
                txt: num
                for (txt, num) in completion_repeats.items() if num > 1
            }
            raise AssertionError(
                f"{num_unique} unique completions, expected {n};"
                f" repeats: {repeats}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_name",
        [MODEL_NAME],
    )
    async def test_completion_stream_options(self, client: openai.AsyncOpenAI,
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
                                                     "include_usage":
                                                     False,
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
                                                     "include_usage":
                                                     False,
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
                                                     "include_usage":
                                                     True,
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
                                                     "include_usage":
                                                     True,
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
            await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=5,
                temperature=0.0,
                stream=False,
                stream_options={"include_usage": None})

        # Test stream=False, stream_options=
        #    {"include_usage": True}
        with pytest.raises(BadRequestError):
            await client.completions.create(
                model=model_name,
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
    async def test_batch_completions(self, client: openai.AsyncOpenAI,
                                     model_name: str):
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
