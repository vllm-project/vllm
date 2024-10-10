# imports for guided decoding tests
import json
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import jsonschema
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
# downloading lora to test lora requests
from huggingface_hub import snapshot_download
from openai import BadRequestError
from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically these adapters use a different base model,
# but we're not testing generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"
PA_NAME = "swapnilbp/llama_tweet_ptune"
# if PA_NAME changes, PA_NUM_VIRTUAL_TOKENS might also
# need to change to match the prompt adapter
PA_NUM_VIRTUAL_TOKENS = 8


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def zephyr_lora_added_tokens_files(zephyr_lora_files):
    tmp_dir = TemporaryDirectory()
    tmp_model_dir = f"{tmp_dir.name}/zephyr"
    shutil.copytree(zephyr_lora_files, tmp_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Copy tokenizer to adapter and add some unique tokens
    # 32000, 32001, 32002
    added = tokenizer.add_tokens(["vllm1", "vllm2", "vllm3"],
                                 special_tokens=True)
    assert added == 3
    tokenizer.save_pretrained(tmp_model_dir)
    yield tmp_model_dir
    tmp_dir.cleanup()


@pytest.fixture(scope="module")
def zephyr_pa_files():
    return snapshot_download(repo_id=PA_NAME)


@pytest.fixture(scope="module")
def default_server_args(zephyr_lora_files, zephyr_lora_added_tokens_files,
                        zephyr_pa_files):
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        # lora config
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora={zephyr_lora_files}",
        f"zephyr-lora2={zephyr_lora_added_tokens_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        # pa config
        "--enable-prompt-adapter",
        "--prompt-adapters",
        f"zephyr-pa={zephyr_pa_files}",
        f"zephyr-pa2={zephyr_pa_files}",
        "--max-prompt-adapters",
        "2",
        "--max-prompt-adapter-token",
        "128",
    ]


@pytest.fixture(scope="module",
                params=["", "--disable-frontend-multiprocessing"])
def server(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras, then test prompt adapters
    "model_name,num_virtual_tokens",
    [(MODEL_NAME, 0), ("zephyr-lora", 0), ("zephyr-lora2", 0),
     ("zephyr-pa", PA_NUM_VIRTUAL_TOKENS),
     ("zephyr-pa2", PA_NUM_VIRTUAL_TOKENS)],
)
async def test_single_completion(client: openai.AsyncOpenAI, model_name: str,
                                 num_virtual_tokens: int):
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
        completion_tokens=5,
        prompt_tokens=6 + num_virtual_tokens,
        total_tokens=11 + num_virtual_tokens)

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
async def test_added_lora_tokens(client: openai.AsyncOpenAI):
    # test using token IDs
    completion = await client.completions.create(
        model="zephyr-lora2",
        prompt=[0, 0, 32000, 32001, 32002],
        echo=True,
        max_tokens=5,
        temperature=0.0,
    )
    # Added tokens should appear in tokenized prompt
    assert completion.choices[0].text.startswith("<unk><unk>vllm1vllm2vllm3")


@pytest.mark.asyncio
async def test_added_lora_tokens_base_model(client: openai.AsyncOpenAI):
    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 32000, 32001, 32002],
        echo=True,
        max_tokens=5,
        temperature=0.0,
    )
    # Added tokens should not appear in tokenized prompt
    assert "vllm" not in completion.choices[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras, then test prompt adapters
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2", "zephyr-pa", "zephyr-pa2"],
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
    # just test 1 lora and 1 pa hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
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
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
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
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
)
async def test_too_many_completion_logprobs(client: openai.AsyncOpenAI,
                                            model_name: str):

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
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
)
async def test_completion_streaming(client: openai.AsyncOpenAI,
                                    model_name: str):
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
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
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
    [MODEL_NAME, "zephyr-lora", "zephyr-pa"],
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
async def test_logits_bias(client: openai.AsyncOpenAI):
    prompt = "Hello, my name is"
    max_tokens = 5
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

    # Test exclusive selection
    token_id = 1000
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        logit_bias={str(token_id): 100},
        seed=42,
    )
    assert len(completion.choices[0].text) >= 5
    response_tokens = tokenizer(completion.choices[0].text,
                                add_special_tokens=False)["input_ids"]
    expected_tokens = tokenizer(tokenizer.decode([token_id] * 5),
                                add_special_tokens=False)["input_ids"]
    assert all([
        response == expected
        for response, expected in zip(response_tokens, expected_tokens)
    ])

    # Test ban
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    response_tokens = tokenizer(completion.choices[0].text,
                                add_special_tokens=False)["input_ids"]
    first_response = completion.choices[0].text
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        logit_bias={str(token): -100
                    for token in response_tokens},
    )
    assert first_response != completion.choices[0].text


@pytest.mark.asyncio
async def test_allowed_token_ids(client: openai.AsyncOpenAI):
    prompt = "Hello, my name is"
    max_tokens = 1
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

    # Test exclusive selection
    allowed_ids = [21555, 21557, 21558]
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=42,
        extra_body=dict(allowed_token_ids=allowed_ids),
        logprobs=1,
    )
    response_tokens = completion.choices[0].logprobs.tokens
    assert len(response_tokens) == 1
    assert tokenizer.convert_tokens_to_ids(response_tokens)[0] in allowed_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_json_completion(client: openai.AsyncOpenAI,
                                      guided_decoding_backend: str,
                                      sample_json_schema):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example JSON for an employee profile "
        f"that fits this schema: {sample_json_schema}",
        n=3,
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(guided_json=sample_json_schema,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 3
    for i in range(3):
        output_json = json.loads(completion.choices[i].text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_regex_completion(client: openai.AsyncOpenAI,
                                       guided_decoding_backend: str,
                                       sample_regex):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example IPv4 address with this regex: {sample_regex}",
        n=3,
        temperature=1.0,
        max_tokens=20,
        extra_body=dict(guided_regex=sample_regex,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 3
    for i in range(3):
        assert re.fullmatch(sample_regex,
                            completion.choices[i].text) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_completion(client: openai.AsyncOpenAI,
                                        guided_decoding_backend: str,
                                        sample_guided_choice):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt="The best language for type-safe systems programming is ",
        n=2,
        temperature=1.0,
        max_tokens=10,
        extra_body=dict(guided_choice=sample_guided_choice,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 2
    for i in range(2):
        assert completion.choices[i].text in sample_guided_choice


@pytest.mark.asyncio
async def test_guided_grammar(client: openai.AsyncOpenAI,
                              sample_sql_statements):

    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=("Generate a sql state that select col_1 from "
                "table_1 where it is equals to 1"),
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(guided_grammar=sample_sql_statements))

    content = completion.choices[0].text

    # use Lark to parse the output, and make sure it's a valid parse tree
    from lark import Lark
    parser = Lark(sample_sql_statements)
    parser.parse(content)

    # remove spaces for comparison b/c we removed them in the grammar
    ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

    assert content.strip() == ground_truth


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_decoding_type_error(client: openai.AsyncOpenAI,
                                          guided_decoding_backend: str,
                                          sample_json_schema, sample_regex):
    with pytest.raises(openai.BadRequestError):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example JSON that fits this schema: 42",
            extra_body=dict(guided_json=42,
                            guided_decoding_backend=guided_decoding_backend))

    with pytest.raises(openai.BadRequestError):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example string that fits this regex",
            extra_body=dict(guided_regex=sample_regex,
                            guided_json=sample_json_schema))
