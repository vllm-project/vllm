# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import httpx
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig
from vllm.config.utils import getattr_iter
from vllm.v1.engine.detokenizer import check_stop_strings

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"


def get_vocab_size(model_name):
    config = ModelConfig(
        model=model_name,
        seed=0,
        dtype="bfloat16",
    )
    return config.get_vocab_size()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]


@pytest.fixture(scope="module")
def server(request):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        # On ROCm (e.g. MI355X/gfx950), bf16 GEMM results can differ by
        # 1 ULP when the batch dimension (M) changes, because different M
        # values cause the Tensile backend to select different tile
        # configurations with different fp32 accumulation orders. With
        # prefix caching, cache-miss prefills compute all tokens in one
        # pass (large M) while cache-hit requests compute only the
        # uncached suffix (small M), seeding a divergence that amplifies
        # through the residual stream and flips argmax tokens.
        # See: https://github.com/vllm-project/vllm/issues/33123
        #
        # Either disable prefix caching entirely, or enable it with
        # --deterministic-prefix-caching which forces cache-miss prefills
        # to split at block boundaries so the suffix GEMM shape is always
        # identical regardless of cache state.
        #
        # Option A: disable prefix caching
        "--no-enable-prefix-caching",
        #
        # Option B: deterministic prefix caching
        # "--enable-prefix-caching",
        # "--deterministic-prefix-caching",
    ]

    extra_args = getattr(request, "param", None)
    if extra_args is not None:
        args = args + (
            list(extra_args)
            if isinstance(extra_args, (list, tuple))
            else [str(extra_args)]
        )

    envs = os.environ.copy()
    # See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3888060787
    envs["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers=headers,
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_generate_endpoint(client):
    payload = {
        "model": MODEL_NAME,
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 5},
        "stream": False,
    }
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_generate_stream(client):
    payload = {
        "model": MODEL_NAME,
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 5},
        "stream": True,
    }
    async with client.stream("POST", GEN_ENDPOINT, json=payload) as resp:
        resp.raise_for_status()
        chunks = []
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: ") :]
            if payload_str == "[DONE]":
                break
            chunks.append(json.loads(payload_str))

    assert len(chunks) > 0
    # Every chunk has choices with token_ids
    all_token_ids = []
    for chunk in chunks:
        assert "choices" in chunk
        assert len(chunk["choices"]) == 1
        choice = chunk["choices"][0]
        assert "token_ids" in choice
        assert len(choice["token_ids"]) > 0
        all_token_ids.extend(choice["token_ids"])

    # Last chunk should have a finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] is not None

    # Streaming should produce the same tokens as non-streaming
    non_stream_resp = await client.post(
        GEN_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "token_ids": [1, 2, 3],
            "sampling_params": {"max_tokens": 5, "temperature": 0.0},
            "stream": False,
        },
    )
    non_stream_data = non_stream_resp.json()
    # Just verify we got the right number of tokens
    assert len(all_token_ids) == len(non_stream_data["choices"][0]["token_ids"])


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs_value", [0, 1, 5])
async def test_generate_logprobs(client, logprobs_value):
    payload = {
        "model": MODEL_NAME,
        "token_ids": [1, 2, 3],
        "sampling_params": {
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": logprobs_value,
        },
        "stream": False,
    }
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    assert choice["logprobs"] is not None
    logprobs_content = choice["logprobs"]["content"]
    assert len(logprobs_content) == len(choice["token_ids"])
    for entry in logprobs_content:
        assert "logprob" in entry
        assert len(entry["top_logprobs"]) >= 1
        assert len(entry["top_logprobs"]) == max(logprobs_value, 1)


@pytest.mark.asyncio
async def test_same_response_as_chat_completions(client, tokenizer, messages):
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # default with Qwen3
        return_dict=True,  # default with Transformers v5
    ).input_ids

    for ignore_eos in [True, False]:
        payload = {
            "model": MODEL_NAME,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 24,
                "temperature": 0.0,
                # NOTE coordinator will set this to skip detokenization
                "detokenize": False,
                "ignore_eos": ignore_eos,
            },
            "stream": False,
        }
        generate_resp = await client.post(GEN_ENDPOINT, json=payload)
        generate_data = generate_resp.json()
        gen_token_ids = generate_data["choices"][0]["token_ids"]
        generate_res = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 24,
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        completions_resp = await client.post("/v1/chat/completions", json=payload)
        completions_data = completions_resp.json()
        completions_res = completions_data["choices"][0]["message"]["content"]

        if ignore_eos:
            # When ignoring EOS, only compare up to the first EOS token
            # Post-EOS generation is undefined and may differ
            eos_tokens = {
                tokenizer.eos_token_id,
                *getattr_iter(
                    tokenizer,
                    [
                        "extra_special_tokens_ids",  # Transformers v5
                        "additional_special_tokens_ids",  # Transformers v4
                    ],
                    [],
                ),
            }
            # Find first EOS in generated tokens
            eos_pos = None
            for i, tid in enumerate(gen_token_ids):
                if tid in eos_tokens:
                    eos_pos = i
                    break
            if eos_pos is not None:
                gen_token_ids_truncated = gen_token_ids[:eos_pos]
                generate_res = tokenizer.decode(
                    gen_token_ids_truncated, skip_special_tokens=True
                )
                # Truncate completions_res to same length for comparison
                completions_res = completions_res[: len(generate_res)]

        assert generate_res == completions_res


@pytest.mark.asyncio
async def test_stop_string_workflow(client, tokenizer, messages):
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # default with Qwen3
        return_dict=True,  # default with Transformers v5
    ).input_ids
    payload = {
        "model": MODEL_NAME,
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": 24,
            "temperature": 0.0,
            "detokenize": False,
            # stop strings are only supported when detokenize is True.
            "stop": ["27 member"],
        },
        # TODO stream test is much more interesting
        "stream": False,
    }
    with pytest.raises(httpx.HTTPStatusError):
        generate_resp = await client.post(GEN_ENDPOINT, json=payload)
        generate_resp.raise_for_status()

    payload["sampling_params"]["stop"] = None
    generate_resp = await client.post(
        GEN_ENDPOINT, json=payload, headers={"X-Request-Id": "42"}
    )
    generate_data = generate_resp.json()
    generate_res = tokenizer.decode(
        generate_data["choices"][0]["token_ids"], skip_special_tokens=True
    )

    # NOTE This is under the responsibility of the coordinator
    # stop_checker = StopChecker(
    #     max_model_len=1024, get_tokenizer_for_seq=lambda _: tokenizer
    # )
    stop_str, truncate_to = check_stop_strings(
        generate_res, len(generate_res), ["27 member"], False
    )
    assert stop_str == "27 member"
    # abort request that hit stop string (requires tokens-only mode)
    # res = await client.post("/abort_requests", json={"request_ids": ["generate-tokens-42"]}) # noqa: E501
    # res.raise_for_status()
    generate_res = generate_res[:truncate_to]

    # Get stop_str response from chat completions
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
        "stop": ["27 member"],
        "chat_template_kwargs": dict(enable_thinking=False),
    }
    completions_resp = await client.post("/v1/chat/completions", json=payload)
    completions_data = completions_resp.json()
    completions_res = completions_data["choices"][0]["message"]["content"]
    assert generate_res == completions_res


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        [
            "--enable-lora",
            "--lora-modules",
            "Alice=charent/self_cognition_Alice",
            "Bob=charent/self_cognition_Bob",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
        ]
    ],
    indirect=True,
)
async def test_generate_with_lora_adapter(client, tokenizer, messages):
    # Verify adapters are listed
    models_resp = await client.get("/v1/models")
    models_resp.raise_for_status()
    models = {m["id"] for m in models_resp.json().get("data", [])}
    assert {"Alice", "Bob"}.issubset(models)

    # Generate using a LoRA adapter by specifying its name as the model
    payload = {
        "model": "Alice",
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 5},
        "stream": False,
    }
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assert "choices" in data

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # default with Qwen3
        return_dict=True,  # default with Transformers v5
    ).input_ids
    payload = {
        "model": "Alice",
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": 24,
            "temperature": 0.0,
            "detokenize": False,
        },
        "stream": False,
    }
    generate_resp = await client.post(GEN_ENDPOINT, json=payload)
    generate_data = generate_resp.json()
    generate_res = tokenizer.decode(
        generate_data["choices"][0]["token_ids"], skip_special_tokens=True
    )

    payload = {
        "model": "Alice",
        "messages": messages,
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": dict(enable_thinking=False),
    }
    completions_resp = await client.post("/v1/chat/completions", json=payload)
    completions_data = completions_resp.json()
    completions_res = completions_data["choices"][0]["message"]["content"]

    assert generate_res == completions_res
