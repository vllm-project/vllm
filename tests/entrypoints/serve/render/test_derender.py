# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /derender endpoints (postprocessing counterpart to /render)."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer

# ---------------------------------------------------------------------------
# Unit tests for build_chat_message (no server required)
# ---------------------------------------------------------------------------


def _make_chat_request(
    tool_choice=None,
    tools=None,
    include_reasoning: bool = True,
):
    """Create a minimal ChatCompletionRequest for unit testing."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    kwargs = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "include_reasoning": include_reasoning,
    }
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if tools is not None:
        kwargs["tools"] = tools
    return ChatCompletionRequest(**kwargs)


def _make_mock_tokenizer():
    """Return a non-Mistral mock tokenizer."""
    from unittest.mock import MagicMock

    return MagicMock()


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_build_chat_message_no_parser_passthrough():
    """Without a parser, output_text passes straight through as content."""
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    req = _make_chat_request()
    msg, auto, cnt = build_chat_message(
        output_text="hello world",
        output_token_ids=[1, 2, 3],
        request=req,
        parser=None,
        tool_parser=None,
        use_harmony=False,
        enable_auto_tools=False,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    assert msg.content == "hello world"
    assert msg.role == "assistant"
    assert msg.reasoning is None
    assert auto is False
    assert cnt == 0


def test_build_chat_message_parser_include_reasoning():
    """Parser output reasoning is preserved when include_reasoning=True."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    parser = MagicMock()
    parser.parse.return_value = ("I am thinking...", "final answer", [])

    req = _make_chat_request(include_reasoning=True)
    msg, _, _ = build_chat_message(
        output_text="<think>I am thinking...</think>final answer",
        output_token_ids=[1, 2, 3],
        request=req,
        parser=parser,
        tool_parser=None,
        use_harmony=False,
        enable_auto_tools=False,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    assert msg.content == "final answer"
    assert msg.reasoning == "I am thinking..."


def test_build_chat_message_parser_exclude_reasoning():
    """Parser output reasoning is stripped when include_reasoning=False."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    parser = MagicMock()
    parser.parse.return_value = ("I am thinking...", "final answer", [])

    req = _make_chat_request(include_reasoning=False)
    msg, _, _ = build_chat_message(
        output_text="<think>I am thinking...</think>final answer",
        output_token_ids=[1, 2, 3],
        request=req,
        parser=parser,
        tool_parser=None,
        use_harmony=False,
        enable_auto_tools=False,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    assert msg.content == "final answer"
    assert msg.reasoning is None


def test_build_chat_message_auto_tool_calls():
    """Auto tool choice with matching parser output populates tool_calls."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.engine.protocol import FunctionCall
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    func_call = FunctionCall(name="get_weather", arguments='{"city": "London"}')
    parser = MagicMock()
    parser.parse.return_value = (None, None, [func_call])
    tool_parser = MagicMock()  # truthy but not a Mistral parser

    req = _make_chat_request(tool_choice="auto", tools=_TOOLS)
    msg, auto, cnt = build_chat_message(
        output_text='get_weather({"city": "London"})',
        output_token_ids=[1, 2, 3],
        request=req,
        parser=parser,
        tool_parser=tool_parser,
        use_harmony=False,
        enable_auto_tools=True,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    assert auto is True
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "get_weather"
    assert cnt == 1  # history counter incremented


def test_build_chat_message_required_tool_calls():
    """Required tool_choice produces tool_calls. auto_tools_called stays False."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.engine.protocol import FunctionCall
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    func_call = FunctionCall(name="search", arguments='{"q": "vllm"}')
    parser = MagicMock()
    parser.parse.return_value = (None, "", [func_call])
    tool_parser = MagicMock()

    req = _make_chat_request(tool_choice="required", tools=_TOOLS)
    msg, auto, cnt = build_chat_message(
        output_text='search({"q": "vllm"})',
        output_token_ids=[1, 2],
        request=req,
        parser=parser,
        tool_parser=tool_parser,
        use_harmony=False,
        enable_auto_tools=True,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    # auto_tools_called is False for "required". The finish_reason is set by
    # the caller using the `required + finish_reason=="stop"` rule.
    assert auto is False
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "search"
    assert cnt == 1


def test_build_chat_message_history_cnt_threads_across_calls():
    """history_tool_call_cnt is threaded correctly across multiple outputs."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.engine.protocol import FunctionCall
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    func_a = FunctionCall(name="fn_a", arguments="{}")
    func_b = FunctionCall(name="fn_b", arguments="{}")
    parser = MagicMock()
    tool_parser = MagicMock()

    req = _make_chat_request(tool_choice="auto", tools=_TOOLS)
    tokenizer = _make_mock_tokenizer()

    parser.parse.return_value = (None, None, [func_a])
    _, _, cnt1 = build_chat_message(
        output_text="fn_a({})",
        output_token_ids=[1],
        request=req,
        parser=parser,
        tool_parser=tool_parser,
        use_harmony=False,
        enable_auto_tools=True,
        tokenizer=tokenizer,
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    assert cnt1 == 1

    parser.parse.return_value = (None, None, [func_b])
    _, _, cnt2 = build_chat_message(
        output_text="fn_b({})",
        output_token_ids=[2],
        request=req,
        parser=parser,
        tool_parser=tool_parser,
        use_harmony=False,
        enable_auto_tools=True,
        tokenizer=tokenizer,
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=cnt1,
    )
    assert cnt2 == 2


# ---------------------------------------------------------------------------
# Parity tests: JSON round trip of ChatCompletionRequest must not affect output
# ---------------------------------------------------------------------------


def test_parity_json_roundtrip_reasoning():
    """chat path (in process request) vs derender path (JSON round tripped request)
    produce identical build_chat_message output for a reasoning scenario."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    parser = MagicMock()
    parser.parse.return_value = ("The reasoning trace.", "The final answer.", [])

    req_live = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "What is 6 × 7?"}],
        include_reasoning=True,
    )
    # Derender path: ChatCompletionRequest arrives deserialized from JSON
    req_derender = ChatCompletionRequest.model_validate_json(req_live.model_dump_json())

    common_kwargs: dict = dict(
        output_text="<think>The reasoning trace.</think>The final answer.",
        output_token_ids=[1, 2, 3],
        parser=parser,
        tool_parser=None,
        use_harmony=False,
        enable_auto_tools=False,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    msg_live, auto_live, cnt_live = build_chat_message(
        request=req_live, **common_kwargs
    )
    msg_derender, auto_derender, cnt_derender = build_chat_message(
        request=req_derender, **common_kwargs
    )

    assert msg_live.content == msg_derender.content
    assert msg_live.reasoning == msg_derender.reasoning
    assert msg_live.tool_calls == msg_derender.tool_calls
    assert auto_live == auto_derender
    assert cnt_live == cnt_derender


def test_parity_json_roundtrip_tool_calls():
    """chat path vs derender path produce identical output for auto tool calls."""
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import FunctionCall
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message

    func_call = FunctionCall(name="get_weather", arguments='{"city": "London"}')
    parser = MagicMock()
    parser.parse.return_value = (None, None, [func_call])
    tool_parser = MagicMock()

    req_live = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Weather in London?"}],
        tools=_TOOLS,
        tool_choice="auto",
    )
    req_derender = ChatCompletionRequest.model_validate_json(req_live.model_dump_json())

    common_kwargs = dict(
        output_text='<tool_call>{"name": "get_weather", "arguments": '
        '{"city": "London"}}</tool_call>',
        output_token_ids=[1, 2, 3],
        parser=parser,
        tool_parser=tool_parser,
        use_harmony=False,
        enable_auto_tools=True,
        tokenizer=_make_mock_tokenizer(),
        role="assistant",
        tool_call_id_type="random",
        history_tool_call_cnt=0,
    )
    msg_live, auto_live, cnt_live = build_chat_message(
        request=req_live, **common_kwargs
    )
    msg_derender, auto_derender, cnt_derender = build_chat_message(
        request=req_derender, **common_kwargs
    )

    assert auto_live == auto_derender
    assert cnt_live == cnt_derender
    live_calls = msg_live.tool_calls or []
    derender_calls = msg_derender.tool_calls or []
    assert len(live_calls) == len(derender_calls)
    if live_calls:
        assert live_calls[0].function.name == derender_calls[0].function.name


def test_grammar_from_tool_parser_not_preserved_by_json_but_rederived():
    """_grammar_from_tool_parser is a PrivateAttr excluded from JSON serialization.

    After round trip it is always False, but build_chat_message re-derives the
    Mistral grammar branch from is_mistral_tool_parser(tool_parser) +
    structured_outputs.grammar so both paths produce identical output.
    """
    from unittest.mock import MagicMock, patch

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.serve.utils.chat_message_builder import build_chat_message
    from vllm.sampling_params import StructuredOutputsParams

    # Simulate what MistralToolParser.adjust_request() does in-process:
    #   sets _grammar_from_tool_parser = True (PrivateAttr, not serialized)
    #   sets structured_outputs.grammar (regular field, survives round-trip)
    req_live = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "call a tool"}],
    )
    req_live._grammar_from_tool_parser = True
    req_live.structured_outputs = StructuredOutputsParams(grammar="lark-grammar-string")

    # Verify the round trip behaviour that motivated the fix
    req_derender = ChatCompletionRequest.model_validate_json(req_live.model_dump_json())
    assert req_derender._grammar_from_tool_parser is False, (
        "_grammar_from_tool_parser must be False after JSON round-trip (PrivateAttr)"
    )
    assert req_derender.structured_outputs is not None
    assert req_derender.structured_outputs.grammar == "lark-grammar-string", (
        "structured_outputs.grammar must survive JSON round-trip"
    )

    # build_chat_message must take the Mistral grammar branch for BOTH requests
    parser = MagicMock()
    parser.parse.return_value = (None, "content", [])
    mock_tool_parser = MagicMock()
    fake_tool_calls = [MagicMock()]

    with (
        patch(
            "vllm.entrypoints.serve.utils.chat_message_builder.is_mistral_tool_parser",
            return_value=True,
        ),
        patch(
            "vllm.tool_parsers.mistral_tool_parser.MistralToolParser"
            ".build_non_streaming_tool_calls",
            return_value=fake_tool_calls,
        ),
    ):
        kwargs: dict = dict(
            output_text="content",
            output_token_ids=[10, 20, 30],
            parser=parser,
            tool_parser=mock_tool_parser,
            use_harmony=False,
            enable_auto_tools=True,
            tokenizer=_make_mock_tokenizer(),
            role="assistant",
            tool_call_id_type="random",
            history_tool_call_cnt=0,
        )
        msg_live, auto_live, _ = build_chat_message(request=req_live, **kwargs)
        msg_derender, auto_derender, _ = build_chat_message(
            request=req_derender, **kwargs
        )

    assert msg_live.tool_calls == msg_derender.tool_calls == fake_tool_calls
    assert msg_live.content == msg_derender.content
    assert auto_live == auto_derender


MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _render_chat(client: httpx.AsyncClient) -> dict:
    """Render a minimal chat request and return the GenerateRequest dict."""
    resp = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    return resp.json()


def _make_generate_response(
    token_ids: list[int] | None,
    request_id: str = "chatcmpl-test-id",
    finish_reason: str = "stop",
    logprobs: dict | None = None,
    prompt_logprobs: list | None = None,
    kv_transfer_params: dict | None = None,
) -> dict:
    choice: dict = {
        "index": 0,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
    }
    return {
        "request_id": request_id,
        "choices": [choice],
        "prompt_logprobs": prompt_logprobs,
        "kv_transfer_params": kv_transfer_params,
    }


def _make_logprobs_with_placeholders(token_id: int = 1234) -> dict:
    entry = {
        "token": f"token_id:{token_id}",
        "logprob": -1.0,
        "bytes": None,
        "top_logprobs": [
            {"token": f"token_id:{token_id + 1}", "logprob": -2.0, "bytes": None}
        ],
    }
    return {"content": [entry]}


# ---------------------------------------------------------------------------
# Chat derender tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derender_chat_roundtrip(client):
    """Render then derender: decoded content should be a non-empty string."""
    gen_req = await _render_chat(client)
    # Use the first 5 rendered token IDs as synthetic "generated" tokens.
    synthetic_ids = gen_req["token_ids"][:5]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_derender_chat_usage(client):
    """Supplied prompt_tokens flows through into usage correctly."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
            "prompt_tokens": 10,
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == len(synthetic_ids)
    assert usage["total_tokens"] == 10 + len(synthetic_ids)


@pytest.mark.asyncio
async def test_derender_chat_usage_default(client):
    """Omitting prompt_tokens gives usage.prompt_tokens == 0."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_derender_chat_logprobs(client):
    """token_id:N placeholders in content.token are resolved to real strings."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    data = response.json()
    logprobs = data["choices"][0]["logprobs"]
    assert logprobs is not None
    content = logprobs["content"]
    assert content is not None and len(content) == 1
    token_str = content[0]["token"]
    assert not token_str.startswith("token_id:"), (
        f"Placeholder was not resolved: {token_str!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_logprobs_bytes(client):
    """Resolved logprob entries have bytes populated as list[int]."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    bytes_field = content[0]["bytes"]
    assert isinstance(bytes_field, list)
    assert len(bytes_field) > 0
    assert all(isinstance(b, int) for b in bytes_field)


@pytest.mark.asyncio
async def test_derender_chat_top_logprobs(client):
    """top_logprobs entries also have their placeholders resolved."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    top = content[0]["top_logprobs"]
    assert len(top) == 1
    assert not top[0]["token"].startswith("token_id:"), (
        f"top_logprobs placeholder not resolved: {top[0]['token']!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_prompt_logprobs_passthrough(client):
    """prompt_logprobs on GenerateResponse passes through unchanged."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    # prompt_logprobs is a list[dict[int, Logprob] | None]; use None entries.
    prompt_logprobs = [None, None]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, prompt_logprobs=prompt_logprobs
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["prompt_logprobs"] == prompt_logprobs


@pytest.mark.asyncio
async def test_derender_chat_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to the ChatCompletionResponse."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    kv = {"key": "value"}

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, kv_transfer_params=kv
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv


@pytest.mark.asyncio
async def test_derender_chat_empty_token_ids(client):
    """Empty token_ids list returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response([]),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_null_token_ids(client):
    """Null token_ids returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(None),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_with_chat_request_no_parser(client):
    """Providing chat_request when server has no parser degrades gracefully:
    content == Phase-1 decoded text and role is 'assistant'."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:5]
    chat_request = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    r_phase1 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert r_phase1.status_code == 200

    r_phase2 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
            "chat_request": chat_request,
        },
    )
    assert r_phase2.status_code == 200

    msg1 = r_phase1.json()["choices"][0]["message"]
    msg2 = r_phase2.json()["choices"][0]["message"]
    assert msg2["content"] == msg1["content"]
    assert msg2["role"] == "assistant"


@pytest.mark.asyncio
async def test_derender_chat_unknown_model(client):
    """Unknown model returns 404."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": "does-not-exist",
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Completion derender tests
# ---------------------------------------------------------------------------


async def _render_completion(client: httpx.AsyncClient, prompt: str) -> dict:
    """Render a completion prompt and return the first GenerateRequest dict."""
    resp = await client.post(
        "/v1/completions/render",
        json={"model": MODEL_NAME, "prompt": prompt},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) >= 1
    return data[0]


def _make_completion_generate_response(
    token_ids: list[int],
    request_id: str,
    kv_transfer_params: dict | None = None,
    logprobs: dict | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": "stop",
                "logprobs": logprobs,
            }
        ],
        "prompt_logprobs": None,
        "kv_transfer_params": kv_transfer_params,
    }


@pytest.mark.asyncio
async def test_derender_completion_roundtrip(client):
    """Two prompts rendered, two GenerateResponses → two choices with indices 0, 1."""
    gr1 = await _render_completion(client, "Hello world")
    gr2 = await _render_completion(client, "Goodbye world")

    ids1 = gr1["token_ids"][:4]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    choices = data["choices"]
    assert len(choices) == 2
    assert choices[0]["index"] == 0
    assert choices[1]["index"] == 1
    assert choices[0]["text"]
    assert choices[1]["text"]


@pytest.mark.asyncio
async def test_derender_completion_usage_aggregation(client):
    """prompt_tokens=[5, 10] is aggregated correctly into usage."""
    gr1 = await _render_completion(client, "Hello")
    gr2 = await _render_completion(client, "World")

    ids1 = gr1["token_ids"][:3]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == len(ids1) + len(ids2)
    assert usage["total_tokens"] == 15 + len(ids1) + len(ids2)


@pytest.mark.asyncio
async def test_derender_completion_prompt_tokens_length_mismatch(client):
    """len(prompt_tokens) != len(generate_responses) returns 400."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_empty_generate_responses(client):
    """Empty generate_responses list returns 400."""
    response = await client.post(
        "/v1/completions/derender",
        json={"model": MODEL_NAME, "generate_responses": []},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_logprobs(client):
    """token_id:N placeholders in logprobs are resolved; CompletionLogProbs
    flat-list structure is returned with non-empty tokens and text_offsets."""
    gr1 = await _render_completion(client, "Hello world")
    ids1 = gr1["token_ids"][:3]
    token_id = ids1[0]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1,
                    gr1["request_id"],
                    logprobs=_make_logprobs_with_placeholders(token_id),
                ),
            ],
        },
    )
    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs is not None
    tokens = logprobs["tokens"]
    assert len(tokens) == 1
    assert not tokens[0].startswith("token_id:"), (
        f"Placeholder was not resolved: {tokens[0]!r}"
    )
    assert len(logprobs["token_logprobs"]) == 1
    assert isinstance(logprobs["token_logprobs"][0], float)
    assert len(logprobs["text_offset"]) == 1
    assert logprobs["text_offset"][0] == 0


@pytest.mark.asyncio
async def test_derender_completion_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to CompletionResponse."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]
    kv = {"node": "abc"}

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1, gr1["request_id"], kv_transfer_params=kv
                ),
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv
