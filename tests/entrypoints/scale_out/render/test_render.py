# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /render endpoints that expose prompt preprocessing."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    args: list[str] = ["--trust-request-chat-template"]

    with RemoteLaunchRenderServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_completion_render_basic(client):
    """Test basic completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "When should a chat-completions handler return an empty string?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - list of GenerateRequest
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify first prompt is a GenerateRequest
    first_prompt = data[0]
    assert "token_ids" in first_prompt
    assert "sampling_params" in first_prompt
    assert "model" in first_prompt
    assert "request_id" in first_prompt
    assert isinstance(first_prompt["token_ids"], list)
    assert len(first_prompt["token_ids"]) > 0
    assert first_prompt["model"] == MODEL_NAME
    assert first_prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_chat_completion_render_basic(client):
    """Test basic chat completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Returning an empty string for the prompt may be confusing."
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - should be a GenerateRequest
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0

    # Verify token IDs are integers and BOS token is present
    token_ids = data["token_ids"]
    assert all(isinstance(tid, int) for tid in token_ids)
    assert token_ids[0] == 1


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts(client):
    """Test completion render with multiple prompts."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return two GenerateRequest items
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify both prompts have GenerateRequest fields
    for prompt in data:
        assert "token_ids" in prompt
        assert "sampling_params" in prompt
        assert "model" in prompt
        assert "request_id" in prompt
        assert len(prompt["token_ids"]) > 0
        assert prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_chat_completion_render_multi_turn(client):
    """Test chat completion render with multi-turn conversation."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify tokenization occurred
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_render_with_stream_true(client):
    """Render accepts stream params but still returns JSON (non-streamed)."""

    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Stream options should be accepted by /render.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0

    # /render should preserve stream fields on the returned token-in request.
    assert data.get("stream") is True
    assert isinstance(data.get("stream_options"), dict)
    assert data["stream_options"].get("include_usage") is True
    assert data["stream_options"].get("continuous_usage_stats") is True


@pytest.mark.asyncio
async def test_completion_render_error_invalid_model(client):
    """Test completion render with invalid model returns error."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": "invalid-model-name",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completion_render_error_invalid_model(client):
    """Test chat completion render with invalid model returns error."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": "invalid-model-name",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_completion_render_no_generation(client):
    """Verify render endpoint does not generate text."""
    # This test verifies that calling render is fast (no generation)
    import time

    start = time.perf_counter()
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Tell me a very long story about " * 10,
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    # Render should be fast (< 1 second) since no generation
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_chat_completion_render_with_sampling_params(client):
    """Verify sampling params are correctly returned by /render."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Test sampling params"}],
            "temperature": 0.123,
            "top_p": 0.456,
            "frequency_penalty": 1.1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "sampling_params" in data
    sampling_params = data["sampling_params"]

    assert sampling_params.get("temperature") == 0.123
    assert sampling_params.get("top_p") == 0.456
    assert sampling_params.get("frequency_penalty") == 1.1

    # Check that internal fields are not present
    assert "_all_stop_token_ids" not in sampling_params


@pytest.mark.asyncio
async def test_completion_render_emits_token_offsets(client):
    """With return_token_offsets, /v1/completions/render returns per-token
    (start, end) char offsets aligned with token_ids."""
    prompt = "Hello, world."
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    offsets = data[0]["token_offsets"]
    assert offsets is not None
    assert len(offsets) == len(data[0]["token_ids"])
    for start, end in offsets:
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start <= end <= len(prompt)


@pytest.mark.asyncio
async def test_completion_render_default_no_token_offsets(client):
    """Without the flag, token_offsets must be null (existing responses
    unchanged)."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Hello, world.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data[0]["token_offsets"] is None


@pytest.mark.asyncio
async def test_chat_render_emits_token_offsets(client):
    """With return_token_offsets, /v1/chat/completions/render returns
    per-token offsets relative to the templated prompt string."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, world."}],
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    offsets = data["token_offsets"]
    assert offsets is not None
    assert len(offsets) == len(data["token_ids"])
    for start, end in offsets:
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start <= end


@pytest.mark.asyncio
async def test_chat_render_default_no_token_offsets(client):
    """Without the flag, chat render token_offsets must be null."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, world."}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["token_offsets"] is None


@pytest.mark.asyncio
async def test_completion_render_truncated_token_offsets(client):
    """Truncation must shorten token_offsets together with token_ids.

    An explicit truncation_side turns off tokenizer-level truncation, so the
    tokenizer returns offsets for the whole prompt and they are reduced
    afterwards -- separately from token_ids. GenerateRequest documents that the
    two lists have equal length.
    """
    prompt = "The quick brown fox jumps over the lazy dog."
    keep = 4
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "return_token_offsets": True,
            "truncate_prompt_tokens": keep,
            "truncation_side": "left",
        },
    )

    assert response.status_code == 200
    data = response.json()
    token_ids = data[0]["token_ids"]
    offsets = data[0]["token_offsets"]

    assert len(token_ids) == keep
    assert len(offsets) == len(token_ids)

    # Equal length is not enough: truncating from the left keeps the *last*
    # tokens, so the surviving offsets must cover the end of the prompt.
    assert offsets[-1][1] == len(prompt)
    assert offsets[0][0] > 0


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts_token_offsets(client):
    """Each prompt in a batch gets its own offsets aligned with its tokens."""
    prompts = ["Hello, world.", "Goodbye, world."]
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompts,
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(prompts)
    for item, prompt in zip(data, prompts):
        offsets = item["token_offsets"]
        assert offsets is not None
        assert len(offsets) == len(item["token_ids"])
        for start, end in offsets:
            assert 0 <= start <= end <= len(prompt)


@pytest.mark.asyncio
async def test_chat_completion_render_assistant_tokens_mask_default(client):
    """Without return_assistant_tokens_mask, assistant_tokens_mask should be null."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("assistant_tokens_mask") is None


@pytest.mark.asyncio
async def test_chat_completion_render_assistant_tokens_mask_false(client):
    """Explicitly setting return_assistant_tokens_mask=false gives null."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "return_assistant_tokens_mask": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("assistant_tokens_mask") is None


@pytest.mark.asyncio
async def test_chat_render_assistant_tokens_mask_null_without_gen_tags(
    client,
):
    """The tiny test model lacks ``{% generation %}`` tags, so the mask is null."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "return_assistant_tokens_mask": True,
        },
    )

    assert response.status_code == 200
    assert response.json().get("assistant_tokens_mask") is None


# A minimal chat template with {% generation %} tags so we can test that
# the mask correctly marks assistant tokens.
_TEMPLATE_WITH_GENERATION = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n"
    "{% elif m['role'] == 'assistant' %}"
    "{% generation %}Assistant: {{ m['content'] }}\n{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
)


@pytest.mark.asyncio
async def test_chat_completion_render_assistant_tokens_mask_with_generation_tags(
    client,
):
    """With a ``{% generation %}``-enabled template, the mask marks assistant
    tokens and the masked tokens decode to the assistant content."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Bye"},
            ],
            "chat_template": _TEMPLATE_WITH_GENERATION,
            "return_assistant_tokens_mask": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    mask = data["assistant_tokens_mask"]
    token_ids = data["token_ids"]
    assert mask is not None
    assert isinstance(mask, list)
    assert len(mask) == len(token_ids)
    assert all(v in (0, 1) for v in mask)
    assert sum(mask) > 0, "mask should mark at least one assistant token"

    # Detokenize masked (assistant) and unmasked (non-assistant) tokens
    # separately to verify the mask is correct, not just non-empty.
    masked_ids = [t for t, m in zip(token_ids, mask, strict=True) if m]
    unmasked_ids = [t for t, m in zip(token_ids, mask, strict=True) if not m]

    detok = await client.post(
        "/detokenize",
        json={"model": MODEL_NAME, "tokens": masked_ids},
    )
    assert detok.status_code == 200
    assert "Hi!" in detok.json()["prompt"]

    detok_rest = await client.post(
        "/detokenize",
        json={"model": MODEL_NAME, "tokens": unmasked_ids},
    )
    assert detok_rest.status_code == 200
    assert "Hi!" not in detok_rest.json()["prompt"]
    assert "Bye" in detok_rest.json()["prompt"]


@pytest.mark.asyncio
async def test_chat_render_assistant_tokens_mask_follows_truncation(client):
    """The assistant mask must be truncated with the prompt it describes.

    `assistant_tokens_mask` is positional: entry i labels token i. Truncating
    `token_ids` from the left without truncating the mask leaves the two
    describing different positions, and the mask ends up marking whichever
    tokens happen to sit at the old offsets.
    """
    messages = [
        # Deliberately lopsided: a long leading user turn and a short trailing
        # one, so keeping the head of the mask is distinguishable from keeping
        # its tail.
        {"role": "user", "content": "Hello hello hello hello hello hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "Bye"},
    ]
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "chat_template": _TEMPLATE_WITH_GENERATION,
        "return_assistant_tokens_mask": True,
    }

    full = await client.post("/v1/chat/completions/render", json=body)
    assert full.status_code == 200
    full_token_ids = full.json()["token_ids"]
    full_mask = full.json()["assistant_tokens_mask"]
    assert sum(full_mask) > 0

    keep = len(full_token_ids) - 4
    # Precondition: with this prompt the head and tail slices of the mask
    # really do differ, so the assertion below can tell them apart.
    assert full_mask[-keep:] != full_mask[:keep]

    truncated = await client.post(
        "/v1/chat/completions/render",
        json={**body, "truncate_prompt_tokens": keep, "truncation_side": "left"},
    )
    assert truncated.status_code == 200
    data = truncated.json()

    assert data["token_ids"] == full_token_ids[-keep:]
    assert data["assistant_tokens_mask"] == full_mask[-keep:]


@pytest.mark.asyncio
async def test_messages_render_basic(client):
    """Test basic Anthropic Messages render endpoint."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Render this Anthropic message."}],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Single GenerateRequest, like chat render.
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert "sampling_params" in data
    assert "model" in data
    assert data["model"] == MODEL_NAME

    token_ids = data["token_ids"]
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(tid, int) for tid in token_ids)
    assert token_ids[0] == 1  # BOS


@pytest.mark.asyncio
async def test_messages_render_system_and_multi_turn(client):
    """System field + multi-turn messages render to a single prompt."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data["token_ids"]) > 0
    assert data["token_ids"][0] == 1  # BOS


@pytest.mark.asyncio
async def test_messages_render_merges_inline_system(client):
    """Inline system messages merge into the leading system block.

    Without a --chat-template arg the /v1/messages server path detects
    merge_inline_system=True, so render must produce the same tokens as
    the manually pre-merged request.
    """
    inline = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )
    assert inline.status_code == 200

    merged = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.Be brief.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )
    assert merged.status_code == 200

    assert inline.json()["token_ids"] == merged.json()["token_ids"]


@pytest.mark.asyncio
async def test_messages_render_error_invalid_model(client):
    """Messages render with an invalid model returns an error."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": "invalid-model-name",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
