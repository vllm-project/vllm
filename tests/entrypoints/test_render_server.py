# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM render server (gRPC and HTTP).
"""

import asyncio
import socket
import subprocess
import sys
import time

import grpc
import httpx
import pytest
import pytest_asyncio

from vllm.grpc import render_pb2, render_pb2_grpc

# Use a small model for fast testing
MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# =====================
# gRPC server fixtures & helpers
# =====================


async def wait_for_grpc(port: int, timeout: float = 120.0) -> bool:
    """Wait for the gRPC render server to be ready."""
    start_time = time.time()
    print("waiting for gRPC render server to start...")
    while time.time() - start_time < timeout:
        try:
            channel = grpc.aio.insecure_channel(f"localhost:{port}")
            stub = render_pb2_grpc.RenderServiceStub(channel)
            request = render_pb2.HealthCheckRequest()
            response = await stub.HealthCheck(request, timeout=5.0)
            await channel.close()
            if response.healthy:
                print("gRPC render server returned healthy=True")
                return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


@pytest_asyncio.fixture(scope="module")
async def grpc_server():
    """Fixture providing a running gRPC render server."""
    port = find_free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "render",
            "--model",
            MODEL_NAME,
            "--host",
            "localhost",
            "--port",
            str(port),
            "--server",
            "grpc",
        ],
    )

    if not await wait_for_grpc(port):
        process.terminate()
        process.wait()
        raise RuntimeError("gRPC render server failed to start")

    yield {"port": port, "process": process}

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest_asyncio.fixture
async def render_client(grpc_server):
    """Fixture providing a gRPC client connected to the render server."""
    channel = grpc.aio.insecure_channel(f"localhost:{grpc_server['port']}")
    stub = render_pb2_grpc.RenderServiceStub(channel)
    yield stub
    await channel.close()


# =====================
# HTTP server fixtures & helpers
# =====================


async def wait_for_http(port: int, timeout: float = 120.0) -> bool:
    """Wait for the HTTP render server to be ready."""
    start_time = time.time()
    print("waiting for HTTP render server to start...")
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/health", timeout=5.0
                )
                if response.status_code == 200:
                    print("HTTP render server returned healthy")
                    return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


@pytest_asyncio.fixture(scope="module")
async def http_server():
    """Fixture providing a running HTTP render server."""
    port = find_free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "render",
            "--model",
            MODEL_NAME,
            "--host",
            "localhost",
            "--port",
            str(port),
            "--server",
            "http",
        ],
    )

    if not await wait_for_http(port):
        process.terminate()
        process.wait()
        raise RuntimeError("HTTP render server failed to start")

    yield {"port": port, "process": process}

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest_asyncio.fixture
async def http_client(http_server):
    """Fixture providing an HTTP client connected to the render server."""
    async with httpx.AsyncClient(
        base_url=f"http://localhost:{http_server['port']}",
        timeout=30.0,
    ) as client:
        yield client


# =====================
# gRPC Tests
# =====================


@pytest.mark.asyncio
async def test_health_check(render_client):
    """Test the HealthCheck RPC."""
    request = render_pb2.HealthCheckRequest()
    response = await render_client.HealthCheck(request)

    assert response.healthy is True
    assert response.message == "OK"


@pytest.mark.asyncio
async def test_render_chat(render_client):
    """Test RenderChat with explicit chat_params and tok_params."""
    request = render_pb2.RenderChatRequest(
        messages=[
            render_pb2.ChatMessage(
                role="user",
                text_content="Hello, how are you?",
            ),
        ],
        chat_params=render_pb2.ChatParams(
            chat_template_content_format="auto",
        ),
        tok_params=render_pb2.TokenizeParams(
            add_special_tokens=False,
        ),
    )

    response = await render_client.RenderChat(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0
    assert response.num_tokens == len(response.prompt_token_ids)
    assert len(response.conversation) > 0
    assert any(msg.role == "user" for msg in response.conversation)


@pytest.mark.asyncio
async def test_render_chat_default_params(render_client):
    """Test RenderChat with no explicit params (server uses defaults)."""
    request = render_pb2.RenderChatRequest(
        messages=[
            render_pb2.ChatMessage(
                role="system",
                text_content="You are a helpful assistant.",
            ),
            render_pb2.ChatMessage(
                role="user",
                text_content="Hi!",
            ),
        ],
    )

    response = await render_client.RenderChat(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0
    assert len(response.conversation) >= 2


@pytest.mark.asyncio
async def test_render_chat_multipart_content(render_client):
    """Test RenderChat with multipart content."""
    request = render_pb2.RenderChatRequest(
        messages=[
            render_pb2.ChatMessage(
                role="user",
                parts=render_pb2.ContentPartList(
                    parts=[
                        render_pb2.ContentPart(
                            type="text",
                            text="What is in this image?",
                        ),
                    ]
                ),
            ),
        ],
    )

    response = await render_client.RenderChat(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0


@pytest.mark.asyncio
async def test_render_completion_text(render_client):
    """Test RenderCompletion with a text prompt."""
    request = render_pb2.RenderCompletionRequest(
        text_prompt="The quick brown fox",
    )

    response = await render_client.RenderCompletion(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0
    assert response.num_tokens == len(response.prompt_token_ids)


@pytest.mark.asyncio
async def test_render_completion_tokens(render_client):
    """Test RenderCompletion with token IDs (passthrough)."""
    input_token_ids = [1, 2, 3, 4, 5]

    request = render_pb2.RenderCompletionRequest(
        token_ids_prompt=render_pb2.TokenIds(token_ids=input_token_ids),
    )

    response = await render_client.RenderCompletion(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0


@pytest.mark.asyncio
async def test_render_completion_with_tok_params(render_client):
    """Test RenderCompletion with explicit TokenizeParams."""
    request = render_pb2.RenderCompletionRequest(
        text_prompt="Hello world",
        tok_params=render_pb2.TokenizeParams(
            add_special_tokens=True,
        ),
    )

    response = await render_client.RenderCompletion(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0


@pytest.mark.asyncio
async def test_render_chat_multi_turn(render_client):
    """Test RenderChat with a multi-turn conversation."""
    request = render_pb2.RenderChatRequest(
        messages=[
            render_pb2.ChatMessage(
                role="user",
                text_content="What is the weather?",
            ),
            render_pb2.ChatMessage(
                role="assistant",
                text_content="It is sunny today.",
            ),
            render_pb2.ChatMessage(
                role="user",
                text_content="Thanks!",
            ),
        ],
    )

    response = await render_client.RenderChat(request)

    assert len(response.prompt_token_ids) > 0
    assert response.num_tokens > 0
    assert len(response.conversation) == 3


# =====================
# HTTP Tests
# =====================


@pytest.mark.asyncio
async def test_http_health(http_client):
    """Test HTTP health endpoint."""
    response = await http_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is True
    assert data["message"] == "OK"


@pytest.mark.asyncio
async def test_http_render_chat(http_client):
    """Test HTTP render chat endpoint."""
    response = await http_client.post(
        "/render/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["prompt_token_ids"]) > 0
    assert data["num_tokens"] > 0
    assert data["num_tokens"] == len(data["prompt_token_ids"])
    assert len(data["conversation"]) > 0


@pytest.mark.asyncio
async def test_http_render_chat_with_params(http_client):
    """Test HTTP render chat with explicit params."""
    response = await http_client.post(
        "/render/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hi!"},
            ],
            "tok_params": {
                "add_special_tokens": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["prompt_token_ids"]) > 0
    assert data["num_tokens"] > 0


@pytest.mark.asyncio
async def test_http_render_completion(http_client):
    """Test HTTP render completion endpoint."""
    response = await http_client.post(
        "/render/completion",
        json={"prompt": "The quick brown fox"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["prompt_token_ids"]) > 0
    assert data["num_tokens"] > 0
    assert data["num_tokens"] == len(data["prompt_token_ids"])


@pytest.mark.asyncio
async def test_http_render_completion_with_tok_params(http_client):
    """Test HTTP render completion with explicit TokenizeParams."""
    response = await http_client.post(
        "/render/completion",
        json={
            "prompt": "Hello world",
            "tok_params": {
                "add_special_tokens": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["prompt_token_ids"]) > 0
    assert data["num_tokens"] > 0
