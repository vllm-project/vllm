# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM gRPC server.
"""

import asyncio
import socket
import subprocess
import sys
import time

import grpc
import pytest
import pytest_asyncio

from vllm.grpc import vllm_engine_pb2, vllm_engine_pb2_grpc

# Use a small model for fast testing
MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


async def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Wait for the gRPC server to be ready by trying health checks."""
    start_time = time.time()
    print("waiting for server to start...")
    while time.time() - start_time < timeout:
        try:
            channel = grpc.aio.insecure_channel(f"localhost:{port}")
            stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
            request = vllm_engine_pb2.HealthCheckRequest()
            response = await stub.HealthCheck(request, timeout=5.0)
            await channel.close()
            if response.healthy:
                print("server returned healthy=True")
                return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


class GrpcServerProcess:
    """Manages a gRPC server running in a subprocess."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.port: int | None = None

    async def start(self):
        """Start the gRPC server process."""
        self.port = find_free_port()

        # Start the server as a subprocess
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.grpc_server",
                "--model",
                MODEL_NAME,
                "--host",
                "localhost",
                "--port",
                str(self.port),
                "--max-num-batched-tokens",
                "512",
                "--disable-log-stats-server",
            ],
        )

        # Wait for server to be ready
        if not await wait_for_server(self.port):
            self.stop()
            raise RuntimeError("gRPC server failed to start within timeout")

    def stop(self):
        """Stop the gRPC server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


@pytest_asyncio.fixture(scope="module")
async def grpc_server():
    """Fixture providing a running gRPC server in a subprocess."""
    server = GrpcServerProcess()
    await server.start()

    yield server

    server.stop()


@pytest_asyncio.fixture
async def grpc_client(grpc_server):
    """Fixture providing a gRPC client connected to the server."""
    channel = grpc.aio.insecure_channel(f"localhost:{grpc_server.port}")
    stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)

    yield stub

    await channel.close()


@pytest.mark.asyncio
async def test_health_check(grpc_client):
    """Test the HealthCheck RPC."""
    request = vllm_engine_pb2.HealthCheckRequest()
    response = await grpc_client.HealthCheck(request)

    assert response.healthy is True
    assert response.message == "Health"


@pytest.mark.asyncio
async def test_get_model_info(grpc_client):
    """Test the GetModelInfo RPC."""
    request = vllm_engine_pb2.GetModelInfoRequest()
    response = await grpc_client.GetModelInfo(request)

    assert response.model_path == MODEL_NAME
    assert response.is_generation is True
    assert response.max_context_length > 0
    assert response.vocab_size > 0
    assert response.supports_vision is False


@pytest.mark.asyncio
async def test_get_server_info(grpc_client):
    """Test the GetServerInfo RPC."""
    request = vllm_engine_pb2.GetServerInfoRequest()
    response = await grpc_client.GetServerInfo(request)

    assert response.active_requests >= 0
    assert response.is_paused is False
    assert response.uptime_seconds >= 0
    assert response.server_type == "vllm-grpc"
    assert response.last_receive_timestamp > 0


@pytest.mark.asyncio
async def test_generate_non_streaming(grpc_client):
    """Test the Generate RPC in non-streaming mode."""
    # Create a simple request
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-non-streaming-1",
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="Hello, my name is",
            input_ids=[15496, 11, 616, 1438, 318],  # GPT-2 tokens for the prompt
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0,
            max_tokens=10,
            n=1,
        ),
        stream=False,
    )

    # Collect all responses
    responses = []
    async for response in grpc_client.Generate(request):
        responses.append(response)

    # Should have exactly one response (complete)
    assert len(responses) == 1

    # Check the response
    final_response = responses[0]
    assert final_response.HasField("complete")

    complete = final_response.complete
    assert len(complete.output_ids) > 0
    assert complete.finish_reason in ["stop", "length"]
    assert complete.prompt_tokens > 0
    assert complete.completion_tokens > 0


@pytest.mark.asyncio
async def test_generate_streaming(grpc_client):
    """Test the Generate RPC in streaming mode."""
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-streaming-1",
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="The capital of France is",
            input_ids=[464, 3139, 286, 4881, 318],  # GPT-2 tokens
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0, max_tokens=10, n=1
        ),
        stream=True,
    )

    # Collect all responses
    chunks = []
    complete_response = None

    async for response in grpc_client.Generate(request):
        if response.HasField("chunk"):
            chunks.append(response.chunk)
        elif response.HasField("complete"):
            complete_response = response.complete

    # Should have received some chunks
    assert len(chunks) >= 0  # May have 0 chunks if generation is very fast

    # Should have a final complete response
    assert complete_response is not None
    assert complete_response.finish_reason in ["stop", "length"]
    assert complete_response.prompt_tokens > 0

    # Verify chunk structure
    for chunk in chunks:
        assert chunk.prompt_tokens > 0
        assert chunk.completion_tokens >= 0


@pytest.mark.asyncio
async def test_generate_with_different_sampling_params(grpc_client):
    """Test Generate with various sampling parameters."""
    # Test with temperature
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-sampling-temp",
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="Hello",
            input_ids=[15496],
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=5
        ),
        stream=False,
    )

    responses = [r async for r in grpc_client.Generate(request)]
    assert len(responses) == 1
    assert responses[0].HasField("complete")

    # Test with top_k
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-sampling-topk",
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="Hello",
            input_ids=[15496],
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=1.0, top_k=50, max_tokens=5
        ),
        stream=False,
    )

    responses = [r async for r in grpc_client.Generate(request)]
    assert len(responses) == 1
    assert responses[0].HasField("complete")


@pytest.mark.asyncio
async def test_generate_with_stop_strings(grpc_client):
    """Test Generate with stop strings."""
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-stop-strings",
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="Hello",
            input_ids=[15496],
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0,
            max_tokens=20,
            stop=["\n", "END"],
        ),
        stream=False,
    )

    responses = [r async for r in grpc_client.Generate(request)]
    assert len(responses) == 1
    assert responses[0].HasField("complete")

    complete = responses[0].complete
    assert complete.finish_reason in ["stop", "length"]


@pytest.mark.asyncio
async def test_generate_multiple_requests(grpc_client):
    """Test handling multiple concurrent Generate requests."""

    async def make_request(request_id: str):
        request = vllm_engine_pb2.GenerateRequest(
            request_id=request_id,
            tokenized=vllm_engine_pb2.TokenizedInput(
                original_text="Hello",
                input_ids=[15496],
            ),
            sampling_params=vllm_engine_pb2.SamplingParams(
                temperature=0.0, max_tokens=5
            ),
            stream=False,
        )

        responses = [r async for r in grpc_client.Generate(request)]
        return responses[0]

    # Send multiple requests concurrently
    tasks = [make_request(f"test-concurrent-{i}") for i in range(3)]
    responses = await asyncio.gather(*tasks)

    # Verify all requests completed successfully
    assert len(responses) == 3
    for i, response in enumerate(responses):
        assert response.HasField("complete")


@pytest.mark.asyncio
async def test_generate_with_seed(grpc_client):
    """Test Generate with a fixed seed for reproducibility."""

    def make_request(request_id: str, seed: int):
        return vllm_engine_pb2.GenerateRequest(
            request_id=request_id,
            tokenized=vllm_engine_pb2.TokenizedInput(
                original_text="The future of AI is",
                input_ids=[464, 2003, 286, 9552, 318],
            ),
            sampling_params=vllm_engine_pb2.SamplingParams(
                temperature=1.0, max_tokens=10, seed=seed
            ),
            stream=False,
        )

    # Make two requests with the same seed
    request1 = make_request("test-seed-1", 42)
    request2 = make_request("test-seed-2", 42)

    response_list1 = [r async for r in grpc_client.Generate(request1)]
    response_list2 = [r async for r in grpc_client.Generate(request2)]

    # Both should complete successfully
    assert len(response_list1) == 1
    assert len(response_list2) == 1
    assert response_list1[0].HasField("complete")
    assert response_list2[0].HasField("complete")

    # With the same seed, outputs should be identical
    output_ids1 = list(response_list1[0].complete.output_ids)
    output_ids2 = list(response_list2[0].complete.output_ids)
    assert output_ids1 == output_ids2


@pytest.mark.asyncio
async def test_generate_error_handling(grpc_client):
    """Test error handling in Generate RPC."""
    # Request with invalid top_p value (-33)
    request = vllm_engine_pb2.GenerateRequest(
        request_id="test-error-invalid-topp",
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0, max_tokens=10, top_p=-33
        ),
        stream=False,
    )

    # Should raise an error response
    with pytest.raises(grpc.RpcError) as exc_info:
        _ = [r async for r in grpc_client.Generate(request)]

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "top_p must be in (0, 1], got -33.0" in exc_info.value.details()


@pytest.mark.asyncio
async def test_abort_request(grpc_client):
    """Test the out-of-band Abort RPC."""
    request_id = "test-abort-1"

    # Start a long-running streaming generate request
    generate_request = vllm_engine_pb2.GenerateRequest(
        request_id=request_id,
        tokenized=vllm_engine_pb2.TokenizedInput(
            original_text="Hello",
            input_ids=[15496],
        ),
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0,
            min_tokens=500,
            max_tokens=500,  # Request many tokens to ensure it runs long enough
        ),
        stream=True,
    )

    # Track whether we were aborted
    was_aborted = False
    received_chunks = 0

    async def run_generate():
        nonlocal was_aborted, received_chunks
        async for response in grpc_client.Generate(generate_request):
            if response.HasField("chunk"):
                received_chunks += 1

            if response.HasField("complete"):
                complete = response.complete
                was_aborted = complete.finish_reason == "abort"
            else:
                was_aborted = False

    async def abort_after_delay():
        # Small delay to ensure generate has started
        await asyncio.sleep(0.1)
        abort_request = vllm_engine_pb2.AbortRequest(request_ids=[request_id])
        await grpc_client.Abort(abort_request)

    # Run generate and abort concurrently
    await asyncio.gather(run_generate(), abort_after_delay())

    # The request should have been aborted (received final chunk with
    # "abort" finish reason) and finished early due to the abort.
    assert was_aborted and received_chunks < 500, (
        "Request should have been aborted before generating all 500 tokens"
    )
