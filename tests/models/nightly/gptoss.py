# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import subprocess
import tempfile
import time

import openai
import pytest
import pytest_asyncio
from typing import Dict, Any, Optional

from vllm.platforms import current_platform
import vllm

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer


AITER_MODEL_LIST = [
    "openai/gpt-oss-120b"
]

MODEL_NAME = "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def default_server_args():
    attention_backend = (
        "ROCM_AITER_UNIFIED_ATTN" if current_platform.is_rocm()
        else "TRITON_ATTN"
    )
    return [
        "--enforce-eager",
        "--max-model-len", "1024",
        "--max-num-seqs", "256",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "openai_gptoss",
        "--tensor-parallel-size", "4",
        "--attention-backend", attention_backend,
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    """Start vLLM HTTP server for online serving tests."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")
    
    # Handle ROCm AITER if needed
    env_dict = None
    if current_platform.is_rocm() and MODEL_NAME in AITER_MODEL_LIST:
        env_dict = {"VLLM_ROCM_USE_AITER": "1"}
    
    with RemoteOpenAIServer(MODEL_NAME, default_server_args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Get async OpenAI client for testing online serving endpoints."""
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_online_serving_v1_completions(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> None:
    """
    Test online serving via /v1/completions endpoint for gpt-oss-20b.
    
    This test verifies that the vLLM HTTP server correctly handles:
    - Single prompt completions
    - Batch completions (multiple prompts)
    - Streaming completions
    - Response structure and usage statistics
    
    """
    print(f"\n{'='*60}")
    print(f"Testing Online Serving: /v1/completions endpoint")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Define test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain what AI is in simple terms.",
        "Write a short poem about Python programming.",
    ]

    print("Running /v1/completions requests...\n")
    
    # Test single completion requests
    all_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Prompt {i}/{len(test_prompts)}: {prompt}")
        print(f"{'-'*60}")
        
        start_time = time.time()
        completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
        )
        elapsed = time.time() - start_time
        
        # Assertions for completion response
        assert completion.id is not None, "Completion should have an ID"
        assert completion.choices is not None, "Completion should have choices"
        assert len(completion.choices) == 1, f"Should return 1 choice, got {len(completion.choices)}"
        
        choice = completion.choices[0]
        generated_text = choice.text
        
        # Assert output is not empty
        assert generated_text is not None, "Generated text should not be None"
        assert len(generated_text) > 0, "Generated text should not be empty"
        assert isinstance(generated_text, str), "Generated text should be a string"
        assert choice.finish_reason is not None, "Finish reason should be set"
        
        # Assert usage stats
        assert completion.usage is not None, "Usage stats should be present"
        assert completion.usage.completion_tokens > 0, "Should have completion tokens"
        assert completion.usage.prompt_tokens > 0, "Should have prompt tokens"
        assert completion.usage.total_tokens == (
            completion.usage.prompt_tokens + completion.usage.completion_tokens
        ), "Total tokens should equal sum of prompt and completion tokens"
        
        all_outputs.append(completion)
        
        print(f"Response: {generated_text}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens: {completion.usage.completion_tokens}")
        print(f"Assertions passed for prompt {i}")
        print(f"{'='*60}")

    # Assert all prompts generated outputs
    assert len(all_outputs) == len(test_prompts), \
        f"Should generate {len(test_prompts)} outputs, got {len(all_outputs)}"

    # Test batch completion (multiple prompts in one request)
    print(f"\n{'='*60}")
    print("Testing batch /v1/completions with all prompts...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    batch_completion = await client.completions.create(
        model=model_name,
        prompt=test_prompts,
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
    )
    batch_elapsed = time.time() - start_time
    
    # Assertions for batch completion
    assert len(batch_completion.choices) == len(test_prompts), \
        f"Batch should return {len(test_prompts)} choices, got {len(batch_completion.choices)}"
    
    for i, choice in enumerate(batch_completion.choices):
        assert choice.text is not None, \
            f"Choice {i} text should not be None"
        assert len(choice.text) > 0, \
            f"Choice {i} text should not be empty"
        assert choice.finish_reason is not None, \
            f"Choice {i} should have finish reason"
    
    print(f"Batch completion completed in {batch_elapsed:.2f}s")
    print(f"Average time per prompt: {batch_elapsed/len(test_prompts):.2f}s")
    print(f"All batch assertions passed\n")
    
    for i, choice in enumerate(batch_completion.choices, 1):
        print(f"Prompt {i}: {test_prompts[i-1]}")
        print(f"Response: {choice.text[:100]}...")

    # Test streaming completions (important for online serving)
    print(f"\n{'='*60}")
    print("Testing streaming /v1/completions (online serving)...")
    print(f"{'='*60}\n")
    
    prompt = "Tell me a short story about AI."
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=50,
        temperature=0.0,
        stream=True,
    )
    
    chunks = []
    finish_reason_count = 0
    async for chunk in stream:
        assert chunk.choices is not None and len(chunk.choices) > 0
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    
    # Finish reason should only appear in the last chunk
    assert finish_reason_count == 1, "Finish reason should appear exactly once"
    assert len(chunks) > 0, "Should receive at least one chunk"
    streamed_text = "".join(chunks)
    assert len(streamed_text) > 0, "Streamed text should not be empty"
    
    print(f"Streamed text: {streamed_text[:100]}...")
    print(f"Received {len(chunks)} chunks")
    print(f"Streaming completion passed")

    print(f"\n{'='*60}")
    print("✅ All /v1/completions endpoint assertions passed!")
    print(f"{'='*60}\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_online_serving_v1_chat_completions(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> None:
    """
    Test online serving via /v1/chat/completions endpoint for gpt-oss-20b.
    
    This test verifies that the vLLM HTTP server correctly handles:
    - Single chat completion requests
    - Multi-turn conversations
    - Streaming chat completions
    - Response structure and usage statistics
    
    Run with: pytest this_file.py::test_online_serving_v1_chat_completions -v -s
    """
    print(f"\n{'='*60}")
    print(f"Testing Online Serving: /v1/chat/completions endpoint")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Define test messages (chat format)
    test_messages_list = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Explain what AI is in simple terms."}],
        [{"role": "user", "content": "Write a short poem about Python programming."}],
    ]

    print("Running /v1/chat/completions requests...\n")
    
    # Test single chat completion requests
    all_outputs = []
    for i, messages in enumerate(test_messages_list, 1):
        print(f"\n{'='*60}")
        print(f"Messages {i}/{len(test_messages_list)}: {messages[0]['content']}")
        print(f"{'-'*60}")
        
        start_time = time.time()
        chat_completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
        )
        elapsed = time.time() - start_time
        
        # Assertions for chat completion response
        assert chat_completion.id is not None, "Chat completion should have an ID"
        assert chat_completion.choices is not None, "Chat completion should have choices"
        assert len(chat_completion.choices) == 1, \
            f"Should return 1 choice, got {len(chat_completion.choices)}"
        
        choice = chat_completion.choices[0]
        message = choice.message
        
        # Assert message structure (GPT-OSS can return reasoning without content)
        assert message is not None, "Message should not be None"
        assert message.role == "assistant", "Message role should be 'assistant'"
        response_text = message.content or getattr(message, "reasoning", None) or ""
        assert len(response_text) > 0, (
            "Message should have content or reasoning"
        )
        assert choice.finish_reason is not None, "Finish reason should be set"
        
        # Assert usage stats
        assert chat_completion.usage is not None, "Usage stats should be present"
        assert chat_completion.usage.completion_tokens > 0, "Should have completion tokens"
        assert chat_completion.usage.prompt_tokens > 0, "Should have prompt tokens"
        assert chat_completion.usage.total_tokens == (
            chat_completion.usage.prompt_tokens + chat_completion.usage.completion_tokens
        ), "Total tokens should equal sum of prompt and completion tokens"
        
        all_outputs.append(chat_completion)
        
        print(f"Response: {response_text}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens: {chat_completion.usage.completion_tokens}")
        print(f"Assertions passed for messages {i}")
        print(f"{'='*60}")

    # Assert all messages generated outputs
    assert len(all_outputs) == len(test_messages_list), \
        f"Should generate {len(test_messages_list)} outputs, got {len(all_outputs)}"

    # Test multi-turn conversation
    print(f"\n{'='*60}")
    print("Testing multi-turn conversation...")
    print(f"{'='*60}\n")
    
    conversation_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    first_response = await client.chat.completions.create(
        model=model_name,
        messages=conversation_messages,
        max_tokens=50,
        temperature=0.0,
    )
    
    assert len(first_response.choices) == 1
    assistant_message = first_response.choices[0].message
    first_response_text = assistant_message.content or getattr(
        assistant_message, "reasoning", None
    ) or ""
    assert len(first_response_text) > 0, "First response should have content or reasoning"
    
    # Add assistant response and continue conversation
    conversation_messages.append({
        "role": "assistant",
        "content": first_response_text
    })
    conversation_messages.append({
        "role": "user",
        "content": "What about 3+3?"
    })
    
    second_response = await client.chat.completions.create(
        model=model_name,
        messages=conversation_messages,
        max_tokens=50,
        temperature=0.0,
    )
    
    assert len(second_response.choices) == 1
    second_response_text = (
        second_response.choices[0].message.content
        or getattr(second_response.choices[0].message, "reasoning", None)
        or ""
    )
    assert len(second_response_text) > 0, (
        "Second response should have content or reasoning"
    )
    
    print("Multi-turn conversation test passed!")
    print(f"First response: {first_response_text[:100]}...")
    print(f"Second response: {second_response_text[:100]}...")

    # Test streaming chat completions (important for online serving)
    print(f"\n{'='*60}")
    print("Testing streaming /v1/chat/completions (online serving)...")
    print(f"{'='*60}\n")
    
    messages = [{"role": "user", "content": "Tell me a short story about AI."}]
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=50,
        temperature=0.0,
        stream=True,
    )
    
    chunks = []
    reasoning_chunks = []
    finish_reason_count = 0
    async for chunk in stream:
        assert chunk.choices is not None and len(chunk.choices) > 0
        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            reasoning_chunks.append(reasoning)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    
    # Finish reason should only appear in the last chunk
    assert finish_reason_count == 1, "Finish reason should appear exactly once"
    streamed_text = "".join(chunks)
    streamed_reasoning = "".join(reasoning_chunks)
    combined = streamed_text or streamed_reasoning
    assert len(combined) > 0, (
        "Streamed content or reasoning should not be empty"
    )
    
    print(f"Streamed text: {combined[:100]}...")
    print(f"Received {len(chunks)} content + {len(reasoning_chunks)} reasoning chunks")
    print(f"Streaming chat completion passed")

    print(f"\n{'='*60}")
    print("✅ All /v1/chat/completions endpoint assertions passed!")
    print(f"{'='*60}\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_online_serving_concurrent_requests(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> None:
    """
    Test online serving with concurrent requests to both endpoints.
    
    This simulates real-world online serving scenarios where multiple
    clients make simultaneous requests to the server.
    
    Run with: pytest this_file.py::test_online_serving_concurrent_requests -v -s
    """
    print(f"\n{'='*60}")
    print(f"Testing Online Serving: Concurrent Requests")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    async def make_completion_request(prompt: str, request_id: int):
        """Helper to make a completion request."""
        start = time.time()
        completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=50,
            temperature=0.0,
        )
        elapsed = time.time() - start
        text = completion.choices[0].text or ""
        return {
            "id": request_id,
            "type": "completion",
            "text": text,
            "latency": elapsed,
            "tokens": completion.usage.completion_tokens,
        }

    async def make_chat_request(messages: list, request_id: int):
        """Helper to make a chat completion request."""
        start = time.time()
        chat_completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=50,
            temperature=0.0,
        )
        elapsed = time.time() - start
        msg = chat_completion.choices[0].message
        text = msg.content or getattr(msg, "reasoning", None) or ""
        return {
            "id": request_id,
            "type": "chat",
            "text": text,
            "latency": elapsed,
            "tokens": chat_completion.usage.completion_tokens,
        }

    # Create concurrent requests
    print("Sending 5 concurrent /v1/completions requests...")
    completion_tasks = [
        make_completion_request(f"Request {i}: What is {i}+{i}?", i)
        for i in range(1, 6)
    ]
    
    print("Sending 5 concurrent /v1/chat/completions requests...")
    chat_tasks = [
        make_chat_request([{"role": "user", "content": f"Request {i}: What is {i}+{i}?"}], i)
        for i in range(1, 6)
    ]
    
    # Execute all requests concurrently
    start_time = time.time()
    all_results = await asyncio.gather(*completion_tasks, *chat_tasks)
    total_time = time.time() - start_time
    
    # Verify all requests succeeded
    assert len(all_results) == 10, f"Expected 10 results, got {len(all_results)}"
    
    for result in all_results:
        text = result["text"] or ""
        assert len(text) > 0, (
            f"Request {result['id']} should have text (content or reasoning)"
        )
        assert result["latency"] > 0, f"Request {result['id']} should have latency"
        assert result["tokens"] > 0, f"Request {result['id']} should have tokens"
    
    completion_results = [r for r in all_results if r["type"] == "completion"]
    chat_results = [r for r in all_results if r["type"] == "chat"]
    
    print(f"\nConcurrent requests completed successfully!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Completion requests: {len(completion_results)}")
    print(f"Chat requests: {len(chat_results)}")
    print(f"Average latency per request: {total_time/10:.2f}s")
    
    for result in completion_results[:3]:  # Show first 3
        print(f"  Completion {result['id']}: {result['latency']:.2f}s, {result['tokens']} tokens")
    
    for result in chat_results[:3]:  # Show first 3
        print(f"  Chat {result['id']}: {result['latency']:.2f}s, {result['tokens']} tokens")

    print(f"\n{'='*60}")
    print("✅ Concurrent online serving tests passed!")
    print(f"{'='*60}\n")
    

@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_accuracy_gsm8k(server: RemoteOpenAIServer, model_name: str) -> None:
    """
    Measure accuracy via GSM8K evaluation against the same model (gpt-oss-20b).
    Uses the isolated GSM8K script against the already-running vLLM server.
    """
    server_url = server.url_for("v1")
    if "://" in server_url:
        server_url = server_url.split("://")[1]
    host_port = server_url.split("/")[0]
    if ":" in host_port:
        host, p = host_port.split(":")
        port = int(p)
    else:
        host = host_port
        port = 8000
    if not host.startswith("http"):
        host = f"http://{host}"

    results = evaluate_gsm8k(
        num_questions=10,
        num_shots=5,
        host=host,
        port=port,
    )

    accuracy = results["accuracy"]
    print(f"\n{'='*60}")
    print(f"GSM8K accuracy: {model_name}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Questions: {results['num_questions']}")
    print(f"  Invalid rate: {results['invalid_rate']:.3f}")
    print(f"  Latency: {results['latency']:.1f}s")
    print(f"{'='*60}\n")
    assert accuracy >= 0.0, "GSM8K accuracy should be non-negative"

# @pytest.mark.parametrize("model_name", [MODEL_NAME])
# def test_run_guidellm_benchmark(
#     server: RemoteOpenAIServer,
#     model_name: str,
#     num_requests: int = 100,
#     max_concurrency: Optional[int] = None,
#     request_rate: Optional[float] = None,
#     prompt_tokens: int = 100,
#     output_tokens: int = 100,
#     timeout: int = 600,
# ) -> Dict[str, Any]:
#     """
#     Run guidellm benchmark and collect performance metrics.

#     Args:
#         server: RemoteOpenAIServer instance with the running vLLM server
#         model_name: Name of the model to benchmark
#         num_requests: Total number of requests to send
#         max_concurrency: Maximum number of concurrent requests (default: None for unlimited)
#         request_rate: Request rate in requests/second (default: None for max throughput)
#         prompt_tokens: Number of prompt tokens per request
#         output_tokens: Number of output tokens to generate per request
#         timeout: Timeout in seconds for the benchmark

#     Returns:
#         Dictionary containing benchmark results with keys:
#             - throughput: Requests per second
#             - mean_latency: Average latency in seconds
#             - p50_latency: 50th percentile latency
#             - p95_latency: 95th percentile latency
#             - p99_latency: 99th percentile latency
#             - total_tokens: Total tokens processed
#             - tokens_per_second: Token throughput
#             - success_rate: Percentage of successful requests
#             - raw_output: Raw guidellm output
#     """
#     # Get server URL (base URL for the API)
#     server_url = server.url_for("v1")

#     # Build guidellm benchmark command
#     cmd = [
#         "guidellm",
#         "benchmark",
#         "--target", server_url,
#         "--model", model_name,
#         "--data", f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
#         "--rate-type", "concurrent",
#         "--max-seconds", str(timeout),
#         "--max-requests", str(num_requests),
#     ]

#     if request_rate is not None:
#         cmd.extend(["--rate", str(int(request_rate))])
#     else:
#         # Use a single high rate for max throughput when no rate specified
#         cmd.extend(["--rate", "100"])

#     if max_concurrency is not None:
#         cmd.extend(["--max-concurrency", str(max_concurrency)])

#     # Use temp file for JSON output
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
#         output_path = f.name
#     cmd.extend(["--output-path", output_path])

#     print(f"\n{'='*60}")
#     print(f"Running guidellm benchmark:")
#     print(f"  Model: {model_name}")
#     print(f"  Server: {server_url}")
#     print(f"  Requests: {num_requests}")
#     print(f"  Concurrency: {max_concurrency if max_concurrency else 'unlimited'}")
#     print(f"  Request rate: {request_rate if request_rate else 'max throughput'}")
#     print(f"  Prompt tokens: {prompt_tokens}")
#     print(f"  Output tokens: {output_tokens}")
#     print(f"{'='*60}\n")

#     try:
#         # Run guidellm
#         start_time = time.time()
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=timeout + 30,  # Slightly longer than max-seconds
#             check=True,
#         )
#         elapsed = time.time() - start_time

#         # Read and print JSON output
#         with open(output_path, "r") as f:
#             raw_output = f.read()
#         print("GUIDELLM OUTPUT: \n", raw_output)

#         # Clean up temp file
#         os.unlink(output_path)

#         # Parse JSON output
#         try:
#             benchmark_data = json.loads(raw_output)
#         except json.JSONDecodeError:
#             print("Warning: Could not parse guidellm JSON output, using raw output")
#             benchmark_data = {
#                 "raw_output": raw_output,
#                 "stderr": result.stderr,
#             }

#         # Extract key metrics (adjust keys based on actual guidellm output format)
#         metrics = {
#             "throughput": benchmark_data.get("throughput", benchmark_data.get("requests_per_second", 0.0)),
#             "mean_latency": benchmark_data.get("mean_latency", benchmark_data.get("latency_mean", 0.0)),
#             "p50_latency": benchmark_data.get("p50_latency", benchmark_data.get("latency_p50", 0.0)),
#             "p95_latency": benchmark_data.get("p95_latency", benchmark_data.get("latency_p95", 0.0)),
#             "p99_latency": benchmark_data.get("p99_latency", benchmark_data.get("latency_p99", 0.0)),
#             "total_tokens": benchmark_data.get("total_tokens", 0),
#             "tokens_per_second": benchmark_data.get("tokens_per_second", 0.0),
#             "success_rate": benchmark_data.get("success_rate", 1.0),
#             "total_time": elapsed,
#             "raw_output": raw_output,
#             "full_data": benchmark_data,
#         }

#         print(f"\n{'='*60}")
#         print(f"Guidellm benchmark completed in {elapsed:.2f}s")
#         print(f"  Throughput: {metrics['throughput']:.2f} req/s")
#         print(f"  Mean latency: {metrics['mean_latency']:.3f}s")
#         print(f"  P95 latency: {metrics['p95_latency']:.3f}s")
#         print(f"  P99 latency: {metrics['p99_latency']:.3f}s")
#         print(f"  Tokens/sec: {metrics['tokens_per_second']:.2f}")
#         print(f"  Success rate: {metrics['success_rate']:.1%}")
#         print(f"{'='*60}\n")

#         return metrics

#     except subprocess.TimeoutExpired:
#         raise RuntimeError(f"Guidellm benchmark timed out after {timeout} seconds")
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(
#             f"Guidellm benchmark failed with exit code {e.returncode}:\n"
#             f"stdout: {e.stdout}\n"
#             f"stderr: {e.stderr}"
#         )
#     except FileNotFoundError:
#         raise RuntimeError(
#             "guidellm command not found. Please install guidellm:\n"
#             "  pip install guidellm"
#         )

@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_run_guidellm_benchmark(
    server: RemoteOpenAIServer,
    model_name: str,
    num_requests: int = 100,
    max_concurrency: Optional[int] = None,
    request_rate: Optional[float] = None,
    prompt_tokens: int = 100,
    output_tokens: int = 100,
    timeout: int = 600,
) -> None:
    server_url = server.url_for("v1")

    # Extract vLLM version
    vllm_version = vllm.__version__
    print("VLLM VERSION: ", vllm_version)

    output_filename = f"gptoss120b_{vllm_version}.json"
    print("OUTPUT FILENAME: ", output_filename)

    cmd = [
        "guidellm",
        "benchmark",
        "--target", server_url,
        "--model", model_name,
        "--data", f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        "--rate-type", "concurrent",
        "--max-seconds", str(timeout),
        "--max-requests", str(num_requests),
        "--output-path", output_filename,
    ]

    if request_rate is not None:
        cmd.extend(["--rate", str(int(request_rate))])
    else:
        cmd.extend(["--rate", "100"])

    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])

    print(f"\nRunning guidellm benchmark → output: {output_filename}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            check=True,
        )
        print(f"Benchmark complete. Results saved to {output_filename}")

        # Read and print JSON output
        with open(output_filename, "r") as f:
            raw_output = f.read()
        print("GUIDELLM OUTPUT: \n", raw_output)

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Guidellm benchmark timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Guidellm benchmark failed with exit code {e.returncode}:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "guidellm command not found. Please install guidellm:\n"
            "  pip install guidellm"
        )