# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

import requests
from test_const import TEST_PROMPTS as prompts
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer

# Global tokenizer instance - will be initialized once
_tokenizer = None


class QualityResult(TypedDict):
    request_id: int
    content_length: int
    is_quality_good: bool
    issues: list[str]
    metrics: dict[Any, Any]


class RequestResult(TypedDict):
    request_id: int
    tokens_received: int
    cancelled_early: bool
    completed_naturally: bool
    error: str | None
    partial_content: str
    final_response: Any
    logprobs_data: list[Any]


def initialize_tokenizer(model_name: str):
    """
    Initialize the global tokenizer once based on the model name.
    This ensures we only load the tokenizer once per test run.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Tokenizer loaded successfully: {_tokenizer.__class__.__name__}")
    return _tokenizer


def count_actual_tokens(content: str) -> int:
    """
    Count actual tokens in the content using the pre-loaded global tokenizer.

    This uses the global tokenizer instance to avoid reloading it multiple times.
    """
    if not content.strip():
        return 0

    global _tokenizer
    if _tokenizer is not None:
        try:
            tokens = _tokenizer.encode(content, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logging.error("Error tokenizing content: %s", e)
            return 0
    return 0


def validate_output_quality(
    content: str, request_id: int, logprobs_data: list | None = None
) -> QualityResult:
    """
    Validate that generated content is coherent and not degraded.
    Optionally validates logprobs if provided.

    Args:
        content: Generated text content
        request_id: Request identifier
        logprobs_data: Optional logprobs data for additional validation

    Returns dict with quality metrics and pass/fail status.
    """
    import regex as re

    quality_result: QualityResult = {
        "request_id": request_id,
        "content_length": len(content),
        "is_quality_good": True,
        "issues": [],
        "metrics": {},
    }

    # Check for empty or very short content
    if len(content.strip()) < 10:
        quality_result["is_quality_good"] = False
        quality_result["issues"].append("Content too short or empty")
        return quality_result

    # Check if empty string
    words = content.lower().split()
    if len(words) == 0:
        quality_result["is_quality_good"] = False
        quality_result["issues"].append("No words found")
        return quality_result

    # Detect repetitive patterns (doom loops)
    unique_words = set(words)
    repetition_ratio = len(words) / len(unique_words) if unique_words else float("inf")
    quality_result["metrics"]["repetition_ratio"] = repetition_ratio

    if repetition_ratio > 5.0:
        print(
            f"⚠️  WARNING: Doom loop detected for request {request_id} "
            f"(repetition ratio: {repetition_ratio:.2f})"
        )
    elif repetition_ratio > 3.0:
        print(
            f"⚠️  WARNING: Moderate repetition for request {request_id} "
            f"(ratio: {repetition_ratio:.2f}) - monitoring for doom loop"
        )

    # Check for pad tokens or common garbage patterns
    pad_patterns = [
        r"<\|PAD_TOKEN\|>",
    ]

    for pattern in pad_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            quality_result["is_quality_good"] = False
            quality_result["issues"].append(f"Found pad/special tokens: {pattern}")

    # Check for gibberish. Known to trip on B200 nightly under speculative
    # decoding (see docs/cohere/tests/features/speculative_decoding_test.md);
    # surface as a warning so it is visible in logs without failing the test.
    gibberish_patterns = [
        r"[^a-zA-Z\s]{200,}",  # Long sequences of non-letters
        r"(.)\1{200,}",  # Same character repeated 200+ times
    ]

    for pattern in gibberish_patterns:
        if re.search(pattern, content):
            quality_result["metrics"].setdefault("gibberish_patterns", []).append(
                pattern
            )
            print(
                f"⚠️  WARNING: Gibberish pattern detected for request {request_id} "
                f"(pattern: {pattern}). Content sample: {content[:500]}..."
            )

    # Ensure logprobs data is always available for quality validation
    if not logprobs_data or len(logprobs_data) == 0:
        error_msg = f"ERROR: No logprobs data available for request {request_id}."
        print(f"🚨 {error_msg}")
        raise RuntimeError(error_msg)

    # Validate logprobs quality since we have the required data
    logprobs_validation = validate_logprobs_quality(logprobs_data, request_id)

    if "logprobs" not in quality_result["metrics"]:
        quality_result["metrics"]["logprobs"] = {}
    quality_result["metrics"]["logprobs"] = logprobs_validation["metrics"]

    metrics = logprobs_validation["metrics"]
    display_logprobs_metrics(metrics, request_id)

    if not logprobs_validation["is_logprobs_quality_good"]:
        quality_result["is_quality_good"] = False
        quality_result["issues"].extend(logprobs_validation["issues"])

        print(f"Logprobs quality issues detected for request {request_id}:")
        for issue in logprobs_validation["issues"]:
            print(f"   - {issue}")
    else:
        print(f"Request {request_id} logprobs quality: GOOD")

    return quality_result


def validate_completed_requests_quality(
    completed_requests: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Validate quality of completed requests.

    Args:
        completed_requests: List of completed request results

    Returns:
        tuple: (all_passed, failed_requests) where all_passed
        is bool and failed_requests is list of failed results
    """
    failed_requests = []

    for result in completed_requests:
        # Basic content checks
        if len(result["partial_content"]) <= 50:
            failed_requests.append(
                {
                    "request_id": result["request_id"],
                    "reason": (
                        f"Completed request has very short content "
                        f"({len(result['partial_content'])} chars)"
                    ),
                }
            )
            continue

        if result["tokens_received"] <= 10:
            failed_requests.append(
                {
                    "request_id": result["request_id"],
                    "reason": (
                        f"Completed request received too few tokens "
                        f"({result['tokens_received']})"
                    ),
                }
            )
            continue

        logprobs_data = result.get("logprobs_data", [])
        quality_check = validate_output_quality(
            result["partial_content"], result["request_id"], logprobs_data
        )
        if not quality_check["is_quality_good"]:
            print(
                f"Quality issues in completed request {result['request_id']}: "
                f"{quality_check['issues']}"
            )
            print(f"Content sample: {result['partial_content']}")

            failed_requests.append(
                {
                    "request_id": result["request_id"],
                    "reason": f"Quality degraded:{quality_check['issues']}",
                }
            )

    return len(failed_requests) == 0, failed_requests


def validate_cancelled_requests_quality(
    cancelled_requests: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Validate quality of cancelled requests.

    Args:
        cancelled_requests: List of cancelled request results

    Returns:
        tuple: (all_passed, failed_requests) where all_passed is
        bool and failed_requests is list of failed results
    """
    failed_requests = []

    for result in cancelled_requests:
        # Basic checks for cancelled requests
        if result["tokens_received"] == 0:
            failed_requests.append(
                {
                    "request_id": result["request_id"],
                    "reason": "Cancelled request received no tokens",
                }
            )
            continue

        if len(result["partial_content"]) == 0:
            failed_requests.append(
                {
                    "request_id": result["request_id"],
                    "reason": "Cancelled request has no content",
                }
            )
            continue

        # Even partial content should not be gibberish
        # (only check if we have enough content)
        if len(result["partial_content"]) > 20:
            # Include logprobs data if available for enhanced quality validation
            logprobs_data = result.get("logprobs_data", [])
            partial_quality = validate_output_quality(
                result["partial_content"], result["request_id"], logprobs_data
            )
            if not partial_quality["is_quality_good"]:
                print(
                    f"Quality issues in partial content {result['request_id']}: "
                    f"{partial_quality['issues']}"
                )
                print(f"Partial content: {result['partial_content']}")

    return len(failed_requests) == 0, failed_requests


def validate_logprobs_quality(logprobs_data: list, request_id: int) -> dict[str, Any]:
    """
    Validate logprobs for quality issues like very high negative values.

    This function validates generation logprobs only (not input/prompt logprobs).
    High negative logprobs (e.g., < -10) indicate very low model confidence
    and may signal generation quality issues.

    Args:
        logprobs_data: List of logprobs data from streaming response
            (generation tokens only)
        request_id: Request identifier for logging

    Returns:
        dict with validation results and metrics
    """
    validation_result: dict[str, Any] = {
        "request_id": request_id,
        "is_logprobs_quality_good": True,
        "issues": [],
        "metrics": {
            "total_tokens_with_logprobs": 0,
            "min_logprob": 0.0,
            "max_logprob": 0.0,
            "avg_logprob": 0.0,
            "very_negative_count": 0,
            "extremely_negative_count": 0,
        },
    }

    if not logprobs_data or len(logprobs_data) == 0:
        validation_result["issues"].append("No logprobs data available")
        return validation_result

    all_logprobs = []
    very_negative_threshold = -5.0  # Flag logprobs worse than -5
    extremely_negative_threshold = -10.0  # Flag logprobs worse than -10

    very_negative_count = 0
    extremely_negative_count = 0

    try:
        for token_logprobs in logprobs_data:
            if (
                token_logprobs
                and isinstance(token_logprobs, dict)
                and ("token" in token_logprobs or "logprob" in token_logprobs)
            ):
                # Additional validation: ensure this is generation logprobs data
                # Check if it has expected structure for generated tokens
                for token, logprob_info in token_logprobs.items():
                    if hasattr(logprob_info, "logprob"):
                        logprob_value = logprob_info.logprob
                    elif isinstance(logprob_info, dict) and "logprob" in logprob_info:
                        logprob_value = logprob_info["logprob"]
                    elif isinstance(logprob_info, (int, float)):
                        logprob_value = logprob_info
                    else:
                        continue

                    # Validate this is a reasonable generation logprob value
                    # Generation logprobs should typically be in range [-50, 0]
                    if (
                        isinstance(logprob_value, (int, float))
                        and -100 <= logprob_value <= 5
                    ):
                        all_logprobs.append(logprob_value)

                        if logprob_value < extremely_negative_threshold:
                            extremely_negative_count += 1
                        elif logprob_value < very_negative_threshold:
                            very_negative_count += 1
                    else:
                        # Skip obviously invalid logprob values
                        print(
                            f"Warning: Skipping invalid logprob value "
                            f"{logprob_value} for request {request_id}"
                        )
                        continue

        if all_logprobs:
            validation_result["metrics"]["total_tokens_with_logprobs"] = len(
                all_logprobs
            )
            validation_result["metrics"]["min_logprob"] = min(all_logprobs)
            validation_result["metrics"]["max_logprob"] = max(all_logprobs)
            validation_result["metrics"]["avg_logprob"] = sum(all_logprobs) / len(
                all_logprobs
            )
            validation_result["metrics"]["very_negative_count"] = very_negative_count
            validation_result["metrics"]["extremely_negative_count"] = (
                extremely_negative_count
            )

            # Flag quality issues based on logprobs
            total_tokens = len(all_logprobs)
            very_negative_ratio = (
                very_negative_count / total_tokens if total_tokens > 0 else 0
            )
            extremely_negative_ratio = (
                extremely_negative_count / total_tokens if total_tokens > 0 else 0
            )

            # Flag if too many tokens have very negative logprobs
            if extremely_negative_ratio > 0.1:  # More than 10% extremely negative
                validation_result["is_logprobs_quality_good"] = False
                validation_result["issues"].append(
                    f"Too many extremely negative logprobs: "
                    f"{extremely_negative_count}/{total_tokens} "
                    f"({extremely_negative_ratio:.2%}) below "
                    f"{extremely_negative_threshold}"
                )

            if very_negative_ratio > 0.3:  # More than 30% very negative
                validation_result["is_logprobs_quality_good"] = False
                validation_result["issues"].append(
                    f"Too many very negative logprobs: "
                    f"{very_negative_count}/{total_tokens} "
                    f"({very_negative_ratio:.2%}) below {very_negative_threshold}"
                )

            # Flag if average logprob is extremely low
            avg_logprob = validation_result["metrics"]["avg_logprob"]
            if avg_logprob < -10.0:
                validation_result["is_logprobs_quality_good"] = False
                validation_result["issues"].append(
                    f"Average logprob too negative: {avg_logprob:.2f} "
                    f"(threshold: -15.0)"
                )

            # Flag if minimum logprob is extremely low (could indicate numerical issues)
            min_logprob = validation_result["metrics"]["min_logprob"]
            if min_logprob < -50.0:
                validation_result["is_logprobs_quality_good"] = False
                validation_result["issues"].append(
                    f"Minimum logprob extremely negative: {min_logprob:.2f} "
                    f"(threshold: -50.0)"
                )

    except Exception as e:
        validation_result["issues"].append(f"Error processing logprobs: {str(e)}")
        validation_result["is_logprobs_quality_good"] = False

    return validation_result


def test_streaming_request_cancellation(server, model_name=None, num_requests=32):
    """
    Test cancellation of streaming requests during generation.

    This test simulates cancelling streaming requests midway through
    # token generation,
    which is a common real-world scenario when users stop generation early.
    """
    import json

    # Use provided model name or fall back to global
    def send_streaming_request(
        request_id: int,
        cancel_after_tokens: int | None = None,
        prompt: str | None = None,
    ):
        """Send a streaming request and optionally cancel it
        after receiving N tokens"""
        url = f"http://{server.host}:{server.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        content = prompt

        payload = {
            "model": model_name,  # Use the model name from command line
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32000,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": True,  # Enable streaming
            "logprobs": True,  # Enable logprobs for quality validation
            "top_logprobs": 1,  # Get top-1 logprobs for detailed analysis
            "thinking_token_budget": 4096,  # Enable thinking token budget for reasoning
        }

        result: RequestResult = {
            "request_id": request_id,
            "tokens_received": 0,
            "cancelled_early": False,
            "completed_naturally": False,
            "error": None,
            "partial_content": "",
            "final_response": None,
            "logprobs_data": [],
        }

        try:
            response = requests.post(
                url, json=payload, headers=headers, stream=True, timeout=30
            )

            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                return result
            content_parts = []

            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str.strip() == "[DONE]":
                        result["completed_naturally"] = True
                        break

                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]

                            # Collect logprobs data for quality validation
                            # (generation tokens only)
                            if "logprobs" in choice and choice["logprobs"]:
                                logprobs_info = choice["logprobs"]
                                if (
                                    "content" in logprobs_info
                                    and logprobs_info["content"]
                                    and "delta" in choice
                                    and "content" in choice["delta"]
                                    and choice["delta"]["content"]
                                ):
                                    # Only collect logprobs when we have delta content
                                    # (generation phase)
                                    # This ensures we skip input/prompt token logprobs
                                    # Store the logprobs for generated tokens only
                                    for token_logprob in logprobs_info["content"]:
                                        if token_logprob:
                                            result["logprobs_data"].append(
                                                token_logprob
                                            )
                            else:
                                # Check if this is a generation token without
                                # logprobs - ERROR!
                                if (
                                    "delta" in choice
                                    and "content" in choice["delta"]
                                    and choice["delta"]["content"]
                                ):
                                    error_msg = (
                                        f"SYNC ERROR: Request {request_id} - "
                                        f"Generation token received without "
                                        f"logprobs data."
                                    )
                                    print(f"{error_msg}")
                                    result["error"] = error_msg
                                    raise RuntimeError(error_msg)

                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    content_parts.append(content)

                                    # Update accumulated content and count actual tokens
                                    current_content = "".join(content_parts)
                                    actual_tokens = count_actual_tokens(current_content)
                                    result["tokens_received"] = actual_tokens

                                    # Cancel if we've received enough ACTUAL tokens
                                    if (
                                        cancel_after_tokens
                                        and actual_tokens >= cancel_after_tokens
                                    ):
                                        result["cancelled_early"] = True
                                        response.close()
                                        break

                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON

            result["partial_content"] = "".join(content_parts)

        except requests.exceptions.RequestException as e:
            result["error"] = str(e)

        return result

    # Test configuration for {num_requests} concurrent streaming
    # requests with varied prompts
    streaming_config = []
    for i in range(num_requests):
        # Vary cancellation patterns: some complete naturally,
        # others cancel at different points
        if i % 4 == 0:
            cancel_after = None  # Let complete naturally
        elif i % 4 == 1:
            cancel_after = 3 + (i % 5)  # Cancel after 3-7 tokens
        elif i % 4 == 2:
            cancel_after = 10 + (i % 8)  # Cancel after 10-17 tokens
        else:
            cancel_after = 20 + (i % 10)  # Cancel after 20-29 tokens

        streaming_config.append(
            {
                "request_id": i,
                "cancel_after_tokens": cancel_after,
                "prompt": prompts[i % len(prompts)],
            }
        )

    print(
        f"\n==> Starting streaming cancellation test with "
        f"{len(streaming_config)} concurrent requests"
    )
    print("==> Using REAL TOKENIZER for accurate token-based cancellation")
    global _tokenizer
    if _tokenizer:
        print(f"==> Tokenizer loaded: {_tokenizer.__class__.__name__}")
    else:
        print("==> Warning: Using fallback word-based tokenization")

    results = []
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []

        for config in streaming_config:
            future = executor.submit(
                send_streaming_request,
                request_id=config["request_id"],
                cancel_after_tokens=config["cancel_after_tokens"],
                prompt=config["prompt"],
            )
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    # Analyze streaming results
    completed_naturally = [r for r in results if r["completed_naturally"]]
    cancelled_early = [r for r in results if r["cancelled_early"]]
    error_results = [r for r in results if r["error"]]

    print("\n==> Streaming Results:")
    print(f"==> Completed naturally: {len(completed_naturally)}")
    print(f"==> Cancelled early: {len(cancelled_early)}")
    print(f"==> Errors: {len(error_results)}")

    for result in results:
        status = (
            "completed"
            if result["completed_naturally"]
            else "cancelled"
            if result["cancelled_early"]
            else "error"
        )
        print(
            f"Request {result['request_id']}: {status}, "
            f"tokens: {result['tokens_received']}, "
            f"content_len: {len(result['partial_content'])}"
        )

    # Aggregate logprobs statistics across all requests
    aggregate_and_display_logprobs_stats(results, "Sync")

    # Verify completed requests have substantial content AND good quality
    all_passed, failed_requests = validate_completed_requests_quality(
        completed_naturally
    )
    if not all_passed:
        print(
            f"==> Completed requests quality "
            f"validation FAILED for requests: {failed_requests}"
        )
        raise AssertionError(
            f"Completed requests quality validation "
            f"failed for requests: {failed_requests}"
        )

    # Verify cancelled requests received some tokens before
    # cancellation (and they're coherent)
    all_passed, failed_requests = validate_cancelled_requests_quality(cancelled_early)
    if not all_passed:
        print(
            f"==> Cancelled requests quality validation "
            f"FAILED for requests: {failed_requests}"
        )
        # Don't fail the test, but log the issues
        for fr in failed_requests:
            print(f"Request {fr['request_id']} failed: {fr['reason']}")

    print("==> Streaming request cancellation test completed successfully!")


async def test_async_streaming_request_cancellation(
    server, model_name=None, num_requests=32
):
    """
    Async version of streaming request cancellation test.

    This uses asyncio and aiohttp for true async concurrency,
    which is more efficient for I/O-bound operations like HTTP requests.
    """
    import json

    import aiohttp

    # Use provided model name or fall back to global
    async def send_async_streaming_request(
        session: aiohttp.ClientSession,
        request_id: int,
        cancel_after_tokens: int | None = None,
        prompt: str | None = None,
    ):
        """Send an async streaming request and optionally cancel it
        after receiving N tokens"""
        url = f"http://{server.host}:{server.port}/v1/chat/completions"

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": True,  # Enable streaming
            "max_tokens": 4000,
            "logprobs": True,  # Enable logprobs for quality validation
            "top_logprobs": 1,  # Get top-1 logprobs for detailed analysis
        }

        result: RequestResult = {
            "request_id": request_id,
            "tokens_received": 0,
            "cancelled_early": False,
            "completed_naturally": False,
            "error": None,
            "partial_content": "",
            "final_response": None,
            "logprobs_data": [],  # Store logprobs for quality validation
        }

        try:
            # Create timeout for the entire request
            timeout = aiohttp.ClientTimeout(total=30)

            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    result["error"] = f"HTTP {response.status}: {await response.text()}"
                    return result

                content_parts = []

                # Process streaming response line by line
                async for line in response.content:
                    try:
                        line_str = line.decode("utf-8").strip()

                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix

                            if data_str.strip() == "[DONE]":
                                result["completed_naturally"] = True
                                break

                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]

                                    # Collect logprobs data for quality validation
                                    # (generation tokens only)
                                    if "logprobs" in choice and choice["logprobs"]:
                                        logprobs_info = choice["logprobs"]
                                        if (
                                            "content" in logprobs_info
                                            and logprobs_info["content"]
                                            and "delta" in choice
                                            and "content" in choice["delta"]
                                            and choice["delta"]["content"]
                                        ):
                                            # Only collect logprobs when we have delta
                                            # content (generation phase)
                                            # This ensures we skip input/prompt
                                            # token logprobs
                                            # Store the logprobs for generated
                                            # tokens only
                                            for token_logprob in logprobs_info[
                                                "content"
                                            ]:
                                                if token_logprob:
                                                    result["logprobs_data"].append(
                                                        token_logprob
                                                    )
                                    else:
                                        # Check if this is a generation token
                                        # without logprobs - ERROR!
                                        if (
                                            "delta" in choice
                                            and "content" in choice["delta"]
                                            and choice["delta"]["content"]
                                        ):
                                            error_msg = (
                                                f"ASYNC ERROR: Request {request_id} - "
                                                f"Generation token received without "
                                                f"logprobs data."
                                            )
                                            print(f"{error_msg}")
                                            result["error"] = error_msg
                                            raise RuntimeError(error_msg)

                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        content = choice["delta"]["content"]
                                        if content:
                                            content_parts.append(content)

                                            # Update accumulated content and
                                            # count actual tokens
                                            current_content = "".join(content_parts)
                                            actual_tokens = count_actual_tokens(
                                                current_content
                                            )
                                            result["tokens_received"] = actual_tokens

                                            if (
                                                cancel_after_tokens
                                                and actual_tokens >= cancel_after_tokens
                                            ):
                                                result["cancelled_early"] = True
                                                break

                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON

                    except UnicodeDecodeError:
                        continue  # Skip malformed lines

                result["partial_content"] = "".join(content_parts)

        except asyncio.TimeoutError:
            result["error"] = "Request timeout"
        except aiohttp.ClientError as e:
            result["error"] = f"Client error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result

    # Test configuration for {num_requests} concurrent async streaming
    # requests with varied prompts
    streaming_config = []
    for i in range(num_requests):
        # Vary cancellation patterns: some complete naturally,
        # others cancel at different points
        if i % 4 == 0:
            cancel_after = None  # Let complete naturally
        elif i % 4 == 1:
            cancel_after = 3 + (i % 5)  # Cancel after 3-7 tokens
        elif i % 4 == 2:
            cancel_after = 10 + (i % 8)  # Cancel after 10-17 tokens
        else:
            cancel_after = 20 + (i % 10)  # Cancel after 20-29 tokens

        streaming_config.append(
            {
                "request_id": i,
                "cancel_after_tokens": cancel_after,
                "prompt": prompts[i % len(prompts)],
            }
        )

    print(
        f"\n==> Starting ASYNC streaming cancellation test "
        f"with {len(streaming_config)} concurrent requests"
    )
    print("==> Using REAL TOKENIZER for accurate token-based cancellation")
    global _tokenizer
    if _tokenizer:
        print(f"==> Tokenizer loaded: {_tokenizer.__class__.__name__}")
    else:
        print("==> Warning: Using fallback word-based tokenization")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for config in streaming_config:
            task = asyncio.create_task(
                send_async_streaming_request(
                    session=session,
                    request_id=config["request_id"],
                    cancel_after_tokens=config["cancel_after_tokens"],
                    prompt=config["prompt"],
                )
            )
            tasks.append(task)
        # Wait for all tasks to complete concurrently
        print(f"==> All {len(tasks)} async tasks started simultaneously")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out any exceptions and convert to results
        successful_results = [r for r in results if isinstance(r, dict)]
        exceptions = [r for r in results if not isinstance(r, dict)]

        if exceptions:
            print(f"{len(exceptions)} tasks had exceptions: {exceptions}")

    # Analyze async streaming results
    completed_naturally = [r for r in successful_results if r["completed_naturally"]]
    cancelled_early = [r for r in successful_results if r["cancelled_early"]]
    error_results = [r for r in successful_results if r["error"]]

    print("\n==> Async Streaming Results:")
    print(f"==> Completed naturally: {len(completed_naturally)}")
    print(f"==> Cancelled early: {len(cancelled_early)}")
    print(f"==> Errors: {len(error_results)}")
    print(f"==> Exceptions: {len(exceptions)}")

    for result in successful_results:
        status = (
            "completed"
            if result["completed_naturally"]
            else "cancelled"
            if result["cancelled_early"]
            else "error"
        )
        print(
            f"Request {result['request_id']}: {status}, "
            f"tokens: {result['tokens_received']}, "
            f"content_len: {len(result['partial_content'])}"
        )

    # Aggregate logprobs statistics across all async requests
    aggregate_and_display_logprobs_stats(successful_results, "Async")

    # Verify completed requests have substantial content AND good quality
    all_passed, failed_requests = validate_completed_requests_quality(
        completed_naturally
    )
    if not all_passed:
        print(
            f"==> Completed requests quality "
            f"validation FAILED for requests: {failed_requests}"
        )
        raise AssertionError(
            f"Completed requests quality validation "
            f"failed for requests: {failed_requests}"
        )

    # Verify cancelled requests received some tokens before
    # cancellation (and they're coherent)
    all_passed, failed_requests = validate_cancelled_requests_quality(cancelled_early)
    if not all_passed:
        print(
            f"==> Cancelled requests quality "
            f"validation FAILED for requests: {failed_requests}"
        )
        # Don't fail the test, but log the issues
        for fr in failed_requests:
            print(f"Request {fr['request_id']} failed: {fr['reason']}")

    print("==> Async streaming request cancellation test completed successfully!")


# Wrapper to run the async test
def test_async_streaming_request_cancellation_wrapper(
    server, model_name=None, num_requests=32
):
    """Wrapper to run the async test in the event loop"""
    asyncio.run(
        test_async_streaming_request_cancellation(server, model_name, num_requests)
    )


def get_tokenizer_info():
    """Get information about the currently loaded tokenizer"""
    global _tokenizer
    if _tokenizer is not None:
        return {
            "loaded": True,
            "class": _tokenizer.__class__.__name__,
            "vocab_size": len(_tokenizer)
            if hasattr(_tokenizer, "__len__")
            else "Unknown",
        }
    else:
        return {"loaded": False, "fallback": "word-based splitting"}


def display_logprobs_metrics(
    metrics: dict, request_id: int | None = None, prefix: str = ""
):
    """
    Display logprobs metrics in a consistent format.

    Args:
        metrics: Dictionary containing logprobs metrics
        request_id: Optional request ID for per-request display
        prefix: Optional prefix for the display (e.g., "Sync", "Async")
    """
    if metrics.get("total_tokens_with_logprobs", 0) > 0:
        total_tokens = metrics["total_tokens_with_logprobs"]
        very_negative_count = metrics.get("very_negative_count", 0)
        extremely_negative_count = metrics.get("extremely_negative_count", 0)

        if request_id is not None:
            print(f"Request {request_id} logprobs metrics:")
        else:
            print(f"{prefix} Logprobs Statistics:")

        print(f"   Total tokens: {total_tokens}")
        print(f"   Min logprob: {metrics['min_logprob']:.4f}")
        print(f"   Max logprob: {metrics['max_logprob']:.4f}")
        print(f"   Avg logprob: {metrics['avg_logprob']:.4f}")

        # Display negative counts with percentages
        very_negative_pct = (
            (very_negative_count / total_tokens * 100) if total_tokens > 0 else 0
        )
        extremely_negative_pct = (
            (extremely_negative_count / total_tokens * 100) if total_tokens > 0 else 0
        )

        print(
            f"   Very negative (<-5): {very_negative_count} ({very_negative_pct:.1f}%)"
        )
        print(
            f"   Extremely negative (<-15): {extremely_negative_count} "
            f"({extremely_negative_pct:.1f}%)"
        )


def aggregate_and_display_logprobs_stats(results: list[dict], test_type: str = ""):
    """
    Aggregate generation logprobs statistics across all requests and
    display summary.

    This function only processes generation logprobs (not input/prompt logprobs)
    that were collected during the streaming response phase.

    Args:
        results: List of request results containing logprobs_data
            (generation tokens only)
        test_type: Optional test type prefix for display (e.g., "Sync", "Async")
    """
    print(f"\n==> Overall {test_type} Generation Logprobs Statistics Summary:")
    print(
        "    (Note: Only includes logprobs from generated tokens"
        ", not input/prompt tokens)"
    )
    all_logprobs = []
    requests_with_logprobs = 0

    for result in results:
        if result.get("logprobs_data") and len(result["logprobs_data"]) > 0:
            requests_with_logprobs += 1
            for token_logprobs in result["logprobs_data"]:
                if token_logprobs and isinstance(token_logprobs, dict):
                    for token, logprob_info in token_logprobs.items():
                        if hasattr(logprob_info, "logprob"):
                            all_logprobs.append(logprob_info.logprob)
                        elif (
                            isinstance(logprob_info, dict) and "logprob" in logprob_info
                        ):
                            all_logprobs.append(logprob_info["logprob"])
                        elif isinstance(logprob_info, (int, float)):
                            all_logprobs.append(logprob_info)

    prefix = f"{test_type} " if test_type else ""

    if all_logprobs:
        import statistics

        print(
            f"{prefix}Requests with logprobs data: "
            f"{requests_with_logprobs}/{len(results)}"
        )
        print(f"Total {prefix.lower()}tokens analyzed: {len(all_logprobs)}")
        print(f"Min logprob: {min(all_logprobs):.4f}")
        print(f"Max logprob: {max(all_logprobs):.4f}")

        print(f"Average logprob: {statistics.mean(all_logprobs):.4f}")

        # Use thresholds consistent with validate_logprobs_quality function
        very_negative_threshold = -5.0 if test_type == "Async" else -10.0
        extremely_negative_threshold = -10.0 if test_type == "Async" else -20.0

        very_negative = sum(1 for lp in all_logprobs if lp < very_negative_threshold)
        extremely_negative = sum(
            1 for lp in all_logprobs if lp < extremely_negative_threshold
        )
        print(
            f"Very negative logprobs (<{very_negative_threshold}): "
            f"{very_negative} ({very_negative / len(all_logprobs) * 100:.1f}%)"
        )
        print(
            f"Extremely negative logprobs (<{extremely_negative_threshold}): "
            f"{extremely_negative} "
            f"({extremely_negative / len(all_logprobs) * 100:.1f}%)"
        )

        if very_negative > 0 or extremely_negative > 0:
            print(
                f"High negative logprobs detected in {prefix.lower()}requests - "
                f"potential quality issues!"
            )
        else:
            print(f"All {prefix.lower()}logprobs within reasonable ranges")
    else:
        print(f"No logprobs data found in any {prefix.lower()}requests")


if __name__ == "__main__":
    """
    Run tests independently without pytest.

    This allows you to run the file directly with:
    python test_request_cancellation.py
    Usage examples:
      python3 test_request_cancellation.py
      --model /host/engines/c4_hiswar/poseidon/
      --tp-size 2
      --draft_model /host/engines/c4_hiswar_spec/
    """
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run vLLM Request Cancellation Tests with configurable parameters"
    )
    parser.add_argument("--model", type=str, help="Path to the main model)")
    parser.add_argument(
        "--tp-size",
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size (default: 2)",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help='Speculative decoding config JSON string \
        (e.g., \'{"method": "eagle", "model": "/path",\
          "num_speculative_tokens": 3}\')',
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=400000,
        help="Maximum model length (default: 400000)",
    )
    parser.add_argument(
        "--disable-spec",
        action="store_true",
        help="Disable speculative decoding entirely",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        default=[32],
        help="Concurrency levels to sweep (default: 32)",
    )

    parsed_args = parser.parse_args()

    # Use the model from command line arguments
    model_name = parsed_args.model
    draft_model_name = parsed_args.draft_model
    concurrency_levels = sorted(parsed_args.num_requests)
    max_concurrency = max(concurrency_levels)

    # Initialize the tokenizer once at the beginning
    if model_name:
        initialize_tokenizer(model_name)
        tokenizer_info = get_tokenizer_info()
        if tokenizer_info["loaded"]:
            print(
                f"Tokenizer initialized: {tokenizer_info['class']} "
                f"(vocab_size: {tokenizer_info['vocab_size']})"
            )
        else:
            print(
                f"Tokenizer failed to load, using fallback: "
                f"{tokenizer_info['fallback']}"
            )
    else:
        print("Warning: No model name provided, tokenizer will use fallback methods")

    print("=" * 80)
    print("Running vLLM Request Cancellation Concurrency Sweep")
    print(f"Concurrency levels: {concurrency_levels}")
    print("=" * 80)

    # Import the server utility
    from tests.utils import RemoteOpenAIServer

    # Build server arguments based on command line inputs.
    # Profile defaults are applied inside the server process via
    # VLLM_ENABLE_COHERE_AUTO_CONFIG (passed through env_dict below).
    server_args = [
        "--max-model-len",
        str(parsed_args.max_model_len),
        "--tensor-parallel-size",
        str(parsed_args.tp_size),
        "--enable-prefix-caching",
        "--max-num-seqs",
        str(max_concurrency),
        "--quantization",
        "compressed-tensors",
        "--async-scheduling",
        "--gpu_memory_utilization",
        "0.9",
        "--reasoning-config",
        '{"reasoning_start_str": "<|START_THINKING|>", '
        '"reasoning_end_str": "<|END_THINKING|>"}',
    ]

    # Add speculative decoding configuration if provided and not disabled
    if not parsed_args.disable_spec:
        default_spec_config = (
            f'{{"method": "eagle", "model": "{draft_model_name}", '
            f'"num_speculative_tokens": 3, '
            f'"draft_tensor_parallel_size": {parsed_args.tp_size}}}'
        )
        server_args.extend(["--speculative_config", default_spec_config])

    print(f"Starting vLLM server with model: {model_name}")
    print(f"Tensor parallel size: {parsed_args.tp_size}")
    print(f"Max model length: {parsed_args.max_model_len}")
    print(
        f"Speculative decoding: {'Disabled' if parsed_args.disable_spec else 'Enabled'}"
    )
    if not parsed_args.disable_spec:
        spec_config_display = (
            draft_model_name if draft_model_name else "Default (same model)"
        )
        print(f"Speculative config: {spec_config_display}")
    print(f"Server args: {server_args}")

    failures = []

    try:
        with RemoteOpenAIServer(
            model_name,
            server_args,
            env_dict={"VLLM_ENABLE_COHERE_AUTO_CONFIG": "1"},
        ) as server:
            print(f"Server started at http://{server.host}:{server.port}")

            for num_requests in concurrency_levels:
                print("\n" + "=" * 60)
                print(f"  Concurrency level: {num_requests}")
                print("=" * 60)

                # Sync streaming test
                print(f"\n--- SYNC cancellation @ concurrency {num_requests} ---")
                try:
                    test_streaming_request_cancellation(
                        server, model_name, num_requests
                    )
                    print(f"  PASSED sync @ concurrency {num_requests}")
                except Exception as e:
                    print(f"  FAILED sync @ concurrency {num_requests}: {e}")
                    import traceback

                    traceback.print_exc()
                    failures.append(f"sync@{num_requests}")

                # Async streaming test
                print(f"\n--- ASYNC cancellation @ concurrency {num_requests} ---")
                try:
                    test_async_streaming_request_cancellation_wrapper(
                        server, model_name, num_requests
                    )
                    print(f"  PASSED async @ concurrency {num_requests}")
                except Exception as e:
                    print(f"  FAILED async @ concurrency {num_requests}: {e}")
                    import traceback

                    traceback.print_exc()
                    failures.append(f"async@{num_requests}")

    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    if failures:
        print(f"SWEEP COMPLETED WITH FAILURES: {failures}")
        sys.exit(1)
    else:
        print(f"All concurrency levels passed: {concurrency_levels}")
    print("=" * 80)
