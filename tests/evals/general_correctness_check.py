#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
General correctness check for vLLM endpoints.

Evaluates a running vLLM server with:
1. GSM8K correctness (math reasoning accuracy - greedy and sampling)
2. Sampling diversity tests (temp, top_p, top_k, frequency_penalty, presence_penalty)
3. Verification prompts (objective answer checking)
4. Edge case handling
5. Structured outputs (JSON mode validation)

Usage:
    # Start vLLM server
    vllm serve meta-llama/Llama-3.1-8B --port 8000

    # Run evaluation
    python tests/evals/general_correctness_check.py \
        --base-url http://localhost:8000 \
        --model meta-llama/Llama-3.1-8B

    # Quick test with fewer samples
    python tests/evals/general_correctness_check.py \
        --base-url http://localhost:8000 \
        --model meta-llama/Llama-3.1-8B \
        --gsm8k-samples 20 \
        --sampling-prompts 10
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import GSM8K utilities
from tests.evals.gsm8k.gsm8k_eval import (
    INVALID,
    call_vllm_api,
    get_answer_value,
    load_gsm8k_data,
)


class GeneralCorrectnessChecker:
    """General correctness checker for vLLM endpoints."""

    def __init__(self, base_url: str, model_name: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._model_detected = False

    async def detect_model(self) -> str | None:
        """
        Auto-detect the model being served by querying /v1/models.

        Returns:
            Model name if detected, None otherwise
        """
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if models and len(models) > 0:
                        model_name = models[0].get("id")
                        self._model_detected = True
                        return model_name
        except Exception as e:
            print(f"Could not auto-detect model: {e}")
        return None

    async def check_health(self) -> bool:
        """Check if the endpoint is healthy."""
        try:
            # Health endpoint is at /health, not /v1/health
            # Remove /v1 suffix if present
            health_url = self.base_url.rstrip("/")
            if health_url.endswith("/v1"):
                health_url = health_url[:-3]
            health_url = f"{health_url}/health"

            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    health_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response,
            ):
                return response.status == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    async def evaluate_gsm8k_correctness(
        self,
        num_samples: int = 50,
        num_shots: int = 5,
        temperature: float = 0.0,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        """
        Run GSM8K correctness evaluation.

        Args:
            num_samples: Number of test samples to evaluate (max 1319)
            num_shots: Number of few-shot examples
            temperature: Sampling temperature
            seed: Random seed for reproducibility (None for no seed)

        Returns:
            Dictionary with accuracy, invalid_rate, and timing metrics
        """
        test_type = "Greedy" if temperature == 0.0 else f"Sampling (temp={temperature})"
        print(f"\n{'=' * 70}")
        print(f"GSM8K CORRECTNESS TEST - {test_type}")
        print(f"{'=' * 70}")
        print(
            f"Samples: {num_samples} | Few-shot: {num_shots} | "
            f"Temperature: {temperature}"
        )
        print()

        # Load GSM8K data
        train_data, test_data = load_gsm8k_data()
        num_samples = min(num_samples, len(test_data))

        # Build few-shot prompt
        few_shot_examples = ""
        for i in range(num_shots):
            few_shot_examples += (
                f"Question: {train_data[i]['question']}\n"
                f"Answer: {train_data[i]['answer']}\n\n"
            )

        # Prepare test questions
        questions = []
        labels = []
        for i in range(num_samples):
            questions.append(f"Question: {test_data[i]['question']}\nAnswer:")
            labels.append(get_answer_value(test_data[i]["answer"]))

        # Run evaluation
        async def get_answer(session: aiohttp.ClientSession, i: int) -> str:
            prompt = few_shot_examples + questions[i]
            # Use seed for greedy, None for sampling to get diversity
            use_seed = seed if temperature == 0.0 else None
            answer = await call_vllm_api(
                session=session,
                prompt=prompt,
                temperature=temperature,
                max_tokens=256,
                stop=["Question", "Assistant:", "<|separator|>"],
                url=self.base_url,
                seed=use_seed,
            )
            return answer

        tic = time.perf_counter()

        desc = f"GSM8K ({test_type})"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            tasks = [get_answer(session, i) for i in range(num_samples)]
            states = await tqdm.gather(*tasks, desc=desc)

        latency = time.perf_counter() - tic

        # Compute metrics
        preds = [get_answer_value(state) for state in states]
        accuracy = np.mean(np.array(preds) == np.array(labels))
        invalid_rate = np.mean(np.array(preds) == INVALID)

        # Print results
        print("\nResults:")
        correct_count = int(accuracy * num_samples)
        print(
            f"  Accuracy:      {accuracy:.1%} ({correct_count}/{num_samples} correct)"
        )
        print(f"  Invalid rate:  {invalid_rate:.1%}")
        print(f"  Latency:       {latency:.2f}s")
        print(f"  Throughput:    {num_samples / latency:.2f} questions/s")

        return {
            "test_name": "gsm8k_correctness",
            "num_samples": num_samples,
            "num_shots": num_shots,
            "temperature": temperature,
            "accuracy": float(accuracy),
            "invalid_rate": float(invalid_rate),
            "latency_s": float(latency),
            "throughput_qps": float(num_samples / latency),
        }

    async def evaluate_sampling_diversity(
        self,
        num_prompts: int = 30,
    ) -> dict[str, Any]:
        """
        Test various sampling parameters to ensure diversity works correctly.

        Tests:
        - Greedy decoding (temp=0.0)
        - Sampling with temp=1.0
        - Nucleus sampling (top_p=0.9)
        - Top-k sampling (top_k=50)
        - Combined top_p and top_k
        - Frequency penalty (penalizes repetition)
        - Presence penalty (penalizes token reuse)

        Args:
            num_prompts: Number of prompts to test per configuration

        Returns:
            Dictionary with sampling test results
        """
        print(f"\n{'=' * 70}")
        print("SAMPLING DIVERSITY TEST")
        print(f"{'=' * 70}")
        print(f"Prompts per config: {num_prompts}")
        print()

        sampling_configs = [
            {
                "name": "temp_0.0 (greedy)",
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "n_samples": 10,
            },
            {
                "name": "temp_1.0 (sampling)",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "n_samples": 2,
            },
            {
                "name": "top_p_0.9 (nucleus)",
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": -1,
                "n_samples": 2,
            },
            {
                "name": "top_k_50",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "n_samples": 2,
            },
            {
                "name": "top_p+top_k (combined)",
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "n_samples": 2,
            },
            {
                "name": "frequency_penalty",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "frequency_penalty": 0.5,
                "n_samples": 2,
            },
            {
                "name": "presence_penalty",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "presence_penalty": 0.5,
                "n_samples": 2,
            },
        ]

        # Get verification prompts (easy to verify) + diverse prompts
        verification_prompts = self._get_verification_prompts()
        verification_prompt_strs = [p["prompt"] for p in verification_prompts]

        # Add more diverse prompts if needed
        num_verification = len(verification_prompt_strs)
        if num_prompts > num_verification:
            diverse_prompts = self._get_diverse_prompts(num_prompts - num_verification)
            all_prompts = verification_prompt_strs + diverse_prompts
        else:
            all_prompts = verification_prompt_strs[:num_prompts]
            verification_prompts = verification_prompts[:num_prompts]

        all_results = []

        for config in sampling_configs:
            config_name = str(config["name"])
            n_samples = int(config["n_samples"])  # type: ignore[call-overload]

            print(f"Testing: {config_name}")

            # Build parameter display
            params_display = [f"Temperature: {config['temperature']}"]
            if config.get("top_p", 1.0) != 1.0:
                params_display.append(f"Top-p: {config['top_p']}")
            if config.get("top_k", -1) != -1:
                params_display.append(f"Top-k: {config['top_k']}")
            if config.get("frequency_penalty"):
                params_display.append(
                    f"Frequency penalty: {config['frequency_penalty']}"
                )
            if config.get("presence_penalty"):
                params_display.append(f"Presence penalty: {config['presence_penalty']}")
            print(f"  {', '.join(params_display)}")

            result = await self._test_sampling_config(
                prompts=all_prompts,
                config_name=config_name,
                temperature=float(config["temperature"]),  # type: ignore[arg-type]
                top_p=float(config.get("top_p", 1.0)),  # type: ignore[arg-type]
                top_k=int(config.get("top_k", -1)),  # type: ignore[call-overload]
                frequency_penalty=float(config["frequency_penalty"])  # type: ignore[arg-type]  # noqa: E501
                if config.get("frequency_penalty") is not None
                else None,
                presence_penalty=float(config["presence_penalty"])  # type: ignore[arg-type]  # noqa: E501
                if config.get("presence_penalty") is not None
                else None,
                n_samples=n_samples,
                max_tokens=50,
                verification_prompts=verification_prompts,
            )
            all_results.append(result)

            # Print summary
            unique = result["unique_responses"]
            total = result["total_responses"]

            # For greedy decoding, show consistency check
            if result["greedy_consistency"] is not None:
                status = "✓" if result["greedy_consistency"] else "✗"
                print(f"  Greedy consistency: {status}")

                # Show unique count per prompt
                unique_per_prompt = result.get("greedy_unique_per_prompt", [])
                if unique_per_prompt:
                    print(f"    Unique per prompt: {unique_per_prompt}")

                if result["greedy_consistency"]:
                    print(f"    (all prompts consistent across {n_samples} samples)")
                else:
                    inconsistent_count = sum(1 for n in unique_per_prompt if n > 1)
                    print(
                        f"    ({inconsistent_count}/{len(unique_per_prompt)} "
                        f"prompts inconsistent)"
                    )
            else:
                # For sampling, show diversity
                print(f"  Response diversity: {unique}/{total} unique")

            # Show quality check results
            quality = result.get("quality_check", {})
            if quality.get("has_issues"):
                print(f"  ⚠ Quality issues: {', '.join(quality['issues'])}")

                # Print failed verification prompts
                failed_verifs = quality.get("failed_verifications", [])
                if failed_verifs:
                    print("  Failed verification prompts:")
                    for fail in failed_verifs:
                        print(f"    • Prompt: {fail['prompt']}")
                        resp = fail["response"][:80]
                        suffix = "..." if len(fail["response"]) > 80 else ""
                        print(f"      Response: {resp}{suffix}")
            else:
                verified_count = (
                    len(verification_prompts) if verification_prompts else 0
                )
                if verified_count > 0:
                    print(f"  ✓ All {verified_count} verification prompts passed")

            print()

        return {
            "test_name": "sampling_diversity",
            "num_prompts": num_prompts,
            "configs": all_results,
        }

    async def _test_sampling_config(
        self,
        prompts: list[str],
        config_name: str,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n_samples: int = 2,
        max_tokens: int = 50,
        verification_prompts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Test a single sampling configuration with quality verification."""

        async def generate_sample(session: aiohttp.ClientSession, prompt: str) -> str:
            # For greedy, use seed. For sampling, don't use seed for diversity
            seed = 42 if temperature == 0.0 else None

            # Build request payload
            async with session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name or "unknown",
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    **(
                        {"frequency_penalty": frequency_penalty}
                        if frequency_penalty is not None
                        else {}
                    ),
                    **(
                        {"presence_penalty": presence_penalty}
                        if presence_penalty is not None
                        else {}
                    ),
                    **({"seed": seed} if seed is not None else {}),
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                data = await response.json()
                if "choices" not in data or len(data["choices"]) == 0:
                    return ""
                return data["choices"][0]["text"]

        all_responses = []

        tic = time.perf_counter()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            tasks = []
            for prompt in prompts:
                for _ in range(n_samples):
                    task = generate_sample(session, prompt)
                    tasks.append(task)

            responses = await asyncio.gather(*tasks)
            all_responses = responses

        latency = time.perf_counter() - tic

        # Analyze responses
        unique_responses = len(set(all_responses))
        total_responses = len(all_responses)

        # For greedy decoding, all samples for same prompt should be identical
        greedy_consistency = None
        greedy_unique_per_prompt = None
        if temperature == 0.0:
            responses_per_prompt = []

            for i in range(0, len(all_responses), n_samples):
                prompt_responses = all_responses[i : i + n_samples]
                unique_count = len(set(prompt_responses))
                responses_per_prompt.append(unique_count)

            greedy_unique_per_prompt = responses_per_prompt
            # All responses for each prompt should be identical
            greedy_consistency = all(n == 1 for n in responses_per_prompt)

        # Quality checks using verification prompts
        # Check first response per prompt (not all n_samples)
        verification_responses = []
        for i in range(0, min(len(prompts) * n_samples, len(all_responses)), n_samples):
            verification_responses.append(all_responses[i])

        quality_check = self._check_response_quality(
            responses=verification_responses,
            verification_prompts=verification_prompts,
        )

        result = {
            "config_name": config_name,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "total_responses": total_responses,
            "unique_responses": unique_responses,
            "diversity_ratio": unique_responses / total_responses,
            "latency_s": float(latency),
            "throughput_qps": float(total_responses / latency),
            "greedy_consistency": greedy_consistency,
            "greedy_unique_per_prompt": greedy_unique_per_prompt,
            "quality_check": quality_check,
        }

        # Add penalty parameters if used
        if frequency_penalty is not None:
            result["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            result["presence_penalty"] = presence_penalty

        return result

    async def evaluate_edge_cases(self) -> dict[str, Any]:
        """
        Test edge cases and special scenarios.

        Tests:
        - Empty prompts
        - Single token generation
        - Stop sequences
        - Special characters
        """
        print(f"\n{'=' * 70}")
        print("EDGE CASE TESTS")
        print(f"{'=' * 70}")
        print()

        test_cases = [
            {
                "name": "empty_prompt",
                "prompt": "",
                "max_tokens": 20,
                "should_succeed": True,
            },
            {
                "name": "single_token",
                "prompt": "Hello",
                "max_tokens": 1,
                "should_succeed": True,
            },
            {
                "name": "with_stop_sequence",
                "prompt": "Count to 10: 1, 2, 3,",
                "max_tokens": 100,
                "stop": [","],
                "should_succeed": True,
            },
            {
                "name": "unicode_characters",
                "prompt": "Translate: こんにちは 你好 مرحبا",
                "max_tokens": 30,
                "should_succeed": True,
            },
        ]

        results = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            for test_case in test_cases:
                print(f"Testing: {test_case['name']:20s} ... ", end="", flush=True)

                try:
                    response = await call_vllm_api(
                        session=session,
                        prompt=test_case["prompt"],
                        temperature=0.0,
                        max_tokens=test_case["max_tokens"],
                        stop=test_case.get("stop"),
                        url=self.base_url,
                        seed=42,
                    )

                    success = True
                    error = None
                    output_len = len(response.split())
                    print(f"✓ (output: {output_len} tokens)")

                except Exception as e:
                    success = False
                    error = str(e)
                    output_len = 0
                    print(f"✗ {error}")

                results.append(
                    {
                        "test_case": test_case["name"],
                        "success": success,
                        "expected_success": test_case["should_succeed"],
                        "error": error,
                        "output_len": output_len,
                    }
                )

        # Check if all tests passed as expected
        all_passed = all(r["success"] == r["expected_success"] for r in results)

        print()
        print(f"Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")

        return {
            "test_name": "edge_cases",
            "all_passed": all_passed,
            "test_cases": results,
        }

    async def evaluate_structured_outputs(
        self, num_samples: int = 10
    ) -> dict[str, Any]:
        """
        Test structured output generation (JSON mode).

        Tests:
        - Validates that model can generate valid JSON
        - Checks JSON adheres to provided schema structure
        - Tests with simple user profile schema

        Args:
            num_samples: Number of JSON generation attempts

        Returns:
            Dictionary with validation results
        """
        print(f"\n{'=' * 70}")
        print("STRUCTURED OUTPUT TEST (JSON Mode)")
        print(f"{'=' * 70}")
        print(f"Samples: {num_samples}")
        print()

        # Load JSON schema (reuse from benchmarks)
        benchmark_dir = Path(__file__).parent.parent.parent / "benchmarks"
        schema_path = benchmark_dir / "structured_schemas" / "structured_schema_1.json"

        if not schema_path.exists():
            print(f"⚠ Schema file not found: {schema_path}")
            return {
                "test_name": "structured_outputs",
                "error": "Schema file not found",
            }

        with open(schema_path) as f:
            schema = json.load(f)

        # Create prompt
        prompt = (
            f"Generate an example of a brief user profile given the "
            f"following schema: {json.dumps(schema)}"
        )

        print()

        # Generate samples
        async def generate_json(session: aiohttp.ClientSession) -> dict[str, Any]:
            try:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model_name or "unknown",
                        "prompt": prompt,
                        "temperature": 0.7,
                        "max_tokens": 256,
                        "extra_body": {
                            "guided_json": schema,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API error {response.status}: {error_text}",
                            "valid_json": False,
                            "has_required_fields": False,
                        }

                    data = await response.json()
                    if "choices" not in data or len(data["choices"]) == 0:
                        return {
                            "success": False,
                            "error": "No choices in response",
                            "valid_json": False,
                            "has_required_fields": False,
                        }

                    generated_text = data["choices"][0]["text"].strip()
                    raw_text = generated_text  # Save for debugging

                    # For thinking models: extract JSON using brace counting
                    # Find the LAST complete JSON object in the output
                    import json as json_module

                    parsed_json = None
                    valid_json = False
                    json_str = None

                    # Find all occurrences of { and try to extract valid JSON from each
                    for start_idx in range(len(generated_text) - 1, -1, -1):
                        if generated_text[start_idx] == "{":
                            # Count braces to find the matching closing brace
                            brace_count = 0
                            end_idx = -1
                            for i in range(start_idx, len(generated_text)):
                                if generated_text[i] == "{":
                                    brace_count += 1
                                elif generated_text[i] == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break

                            if end_idx > 0:
                                candidate = generated_text[start_idx:end_idx]
                                try:
                                    parsed_json = json_module.loads(candidate)
                                    json_str = candidate
                                    valid_json = True
                                    break  # Found valid JSON, stop searching
                                except json_module.JSONDecodeError:
                                    continue  # Try next {

                    if valid_json and parsed_json is not None:
                        # Check required fields
                        required_fields = schema.get("required", [])
                        has_required_fields = all(
                            field in parsed_json for field in required_fields
                        )

                        return {
                            "success": True,
                            "error": None,
                            "generated_text": json_str,
                            "raw_text": raw_text,
                            "parsed_json": parsed_json,
                            "valid_json": valid_json,
                            "has_required_fields": has_required_fields,
                        }
                    else:
                        # No valid JSON found
                        return {
                            "success": True,
                            "error": "No valid JSON found in output",
                            "generated_text": generated_text[:200]
                            if len(generated_text) > 200
                            else generated_text,
                            "raw_text": raw_text,
                            "valid_json": False,
                            "has_required_fields": False,
                        }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "valid_json": False,
                    "has_required_fields": False,
                }

        tic = time.perf_counter()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            tasks = [generate_json(session) for _ in range(num_samples)]
            # Use regular gather to avoid potential tqdm issues
            results = await asyncio.gather(*tasks)

        latency = time.perf_counter() - tic

        # Analyze results
        successful = sum(1 for r in results if r["success"])
        valid_json_count = sum(1 for r in results if r["valid_json"])
        has_required_count = sum(1 for r in results if r["has_required_fields"])

        valid_json_rate = valid_json_count / num_samples if num_samples > 0 else 0
        required_fields_rate = (
            has_required_count / num_samples if num_samples > 0 else 0
        )

        # Print results
        print("Results:")
        print(f"  Successful requests: {successful}/{num_samples}")
        print(
            f"  Valid JSON:          {valid_json_count}/{num_samples} "
            f"({valid_json_rate:.1%})"
        )
        print(
            f"  Has required fields: {has_required_count}/{num_samples} "
            f"({required_fields_rate:.1%})"
        )
        print(f"  Latency:             {latency:.2f}s")

        # Show status
        if valid_json_rate >= 0.9 and required_fields_rate >= 0.9:
            print("\n✓ Structured output test PASSED")
        else:
            print("\n⚠ Structured output test had issues")
            if valid_json_rate < 0.9:
                print(f"  - Low valid JSON rate: {valid_json_rate:.1%}")
            if required_fields_rate < 0.9:
                print("  - Missing required fields in some outputs")

        return {
            "test_name": "structured_outputs",
            "num_samples": num_samples,
            "successful_requests": successful,
            "valid_json_count": valid_json_count,
            "valid_json_rate": float(valid_json_rate),
            "has_required_fields_count": has_required_count,
            "required_fields_rate": float(required_fields_rate),
            "latency_s": float(latency),
            "sample_outputs": [
                r.get("generated_text", "") for r in results[:3]
            ],  # First 3 samples
        }

    def _get_diverse_prompts(self, num_prompts: int) -> list[str]:
        """Get a diverse set of prompts for testing."""
        prompt_templates = [
            "Explain the concept of",
            "Write a short story about",
            "What is the capital of",
            "How do you make",
            "List 5 reasons why",
            "Describe the process of",
            "What are the benefits of",
            "Compare and contrast",
            "Summarize the history of",
            "Provide a recipe for",
            "Tell me a joke about",
            "What is the meaning of",
            "How does",
            "Why is",
            "When should you",
            "Where can I find",
            "Who invented",
            "Which is better:",
            "Can you explain",
            "What would happen if",
        ]

        # Cycle through prompts if needed
        result = []
        for i in range(num_prompts):
            result.append(prompt_templates[i % len(prompt_templates)])
        return result

    def _get_verification_prompts(self) -> list[dict[str, Any]]:
        """
        Get prompts with verifiable outputs for quality checking.
        These have objective answers that are easy to verify.
        """
        return [
            {
                "prompt": "The capital of France is",
                "must_contain_any": [["paris"]],
                "type": "factual",
            },
            {
                "prompt": "2 + 2 =",
                "must_contain_any": [["4", "four"]],
                "type": "math",
            },
            {
                "prompt": "The freezing point of water in Celsius is",
                "must_contain_any": [["0", "zero"]],
                "type": "factual",
            },
            {
                "prompt": "10 - 5 =",
                "must_contain_any": [["5", "five"]],
                "type": "math",
            },
        ]

    def _check_response_quality(
        self,
        responses: list[str],
        verification_prompts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Check response quality using multiple methods.

        Args:
            responses: List of generated responses
            verification_prompts: Optional list of prompt dicts with expected outputs

        Returns:
            Dictionary with quality metrics and issues
        """
        issues = []
        failed_verifications = []

        # 1. Empty responses
        empty_count = sum(1 for r in responses if len(r.strip()) == 0)
        if empty_count > 0:
            issues.append(f"{empty_count}/{len(responses)} empty responses")

        # 2. Very short responses (likely truncated or failed)
        short_count = sum(1 for r in responses if len(r.strip()) < 5)
        if short_count > len(responses) * 0.3:
            issues.append(f"{short_count}/{len(responses)} very short responses")

        # 3. Prompt-based verification (if provided)
        verification_failures = 0
        if verification_prompts:
            for i, prompt_spec in enumerate(verification_prompts):
                if i >= len(responses):
                    break

                response = responses[i].lower()
                prompt_type = prompt_spec.get("type", "unknown")

                # Check if any of the expected patterns are present
                must_contain_any = prompt_spec.get("must_contain_any", [])
                found = False
                for pattern_group in must_contain_any:
                    if any(pattern.lower() in response for pattern in pattern_group):
                        found = True
                        break

                if not found and len(response.strip()) > 0:
                    verification_failures += 1
                    failed_verifications.append(
                        {
                            "prompt": prompt_spec.get("prompt", ""),
                            "response": responses[i],
                            "expected": must_contain_any,
                            "type": prompt_type,
                        }
                    )

        if verification_failures > 0:
            num_prompts = len(verification_prompts) if verification_prompts else 0
            issues.append(
                f"{verification_failures}/{num_prompts} verification prompts failed"
            )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "empty_count": empty_count,
            "verification_failures": verification_failures
            if verification_prompts
            else None,
            "failed_verifications": failed_verifications,
        }

    async def run_all_checks(
        self,
        gsm8k_samples: int = 100,
        sampling_prompts: int = 30,
        skip_sampling: bool = False,
        skip_edge_cases: bool = False,
        skip_structured: bool = False,
        structured_samples: int = 10,
        test_sampling_accuracy: bool = True,
    ) -> dict[str, Any]:
        """
        Run all correctness checks.

        Args:
            gsm8k_samples: Number of GSM8K samples to evaluate
            sampling_prompts: Number of prompts for sampling tests
            skip_sampling: Skip sampling diversity tests
            skip_edge_cases: Skip edge case tests
            skip_structured: Skip structured output tests
            structured_samples: Number of structured output samples
            test_sampling_accuracy: Test GSM8K accuracy with temp=1.0

        Returns:
            Dictionary with all test results
        """
        # Check health first
        print(f"Checking endpoint health: {self.base_url}")
        if not await self.check_health():
            print("✗ ERROR: Endpoint is not healthy!")
            return {
                "endpoint": self.base_url,
                "model": self.model_name or "unknown",
                "error": "Endpoint health check failed",
            }

        print("✓ Endpoint is healthy")

        # Auto-detect model if not provided
        if self.model_name is None:
            print("Detecting model name...")
            self.model_name = await self.detect_model()
            if self.model_name:
                print(f"✓ Detected model: {self.model_name}")
            else:
                print("⚠ Could not detect model name, using 'unknown'")
                self.model_name = "unknown"

        print()

        tests_dict: dict[str, Any] = {}
        all_results = {
            "endpoint": self.base_url,
            "model": self.model_name,
            "model_auto_detected": self._model_detected,
            "timestamp": time.time(),
            "tests": tests_dict,
        }

        # Test 1: GSM8K Correctness (Greedy)
        try:
            gsm8k_results = await self.evaluate_gsm8k_correctness(
                num_samples=gsm8k_samples, temperature=0.0
            )
            tests_dict["gsm8k_correctness_greedy"] = gsm8k_results
        except Exception as e:
            print(f"\n✗ GSM8K greedy test failed: {e}")
            tests_dict["gsm8k_correctness_greedy"] = {"error": str(e)}

        # Test 1b: GSM8K Correctness (Sampling with temp=1.0)
        if test_sampling_accuracy:
            try:
                gsm8k_sampling_results = await self.evaluate_gsm8k_correctness(
                    num_samples=gsm8k_samples, temperature=1.0
                )
                tests_dict["gsm8k_correctness_sampling"] = gsm8k_sampling_results
            except Exception as e:
                print(f"\n✗ GSM8K sampling test failed: {e}")
                tests_dict["gsm8k_correctness_sampling"] = {"error": str(e)}

        # Test 2: Sampling Diversity
        if not skip_sampling:
            try:
                sampling_results = await self.evaluate_sampling_diversity(
                    num_prompts=sampling_prompts
                )
                tests_dict["sampling_diversity"] = sampling_results
            except Exception as e:
                print(f"\n✗ Sampling diversity test failed: {e}")
                tests_dict["sampling_diversity"] = {"error": str(e)}

        # Test 3: Edge Cases
        if not skip_edge_cases:
            try:
                edge_case_results = await self.evaluate_edge_cases()
                tests_dict["edge_cases"] = edge_case_results
            except Exception as e:
                print(f"\n✗ Edge case test failed: {e}")
                tests_dict["edge_cases"] = {"error": str(e)}

        # Test 4: Structured Outputs (JSON mode)
        if not skip_structured:
            try:
                structured_results = await self.evaluate_structured_outputs(
                    num_samples=structured_samples
                )
                tests_dict["structured_outputs"] = structured_results
            except Exception as e:
                print(f"\n✗ Structured output test failed: {e}")
                tests_dict["structured_outputs"] = {"error": str(e)}

        return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary of all test results."""
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Endpoint: {results['endpoint']}")
    model_str = results["model"]
    if results.get("model_auto_detected"):
        model_str += " (auto-detected)"
    print(f"Model:    {model_str}")
    print()

    # Check for top-level error
    if "error" in results and "tests" not in results:
        print(f"✗ ERROR: {results['error']}")
        return

    tests = results.get("tests", {})

    # GSM8K Results - Greedy
    if "gsm8k_correctness_greedy" in tests:
        gsm8k = tests["gsm8k_correctness_greedy"]
        if "accuracy" in gsm8k:
            status = "✓" if gsm8k["accuracy"] >= 0.5 else "⚠"
            print(f"{status} GSM8K Accuracy (greedy):   {gsm8k['accuracy']:.1%}")
        else:
            print(f"✗ GSM8K (greedy): FAILED - {gsm8k.get('error', 'Unknown error')}")

    # GSM8K Results - Sampling
    if "gsm8k_correctness_sampling" in tests:
        gsm8k_sampling = tests["gsm8k_correctness_sampling"]
        if "accuracy" in gsm8k_sampling:
            status = "✓" if gsm8k_sampling["accuracy"] >= 0.5 else "⚠"
            temp = gsm8k_sampling.get("temperature", 1.0)
            acc = gsm8k_sampling["accuracy"]
            print(f"{status} GSM8K Accuracy (temp={temp}): {acc:.1%}")
        else:
            error = gsm8k_sampling.get("error", "Unknown error")
            print(f"✗ GSM8K (sampling): FAILED - {error}")

    # Sampling Results
    if "sampling_diversity" in tests:
        sampling = tests["sampling_diversity"]
        if "configs" in sampling:
            configs = sampling["configs"]
            # Count verification prompt passes across all configs
            total_prompts = 0
            passed_prompts = 0
            for c in configs:
                quality = c.get("quality_check", {})
                # Assume 4 verification prompts per config
                num_verif = 4
                failures = quality.get("verification_failures", 0) or 0
                total_prompts += num_verif
                passed_prompts += num_verif - failures

            if passed_prompts == total_prompts:
                print(
                    f"✓ Sampling Tests: {passed_prompts}/{total_prompts} "
                    f"verification prompts passed"
                )
            else:
                print(
                    f"⚠ Sampling Tests: {passed_prompts}/{total_prompts} "
                    f"verification prompts passed"
                )
        else:
            print(f"✗ Sampling: FAILED - {sampling.get('error', 'Unknown error')}")

    # Edge Case Results
    if "edge_cases" in tests:
        edge = tests["edge_cases"]
        if "all_passed" in edge:
            status = "✓" if edge["all_passed"] else "✗"
            num_tests = len(edge["test_cases"])
            result_str = "passed" if edge["all_passed"] else "failed"
            print(f"{status} Edge Cases: {num_tests} tests {result_str}")
        else:
            error = edge.get("error", "Unknown error")
            print(f"✗ Edge Cases: FAILED - {error}")

    # Structured Output Results
    if "structured_outputs" in tests:
        structured = tests["structured_outputs"]
        if "valid_json_rate" in structured:
            valid_rate = structured["valid_json_rate"]
            required_rate = structured["required_fields_rate"]
            status = "✓" if valid_rate >= 0.9 and required_rate >= 0.9 else "⚠"
            print(
                f"{status} Structured Outputs: {valid_rate:.0%} valid JSON, "
                f"{required_rate:.0%} with required fields"
            )
        else:
            error = structured.get("error", "Unknown error")
            print(f"✗ Structured Outputs: FAILED - {error}")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="General correctness check for vLLM endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model from endpoint
  python tests/evals/general_correctness_check.py

  # Specify endpoint (model auto-detected)
  python tests/evals/general_correctness_check.py \\
      --base-url http://localhost:8000

  # Specify both endpoint and model
  python tests/evals/general_correctness_check.py \\
      --base-url http://localhost:8000 \\
      --model meta-llama/Llama-3.1-8B

  # Quick test with fewer samples
  python tests/evals/general_correctness_check.py \\
      --gsm8k-samples 20 \\
      --sampling-prompts 10

  # Save results to file
  python tests/evals/general_correctness_check.py \\
      --output results.json
        """,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name being served (default: auto-detect from endpoint)",
    )
    parser.add_argument(
        "--gsm8k-samples",
        type=int,
        default=100,
        help="Number of GSM8K samples to evaluate, max 1319 (default: 100)",
    )
    parser.add_argument(
        "--sampling-prompts",
        type=int,
        default=30,
        help="Number of prompts for sampling tests (default: 30)",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip sampling diversity tests",
    )
    parser.add_argument(
        "--skip-edge-cases",
        action="store_true",
        help="Skip edge case tests",
    )
    parser.add_argument(
        "--skip-sampling-accuracy",
        action="store_true",
        help="Skip GSM8K accuracy test with temperature=1.0",
    )
    parser.add_argument(
        "--skip-structured",
        action="store_true",
        help="Skip structured output (JSON mode) tests",
    )
    parser.add_argument(
        "--structured-samples",
        type=int,
        default=10,
        help="Number of structured output samples to test (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)",
    )

    args = parser.parse_args()

    # Run evaluation
    checker = GeneralCorrectnessChecker(args.base_url, args.model)

    results = asyncio.run(
        checker.run_all_checks(
            gsm8k_samples=args.gsm8k_samples,
            sampling_prompts=args.sampling_prompts,
            skip_sampling=args.skip_sampling,
            skip_edge_cases=args.skip_edge_cases,
            skip_structured=args.skip_structured,
            structured_samples=args.structured_samples,
            test_sampling_accuracy=not args.skip_sampling_accuracy,
        )
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
