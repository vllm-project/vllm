#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
General correctness check for vLLM endpoints.

Evaluates a running vLLM server with:
1. GSM8K correctness (math reasoning - greedy and sampling)
2. Sampling diversity tests
3. Edge case handling
4. Structured outputs (JSON mode)

Usage:
    vllm serve meta-llama/Llama-3.1-8B --port 8000

    # Normal mode (only summary)
    python tests/evals/general_correctness_check.py \
        --base-url http://localhost:8000 \
        --gsm8k-samples 100

    # Verbose mode (detailed progress)
    python tests/evals/general_correctness_check.py \
        --base-url http://localhost:8000 \
        --gsm8k-samples 100 \
        --verbose
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import structured output utilities from shared module
from benchmarks.structured_output_utils import (
    generate_json_prompt,
    load_json_schema,
    validate_json_with_schema,
)
from tests.evals.gsm8k.gsm8k_eval import call_vllm_api, evaluate_gsm8k

# Simple verification prompts
VERIFICATION_PROMPTS = [
    {"prompt": "The capital of France is", "must_contain": ["paris"]},
    {"prompt": "2 + 2 =", "must_contain": ["4", "four"]},
    {"prompt": "10 - 5 =", "must_contain": ["5", "five"]},
]


async def check_health(base_url: str) -> bool:
    """Check if the endpoint is healthy."""
    try:
        health_url = base_url.rstrip("/")
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
    except Exception:
        return False


async def detect_model(base_url: str) -> str | None:
    """Auto-detect the model being served."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)
            ) as response,
        ):
            if response.status == 200:
                data = await response.json()
                models = data.get("data", [])
                return models[0].get("id") if models else None
            return None
    except Exception:
        return None


def run_gsm8k_evaluation(
    base_url: str, num_samples: int, temperature: float, verbose: bool = True
) -> dict[str, Any]:
    """Run GSM8K evaluation using existing utilities."""
    test_type = "Greedy" if temperature == 0.0 else f"Sampling (temp={temperature})"
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"GSM8K CORRECTNESS - {test_type}")
        print(f"{'=' * 70}")

    # Extract host and port from base_url
    base_url = base_url.rstrip("/")
    if "://" in base_url:
        protocol_and_rest = base_url.split("://", 1)
        rest = (
            protocol_and_rest[1] if len(protocol_and_rest) > 1 else protocol_and_rest[0]
        )
    else:
        rest = base_url

    if ":" in rest:
        host_part, port_part = rest.rsplit(":", 1)
        host = (
            f"http://{host_part}"
            if "://" not in base_url
            else f"{protocol_and_rest[0]}://{host_part}"
        )
        port = int(port_part)
    else:
        host = base_url
        port = 8000

    # Use existing evaluate_gsm8k function
    seed = 42 if temperature == 0.0 else None
    result = evaluate_gsm8k(
        num_questions=num_samples,
        num_shots=5,
        max_tokens=256,
        host=host,
        port=port,
        temperature=temperature,
        seed=seed,
    )

    if verbose:
        print(f"\nAccuracy:     {result['accuracy']:.1%}")
        print(f"Invalid rate: {result['invalid_rate']:.1%}")
        print(f"Latency:      {result['latency']:.2f}s")

    return {
        "test_name": "gsm8k_correctness",
        "temperature": temperature,
        "accuracy": float(result["accuracy"]),
        "invalid_rate": float(result["invalid_rate"]),
        "latency_s": float(result["latency"]),
    }


async def evaluate_sampling_diversity(
    base_url: str, model_name: str, num_prompts: int = 4, verbose: bool = True
) -> dict[str, Any]:
    """Test sampling parameters."""
    if verbose:
        print(f"\n{'=' * 70}")
        print("SAMPLING DIVERSITY")
        print(f"{'=' * 70}")

    configs = [
        {"name": "greedy", "temperature": 0.0, "n_samples": 5},
        {"name": "temp_1.0", "temperature": 1.0, "n_samples": 2},
        {"name": "top_p_0.9", "temperature": 1.0, "top_p": 0.9, "n_samples": 2},
    ]

    prompts: list[str] = [str(p["prompt"]) for p in VERIFICATION_PROMPTS[:num_prompts]]
    results = []

    for config in configs:
        result = await test_sampling_config(base_url, model_name, prompts, config)
        results.append(result)

        if verbose:
            print(f"\n{config['name']}:")
            if config["temperature"] == 0.0:
                print(f"  Consistent: {'✓' if result['greedy_consistency'] else '✗'}")
            else:
                unique = result["unique_responses"]
                total = result["total_responses"]
                print(f"  Diversity: {unique}/{total}")

            if result.get("verification_failures", 0) > 0:
                print(f"  ⚠ {result['verification_failures']} prompts failed")

    return {"test_name": "sampling_diversity", "configs": results}


async def test_sampling_config(
    base_url: str, model_name: str, prompts: list[str], config: dict
) -> dict[str, Any]:
    """Test a single sampling configuration."""
    temperature = config["temperature"]
    n_samples = config["n_samples"]

    async def generate(session: aiohttp.ClientSession, prompt: str) -> str:
        seed = 42 if temperature == 0.0 else None
        async with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": 50,
                "top_p": config.get("top_p", 1.0),
                "top_k": config.get("top_k", -1),
                **({"seed": seed} if seed else {}),
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            if response.status != 200:
                return ""
            data = await response.json()
            return data["choices"][0]["text"] if data.get("choices") else ""

    async with aiohttp.ClientSession() as session:
        tasks = [generate(session, p) for p in prompts for _ in range(n_samples)]
        responses = await asyncio.gather(*tasks)

    unique_responses = len(set(responses))
    total_responses = len(responses)

    # Check greedy consistency
    greedy_consistency = None
    if temperature == 0.0:
        responses_per_prompt = [
            len(set(responses[i : i + n_samples]))
            for i in range(0, len(responses), n_samples)
        ]
        greedy_consistency = all(n == 1 for n in responses_per_prompt)

    # Verify expected answers
    verification_failures = 0
    for i, prompt_spec in enumerate(VERIFICATION_PROMPTS[: len(prompts)]):
        response = responses[i * n_samples].lower()
        if not any(exp.lower() in response for exp in prompt_spec["must_contain"]):
            verification_failures += 1

    return {
        "config_name": config["name"],
        "temperature": temperature,
        "unique_responses": unique_responses,
        "total_responses": total_responses,
        "greedy_consistency": greedy_consistency,
        "verification_failures": verification_failures,
    }


async def evaluate_edge_cases(base_url: str, verbose: bool = True) -> dict[str, Any]:
    """Test edge cases."""
    if verbose:
        print(f"\n{'=' * 70}")
        print("EDGE CASES")
        print(f"{'=' * 70}\n")

    test_cases = [
        {"name": "empty_prompt", "prompt": "", "max_tokens": 20},
        {"name": "single_token", "prompt": "Hello", "max_tokens": 1},
        {
            "name": "stop_sequence",
            "prompt": "Count: 1, 2, 3,",
            "max_tokens": 100,
            "stop": [","],
        },
        {"name": "unicode", "prompt": "Translate: こんにちは 你好", "max_tokens": 30},
    ]

    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60)
    ) as session:
        for test in test_cases:
            if verbose:
                print(f"Testing: {test['name']:20s} ... ", end="", flush=True)
            try:
                await call_vllm_api(
                    session=session,
                    prompt=test["prompt"],  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=test["max_tokens"],  # type: ignore[arg-type]
                    stop=test.get("stop"),  # type: ignore[arg-type]
                    url=base_url,
                    seed=42,
                )
                if verbose:
                    print("✓")
                results.append({"test_case": test["name"], "success": True})
            except Exception as e:
                if verbose:
                    print(f"✗ {e}")
                results.append({"test_case": test["name"], "success": False})

    all_passed = all(r["success"] for r in results)
    if verbose:
        print(f"\nOverall: {'✓ PASSED' if all_passed else '✗ FAILED'}")

    return {"test_name": "edge_cases", "all_passed": all_passed, "test_cases": results}


async def evaluate_structured_outputs(
    base_url: str,
    model_name: str,
    num_samples: int = 10,
    schema_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test structured output (JSON mode) using benchmark utilities."""
    if verbose:
        print(f"\n{'=' * 70}")
        print("STRUCTURED OUTPUT (JSON)")
        print(f"{'=' * 70}\n")

    # Load schema using benchmark utility function
    try:
        schema = load_json_schema(schema_path)
    except (FileNotFoundError, Exception) as e:
        return {"test_name": "structured_outputs", "error": str(e)}

    # Generate prompt using benchmark utility function
    prompt = generate_json_prompt(schema)

    # Track failures for detailed logging
    failures = []

    async def generate_json(
        session: aiohttp.ClientSession, sample_idx: int
    ) -> dict[str, Any]:
        try:
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "extra_body": {"guided_json": schema},
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_msg = f"HTTP {response.status}"
                    failures.append(
                        {"sample": sample_idx, "error": error_msg, "response": None}
                    )
                    return {"valid_json": False, "has_required_fields": False}

                data = await response.json()
                if not data.get("choices"):
                    failures.append(
                        {
                            "sample": sample_idx,
                            "error": "No choices in response",
                            "response": None,
                        }
                    )
                    return {"valid_json": False, "has_required_fields": False}

                text = data["choices"][0]["text"].strip()

                # Use benchmark validation utility plus required field check
                result = validate_json_with_schema(text, schema)

                # Log failures
                if not result["valid_json"]:
                    failures.append(
                        {
                            "sample": sample_idx,
                            "error": "Invalid JSON",
                            "response": text[:200]
                            if len(text) > 200
                            else text,  # Truncate long responses
                        }
                    )
                elif not result["has_required_fields"]:
                    failures.append(
                        {
                            "sample": sample_idx,
                            "error": "Missing required fields",
                            "response": text[:200] if len(text) > 200 else text,
                        }
                    )

                return result
        except Exception as e:
            failures.append(
                {
                    "sample": sample_idx,
                    "error": f"Exception: {str(e)}",
                    "response": None,
                }
            )
            return {"valid_json": False, "has_required_fields": False}

    async with aiohttp.ClientSession() as session:
        tasks = [generate_json(session, i) for i in range(num_samples)]
        results = await asyncio.gather(*tasks)

    valid_count = sum(r["valid_json"] for r in results)
    required_count = sum(r["has_required_fields"] for r in results)

    valid_rate = valid_count / num_samples
    required_rate = required_count / num_samples

    if verbose:
        print(f"Valid JSON:          {valid_count}/{num_samples} ({valid_rate:.1%})")
        print(
            f"Has required fields: {required_count}/{num_samples} ({required_rate:.1%})"
        )

        # Show detailed failure information if any failures occurred
        if failures:
            print(f"\nFailure details ({len(failures)} samples):")
            for failure in failures[:5]:  # Show first 5 failures
                print(f"  Sample {failure['sample']}: {failure['error']}")
                if failure["response"]:
                    print(f"    Response: {failure['response']}")
            if len(failures) > 5:
                print(f"  ... and {len(failures) - 5} more failures")

        status = (
            "✓ PASSED" if valid_rate >= 0.9 and required_rate >= 0.9 else "⚠ ISSUES"
        )
        print(f"\n{status}")

    return {
        "test_name": "structured_outputs",
        "valid_json_count": valid_count,
        "valid_json_rate": float(valid_rate),
        "required_fields_count": required_count,
        "required_fields_rate": float(required_rate),
        "failures": failures if failures else None,
    }


def run_all_checks_sync(
    base_url: str,
    model_name: str | None,
    gsm8k_samples: int,
    sampling_prompts: int,
    skip_sampling: bool,
    skip_edge_cases: bool,
    skip_structured: bool,
    structured_samples: int,
    test_sampling_accuracy: bool,
    json_schema_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all correctness checks."""
    if verbose:
        print(f"Checking endpoint: {base_url}")

    # Health check and model detection (run in new event loop)
    async def setup():
        if not await check_health(base_url):
            return None, False

        detected = False
        model = model_name
        if model is None:
            model = await detect_model(base_url)
            if model:
                detected = True
            else:
                model = "unknown"
        return model, detected

    model_name, model_detected = asyncio.run(setup())

    if model_name is None:
        return {"endpoint": base_url, "error": "Endpoint health check failed"}

    if verbose:
        print("✓ Endpoint healthy")
        if model_detected:
            print(f"✓ Detected model: {model_name}")

    tests_dict: dict[str, Any] = {}

    # GSM8K - Greedy (runs in its own event loop)
    try:
        tests_dict["gsm8k_greedy"] = run_gsm8k_evaluation(
            base_url, gsm8k_samples, temperature=0.0, verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"✗ GSM8K greedy failed: {e}")
        tests_dict["gsm8k_greedy"] = {"error": str(e)}

    # GSM8K - Sampling (runs in its own event loop)
    if test_sampling_accuracy:
        try:
            tests_dict["gsm8k_sampling"] = run_gsm8k_evaluation(
                base_url, gsm8k_samples, temperature=1.0, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"✗ GSM8K sampling failed: {e}")
            tests_dict["gsm8k_sampling"] = {"error": str(e)}

    # Async tests (run in new event loop)
    async def run_async_tests():
        async_tests = {}

        # Sampling diversity
        if not skip_sampling:
            try:
                async_tests["sampling_diversity"] = await evaluate_sampling_diversity(
                    base_url, model_name, sampling_prompts, verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Sampling diversity failed: {e}")
                async_tests["sampling_diversity"] = {"error": str(e)}

        # Edge cases
        if not skip_edge_cases:
            try:
                async_tests["edge_cases"] = await evaluate_edge_cases(
                    base_url, verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Edge cases failed: {e}")
                async_tests["edge_cases"] = {"error": str(e)}

        # Structured outputs
        if not skip_structured:
            try:
                async_tests["structured_outputs"] = await evaluate_structured_outputs(
                    base_url,
                    model_name,
                    structured_samples,
                    json_schema_path,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Structured outputs failed: {e}")
                async_tests["structured_outputs"] = {"error": str(e)}

        return async_tests

    # Run async tests
    async_results = asyncio.run(run_async_tests())
    tests_dict.update(async_results)

    return {
        "endpoint": base_url,
        "model": model_name,
        "model_auto_detected": model_detected,
        "timestamp": time.time(),
        "tests": tests_dict,
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print summary of all test results."""
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Endpoint: {results['endpoint']}")
    print(f"Model:    {results['model']}")
    if results.get("model_auto_detected"):
        print("          (auto-detected)")

    if "error" in results:
        print(f"\n✗ ERROR: {results['error']}")
        return

    tests = results.get("tests", {})

    # GSM8K
    for key in ["gsm8k_greedy", "gsm8k_sampling"]:
        if key in tests:
            test = tests[key]
            label = "GSM8K (greedy)" if "greedy" in key else "GSM8K (sampling)"
            if "accuracy" in test:
                status = "✓" if test["accuracy"] >= 0.5 else "⚠"
                print(f"\n{status} {label}: {test['accuracy']:.1%}")
            else:
                print(f"\n✗ {label}: {test.get('error')}")

    # Sampling
    if "sampling_diversity" in tests:
        test = tests["sampling_diversity"]
        if "configs" in test:
            failures = sum(c.get("verification_failures", 0) for c in test["configs"])
            total = len(test["configs"]) * len(VERIFICATION_PROMPTS)
            status = "✓" if failures == 0 else "⚠"
            print(f"\n{status} Sampling: {total - failures}/{total} prompts passed")
        else:
            print(f"\n✗ Sampling: {test.get('error')}")

    # Edge cases
    if "edge_cases" in tests:
        test = tests["edge_cases"]
        if "all_passed" in test:
            status = "✓" if test["all_passed"] else "✗"
            print(
                f"\n{status} Edge cases: {'passed' if test['all_passed'] else 'failed'}"
            )
        else:
            print(f"\n✗ Edge cases: {test.get('error')}")

    # Structured
    if "structured_outputs" in tests:
        test = tests["structured_outputs"]
        if "valid_json_rate" in test:
            valid = test["valid_json_rate"]
            required = test["required_fields_rate"]
            status = "✓" if valid >= 0.9 and required >= 0.9 else "⚠"
            print(f"\n{status} Structured: {valid:.0%} valid, {required:.0%} complete")
        else:
            print(f"\n✗ Structured: {test.get('error')}")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="General correctness check for vLLM")
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument(
        "--model", type=str, help="Model name (auto-detected if omitted)"
    )
    parser.add_argument("--gsm8k-samples", type=int, default=100, help="GSM8K samples")
    parser.add_argument(
        "--sampling-prompts", type=int, default=4, help="Sampling test prompts"
    )
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-edge-cases", action="store_true")
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-sampling-accuracy", action="store_true")
    parser.add_argument("--structured-samples", type=int, default=10)
    parser.add_argument("--json-schema-path", type=str, help="Path to JSON schema file")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (default: only show summary)",
    )

    args = parser.parse_args()

    results = run_all_checks_sync(
        base_url=args.base_url,
        model_name=args.model,
        gsm8k_samples=args.gsm8k_samples,
        sampling_prompts=args.sampling_prompts,
        skip_sampling=args.skip_sampling,
        skip_edge_cases=args.skip_edge_cases,
        skip_structured=args.skip_structured,
        structured_samples=args.structured_samples,
        test_sampling_accuracy=not args.skip_sampling_accuracy,
        json_schema_path=args.json_schema_path,
        verbose=args.verbose,
    )

    print_summary(results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
