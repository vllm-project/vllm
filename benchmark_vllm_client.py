#!/usr/bin/env python3
"""
vLLM Client Performance Benchmark Script
Sends concurrent requests to vLLM v1/completions endpoint and measures QPS.
"""

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests
from datetime import datetime
import statistics
import sys


class VLLMBenchmarkClient:
    """Client for benchmarking vLLM service performance."""

    def __init__(self,
                 base_url: str = "http://localhost:8000",
                 max_tokens: int = 128,
                 temperature: float = 0.7,
                 timeout: int = 60,
                 show_samples: int = 0):
        """
        Initialize the benchmark client.

        Args:
            base_url: Base URL of the vLLM service
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            show_samples: Number of sample outputs to display (0 to disable)
        """
        self.base_url = base_url.rstrip('/')
        self.completions_url = f"{self.base_url}/v1/completions"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.show_samples = show_samples
        self.session = requests.Session()
        self.output_samples = []  # Store sample outputs

    def load_prompts(self, file_path: str) -> List[str]:
        """
        Load prompts from JSONL file.

        Args:
            file_path: Path to the JSONL file containing prompts

        Returns:
            List of prompt strings
        """
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        prompts.append(data['prompt'])
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSONL file: {e}")
            sys.exit(1)
        except KeyError:
            print("Error: JSONL file must contain 'prompt' field in each line")
            sys.exit(1)

        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts

    def send_request(self, prompt: str, request_id: int) -> Dict[str, Any]:
        """
        Send a single request to vLLM service.

        Args:
            prompt: The prompt to send
            request_id: Unique identifier for this request

        Returns:
            Dictionary containing request metrics
        """
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }

        start_time = time.time()
        result = {
            "request_id": request_id,
            "start_time": start_time,
            "success": False,
            "status_code": None,
            "error": None,
            "response_time": None,
            "tokens_generated": 0
        }

        try:
            response = self.session.post(
                self.completions_url,
                json=payload,
                timeout=self.timeout
            )
            end_time = time.time()
            result["response_time"] = end_time - start_time
            result["status_code"] = response.status_code

            if response.status_code == 200:
                response_data = response.json()
                result["success"] = True

                # Extract the generated text
                generated_text = ""
                if "choices" in response_data and response_data["choices"]:
                    generated_text = response_data["choices"][0].get("text", "")
                    result["generated_text"] = generated_text  # Save the actual output
                    result["prompt"] = prompt  # Also save the prompt for reference

                # Extract token count if available
                if "usage" in response_data:
                    result["tokens_generated"] = response_data["usage"].get("completion_tokens", 0)
                elif generated_text:
                    # Rough estimate based on response length
                    result["tokens_generated"] = len(generated_text.split())
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

        except requests.exceptions.Timeout:
            result["error"] = "Request timeout"
            result["response_time"] = self.timeout
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection error - is vLLM service running?"
            result["response_time"] = time.time() - start_time
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            result["response_time"] = time.time() - start_time

        return result

    def run_benchmark(self,
                     prompts: List[str],
                     max_workers: int = 128) -> Dict[str, Any]:
        """
        Run the benchmark with concurrent requests.

        Args:
            prompts: List of prompts to send
            max_workers: Maximum number of concurrent threads

        Returns:
            Dictionary containing benchmark results
        """
        print(f"\nStarting benchmark with {len(prompts)} prompts and {max_workers} workers")
        print(f"Target URL: {self.completions_url}")
        print(f"Max tokens per request: {self.max_tokens}")
        print("-" * 60)

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(self.send_request, prompt, i): (i, prompt)
                for i, prompt in enumerate(prompts)
            }

            # Process results as they complete
            completed = 0
            successful = 0
            failed = 0

            for future in as_completed(future_to_prompt):
                result = future.result()
                results.append(result)
                completed += 1

                if result["success"]:
                    successful += 1
                    # Collect sample outputs if requested
                    if self.show_samples > 0 and len(self.output_samples) < self.show_samples:
                        if "generated_text" in result:
                            self.output_samples.append({
                                "request_id": result["request_id"],
                                "prompt": result.get("prompt", "")[:],  # First 200 chars
                                "output": result["generated_text"][:],  # First 500 chars
                                "tokens": result.get("tokens_generated", 0),
                                "time": result.get("response_time", 0)
                            })
                else:
                    failed += 1

                # Progress update every 10 requests or at the end
                if completed % 10 == 0 or completed == len(prompts):
                    elapsed = time.time() - start_time
                    current_qps = completed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed}/{len(prompts)} | "
                          f"Success: {successful} | Failed: {failed} | "
                          f"Current QPS: {current_qps:.2f}")

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate statistics
        stats = self.calculate_statistics(results, total_time)
        return stats

    def calculate_statistics(self,
                           results: List[Dict[str, Any]],
                           total_time: float) -> Dict[str, Any]:
        """
        Calculate benchmark statistics.

        Args:
            results: List of request results
            total_time: Total benchmark duration in seconds

        Returns:
            Dictionary containing statistics
        """
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        response_times = [r["response_time"] for r in successful_results if r["response_time"]]

        stats = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "total_time_seconds": total_time,
            "end_to_end_qps": len(results) / total_time if total_time > 0 else 0,
            "successful_qps": len(successful_results) / total_time if total_time > 0 else 0,
            "success_rate": len(successful_results) / len(results) * 100 if results else 0,
        }

        if response_times:
            stats.update({
                "avg_latency_seconds": statistics.mean(response_times),
                "median_latency_seconds": statistics.median(response_times),
                "min_latency_seconds": min(response_times),
                "max_latency_seconds": max(response_times),
            })

            # Calculate percentiles only if we have at least 2 data points
            if len(response_times) >= 2:
                quantiles = statistics.quantiles(response_times, n=100)
                stats.update({
                    "p50_latency_seconds": quantiles[49],
                    "p90_latency_seconds": quantiles[89],
                    "p95_latency_seconds": quantiles[94],
                    "p99_latency_seconds": quantiles[98],
                })
            else:
                # For single data point, use that value for all percentiles
                single_value = response_times[0]
                stats.update({
                    "p50_latency_seconds": single_value,
                    "p90_latency_seconds": single_value,
                    "p95_latency_seconds": single_value,
                    "p99_latency_seconds": single_value,
                })

            # Calculate tokens per second if available
            total_tokens = sum(r.get("tokens_generated", 0) for r in successful_results)
            if total_tokens > 0:
                stats["total_tokens_generated"] = total_tokens
                stats["tokens_per_second"] = total_tokens / total_time

        # Collect error types
        if failed_results:
            error_types = {}
            for r in failed_results:
                error = r.get("error", "Unknown error")
                error_types[error] = error_types.get(error, 0) + 1
            stats["error_breakdown"] = error_types

        return stats

    def print_results(self, stats: Dict[str, Any]):
        """
        Print benchmark results in a formatted way.

        Args:
            stats: Statistics dictionary
        """
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")

        print(f"\nTotal Time: {stats['total_time_seconds']:.2f} seconds")
        print(f"\n**End-to-End QPS: {stats['end_to_end_qps']:.2f}**")
        print(f"Successful QPS: {stats['successful_qps']:.2f}")

        if "avg_latency_seconds" in stats:
            print(f"\nLatency Statistics (seconds):")
            print(f"  Average: {stats['avg_latency_seconds']:.3f}")
            print(f"  Median: {stats['median_latency_seconds']:.3f}")
            print(f"  Min: {stats['min_latency_seconds']:.3f}")
            print(f"  Max: {stats['max_latency_seconds']:.3f}")
            print(f"  P50: {stats['p50_latency_seconds']:.3f}")
            print(f"  P90: {stats['p90_latency_seconds']:.3f}")
            print(f"  P95: {stats['p95_latency_seconds']:.3f}")
            print(f"  P99: {stats['p99_latency_seconds']:.3f}")

        if "total_tokens_generated" in stats:
            print(f"\nToken Generation:")
            print(f"  Total Tokens: {stats['total_tokens_generated']}")
            print(f"  Tokens/Second: {stats['tokens_per_second']:.2f}")

        if "error_breakdown" in stats:
            print(f"\nError Breakdown:")
            for error_type, count in stats["error_breakdown"].items():
                print(f"  {error_type}: {count}")

        # Print sample outputs if collected
        if self.output_samples:
            print(f"\n" + "=" * 60)
            print(f"SAMPLE OUTPUTS (First {len(self.output_samples)} responses)")
            print("=" * 60)
            for i, sample in enumerate(self.output_samples, 1):
                print(f"\n--- Sample {i} (Request ID: {sample['request_id']}) ---")
                print(f"+++++++++++++++Prompt: {sample['prompt']}...")
                print(f">>>>>>>>>>>>>>>Output: {sample['output']}...")
                #print(f"Tokens: {sample['tokens']}, Time: {sample['time']:.2f}s")

        print("=" * 60)


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="vLLM Performance Benchmark Client")
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="cmu_dog_prompts.jsonl",
        help="Path to JSONL file containing prompts (default: cmu_dog_prompts.jsonl)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of vLLM service (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Maximum number of concurrent workers (default: 128)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per request (default: 8192)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts to use (default: use all)"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="Number of sample outputs to display (default: 0)"
    )

    args = parser.parse_args()

    # Initialize client
    client = VLLMBenchmarkClient(
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        show_samples=args.show_samples
    )

    # Load prompts
    prompts = client.load_prompts(args.prompts_file)

    # Apply limit if specified
    if args.limit and args.limit < len(prompts):
        prompts = prompts[:args.limit]
        print(f"Limited to first {args.limit} prompts")

    # Check if service is reachable
    try:
        test_response = requests.get(f"{args.base_url}/health", timeout=5)
        if test_response.status_code != 200:
            print(f"Warning: Health check returned status {test_response.status_code}")
    except:
        print(f"Warning: Could not reach vLLM service at {args.base_url}")
        print("Make sure the vLLM service is running before proceeding.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    prompts = prompts[1:2]

    # Run benchmark
    print(f"\nBenchmark Configuration:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Timeout: {args.timeout}s")
    if args.show_samples > 0:
        print(f"  Show Samples: {args.show_samples}")

    stats = client.run_benchmark(prompts, max_workers=args.max_workers)

    # Print results
    client.print_results(stats)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"

    # Add sample outputs to stats if collected
    if client.output_samples:
        stats["output_samples"] = client.output_samples

    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()