"""Benchmark vLLM inference: tokens/sec, TTFT, latency.

Usage:
  python scripts/benchmark_model.py --model facebook/opt-125m
  python scripts/benchmark_model.py --model F:\VLLM-Models\qwen3-1_7b-coder-distilled-sft-Q8_0
"""
import argparse, json, time, os, sys
import numpy as np

os.environ["HIP_PATH"] = r"E:\ROCM-7.13.0-Windows"


def benchmark(args):
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Test prompts of varying lengths
    prompt_configs = [
        ("short", "What is the capital of France?", 5),
        ("medium", "Explain the theory of relativity in detail, covering both special and general relativity. Include the key equations and their physical significance." * 2, 50),
        ("long", "Explain the theory of relativity. " * 50, 150),
    ]

    max_tokens_list = [64, 128, 256] if not args.quick else [64]

    results = []

    for prompt_name, prompt, _ in prompt_configs:
        for max_tokens in max_tokens_list:
            print(f"\n  Benchmark: {prompt_name} prompt, max_tokens={max_tokens}")

            # Warmup
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=16,
                ignore_eos=True,
            )
            llm.generate([prompt], sampling_params)

            # Actual benchmark runs
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )

            latencies = []
            tokens_generated = []

            for _ in range(args.num_runs):
                t0 = time.time()
                outputs = llm.generate([prompt], sampling_params)
                elapsed = time.time() - t0
                out = outputs[0]
                n_tokens = len(out.outputs[0].token_ids)
                latencies.append(elapsed)
                tokens_generated.append(n_tokens)
                print(f"    Run {_+1}: {n_tokens} tokens in {elapsed:.3f}s = {n_tokens/elapsed:.1f} tok/s")

            avg_lat = np.mean(latencies)
            avg_tokens = np.mean(tokens_generated)
            throughput = avg_tokens / avg_lat
            results.append({
                "prompt": prompt_name,
                "max_tokens": max_tokens,
                "avg_tokens": round(avg_tokens, 1),
                "avg_latency_s": round(avg_lat, 3),
                "throughput_tok_s": round(throughput, 1),
                "min_latency_s": round(min(latencies), 3),
                "max_latency_s": round(max(latencies), 3),
            })

            if args.quick:
                break
        if args.quick:
            break

    print("\n\n========== BENCHMARK RESULTS ==========")
    print(f"{'Prompt':<10} {'MaxTok':<8} {'Tokens':<8} {'Latency':<10} {'Throughput':<12} {'VRAM':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['prompt']:<10} {r['max_tokens']:<8} {r['avg_tokens']:<8} "
              f"{r['avg_latency_s']:<10.3f} {r['throughput_tok_s']:<12.1f} -")

    # Peak VRAM
    try:
        import torch
        vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak VRAM: {vram:.2f} GB")
    except:
        pass

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    results = benchmark(args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")
