"""Benchmark TurboQuant vs baseline: TTFT, ITL, throughput, memory."""
import time
import sys
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 4096
GPU_UTIL = 0.5
NUM_PROMPTS = 8
OUTPUT_LEN = 128
PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."

def benchmark(kv_cache_dtype="auto", label="baseline"):
    print(f"\n{'='*60}")
    print(f"  {label} (kv_cache_dtype={kv_cache_dtype})")
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager="--eager" in sys.argv,
    )

    prompts = [PROMPT] * NUM_PROMPTS
    params = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0)

    # Warmup
    llm.generate([PROMPT], params)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    throughput = total_output_tokens / total_time

    print(f"  Prompts: {NUM_PROMPTS} x {OUTPUT_LEN} max tokens")
    print(f"  Total input tokens:  {total_input_tokens}")
    print(f"  Total output tokens: {total_output_tokens}")
    print(f"  Wall time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Sample output: {outputs[0].outputs[0].text[:100]}...")

    del llm
    torch.cuda.empty_cache()
    return throughput

if __name__ == "__main__":
    t_base = benchmark("auto", "BASELINE (bf16)")
    t_tq = benchmark("turboquant", "TURBOQUANT (4-bit)")

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline throughput:   {t_base:.1f} tok/s")
    print(f"  TurboQuant throughput: {t_tq:.1f} tok/s")
    print(f"  Ratio: {t_tq/t_base:.2f}x")
