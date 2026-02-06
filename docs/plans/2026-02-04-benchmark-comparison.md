# Benchmark Comparison: iterations.py vs Standalone Benchmark

> **For Claude:** REQUIRED SUB-SKILL: Use 10x-engineer:executing-plans to implement this plan task-by-task.

**Goal:** Compare prefill and decode performance between the server-client iterations.py benchmark and the standalone benchmark (fbcode) using DeepSeek-V3 with DP=8, EP enabled.

**Architecture:** Run both benchmarks with identical parameters, collect latency metrics, and verify they produce similar results. The iterations.py uses HTTP API while standalone uses direct LLM class.

**Tech Stack:** vLLM, DeepSeek-V3, 8x B200 GPUs, Python, aiohttp

---

## Environment

- **Host:** devgpu023.snb3.facebook.com
- **GPUs:** 8x NVIDIA B200 (183GB each)
- **vLLM venv:** ~/uv_env/vllm
- **OSS vLLM repo:** /data/users/jaewon/gitrepos/vllm
- **fbcode repo:** /data/users/jaewon/fbsource/fbcode/vllm/fb/scripts

## Test Parameters

| Parameter | Prefill Test | Decode Test |
|-----------|--------------|-------------|
| Mode | prefill | decode |
| Context Length | 4096 | 4096 |
| Input Length | 2048 | 1 |
| Batch Size (per DP) | 64 | 64 |
| Iterations | 1 | 10 |
| DP Size | 8 | 8 |
| TP Size | 1 | 1 |
| Expert Parallel | Yes | Yes |

---

### Task 1: Start vLLM Server with DeepSeek-V3

**Step 1: Start the server in background**

```bash
cd /data/users/jaewon/gitrepos/vllm
source ~/uv_env/vllm/bin/activate
FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve deepseek-ai/DeepSeek-V3 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /data/users/jaewon/gitrepos/vllm/traces \
  --data-parallel-size 8 --tensor-parallel-size 1 --enable-expert-parallel \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --no-enable-chunked-prefill --max-num-batched-tokens 16384 \
  2>&1 | tee serverlog &
```

**Step 2: Wait for server to be ready**

Poll the health endpoint until it returns 200:
```bash
for i in {1..120}; do
  curl -s http://localhost:8000/health && break
  sleep 5
done
```

Expected: Server healthy after model loads (~5-10 minutes for large model)

---

### Task 2: Run iterations.py Decode Benchmark

**Step 1: Run decode benchmark**

```bash
cd /data/users/jaewon/gitrepos/vllm
source ~/uv_env/vllm/bin/activate
python -m vllm.benchmarks.iterations \
  --endpoints localhost:8000 \
  --mode decode \
  --context-len 4096 \
  --batch-size 64 \
  --iterations 10 \
  --model deepseek-ai/DeepSeek-V3 \
  --profile \
  --output decode_iterations_result.json \
  2>&1 | tee clientlog_decode
```

**Step 2: Record results**

Extract key metrics:
- Total latency (ms)
- Per-iteration latency (ms)
- Tokens per second

---

### Task 3: Run iterations.py Prefill Benchmark

**Step 1: Run prefill benchmark**

```bash
cd /data/users/jaewon/gitrepos/vllm
source ~/uv_env/vllm/bin/activate
python -m vllm.benchmarks.iterations \
  --endpoints localhost:8000 \
  --mode prefill \
  --context-len 4096 \
  --input-len 2048 \
  --batch-size 64 \
  --model deepseek-ai/DeepSeek-V3 \
  --profile \
  --output prefill_iterations_result.json \
  2>&1 | tee clientlog_prefill
```

**Step 2: Record results**

Extract key metrics:
- Total latency (ms)
- TTFT (Time to First Token)

---

### Task 4: Stop vLLM Server

**Step 1: Kill the server**

```bash
pkill -f "vllm serve"
# Wait for cleanup
sleep 10
# Verify no vLLM processes
nvidia-smi | grep -i vllm || echo "Server stopped"
```

---

### Task 5: Run Standalone Benchmark Decode

**Step 1: Run standalone decode benchmark**

```bash
cd /data/users/jaewon/fbsource/fbcode/vllm/fb/scripts
source ~/uv_env/vllm/bin/activate
python3 sweep_standalone_benchmark.py \
  --hosts devgpu023.snb3.facebook.com \
  --node-size 8 --tp-size 1 --expert-parallel \
  --model deepseek-ai/DeepSeek-V3 \
  --mode decode \
  --context-lengths 4096 \
  --input-lengths 1 \
  --batch-sizes 64 \
  --iters 10 \
  --enable-trace \
  --output standalone_decode_result.csv \
  2>&1 | tee standalone_decode_log
```

**Step 2: Record results**

Extract key metrics from CSV and logs.

---

### Task 6: Run Standalone Benchmark Prefill

**Step 1: Run standalone prefill benchmark**

```bash
cd /data/users/jaewon/fbsource/fbcode/vllm/fb/scripts
source ~/uv_env/vllm/bin/activate
python3 sweep_standalone_benchmark.py \
  --hosts devgpu023.snb3.facebook.com \
  --node-size 8 --tp-size 1 --expert-parallel \
  --model deepseek-ai/DeepSeek-V3 \
  --mode prefill \
  --context-lengths 4096 \
  --input-lengths 2048 \
  --batch-sizes 64 \
  --iters 1 \
  --enable-trace \
  --output standalone_prefill_result.csv \
  2>&1 | tee standalone_prefill_log
```

**Step 2: Record results**

Extract key metrics from CSV and logs.

---

### Task 7: Compare Results

**Step 1: Create comparison table**

| Metric | iterations.py | Standalone | Difference |
|--------|---------------|------------|------------|
| Decode Latency/iter (ms) | X | Y | Z% |
| Prefill Latency (ms) | X | Y | Z% |
| Tokens/sec | X | Y | Z% |

**Step 2: Analyze discrepancies**

If >10% difference, investigate:
- Check prefix cache hit rates
- Compare profiler traces
- Review timing methodology differences

---

### Task 8: Iterate if Needed

If results don't match, investigate and fix:
1. Check if batch size semantics are correctly aligned
2. Verify prefix cache warmup is working
3. Compare profiler traces for timing differences
4. Adjust parameters and re-run

---

## Execution Results (2026-02-04)

### iterations.py Benchmark Results (Completed)

**Decode Benchmark:**
| Metric | Value |
|--------|-------|
| Mode | decode |
| Context Length | 4096 |
| Batch Size (per DP) | 64 |
| Iterations | 10 |
| Total Latency | 2109.49 ms |
| Per-iteration Latency | 210.95 ms |
| Tokens/sec | 498,637 |
| Prompt Tokens | 1,049,600 |

**Prefill Benchmark:**
| Metric | Value |
|--------|-------|
| Mode | prefill |
| Context Length | 4096 |
| Input Length | 2048 |
| Batch Size (per DP) | 64 |
| Total Latency | 2611.03 ms |
| Tokens/sec | 602,980 |

### Standalone Benchmark Results (BLOCKED)

**Status: BLOCKED due to fbcode flashinfer kernel issue**

The standalone benchmark fails during CUDA graph capture with the following error:

```
RuntimeError: Error in function 'run' at fbcode/deeplearning/flashinfer/csrc/trtllm_batched_gemm_runner.cu:252:
Error occurred when running GEMM! (numBatches: 32, GemmMNK: 262144 4096 7168,
Kernel: bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_TN_transOut_noShflA_dsFp8_schedP4x2x2x3_bN_ldgsts_tmaOpt_clmp_dynBatch_sm100f)
```

**Root Cause Analysis:**
- The GEMM dimensions (262144 x 4096 x 7168) are fixed during CUDA graph capture, regardless of batch size or context length
- This indicates the issue is in the default cudagraph capture size configuration in the fbcode vLLM build
- The fbcode version uses `max_cudagraph_capture_size: None` which allows very large capture sizes
- The OSS vLLM version limits capture to `max_cudagraph_capture_size: 512`
- The TRT-LLM FP8 batched GEMM kernel in fbcode does not support matrix sizes of 262K tokens

**Workaround Needed:**
- Add `--enforce-eager` flag support to standalone_benchmark.py to disable CUDA graphs
- Or limit `max_cudagraph_capture_size` in fbcode to match OSS (512)
- Or fix the TRT-LLM batched GEMM kernel to handle larger matrix sizes

---

## Conclusions

1. **iterations.py benchmark works correctly** with DeepSeek-V3, DP=8, EP enabled
2. **Batch size semantics were fixed** - iterations.py now interprets batch size as per-DP (matching standalone)
3. **Standalone benchmark is blocked** - requires fix to fbcode flashinfer TRT-LLM kernel or CUDA graph configuration
4. **Next Steps:**
   - File a bug for the fbcode flashinfer GEMM kernel issue
   - Add `--enforce-eager` support to standalone benchmark script
   - Re-run comparison once standalone is fixed

---

## Success Criteria

- Decode per-iteration latency within 10% between both benchmarks
- Prefill latency within 10% between both benchmarks
- Both benchmarks show consistent prefix cache hit behavior
