# DeepSeek DeepGEMM Kernels Benchmark

This directory includes benchmarks between DeepSeek's DeepGEMM block fp8 kernels against vLLM's existing triton and CUTLASS-based kernels.

Currently this just includes dense GEMMs and only works on Hopper GPUs.

## Setup

You need to install vLLM in your usual fashion, then install DeepGEMM from source:

```
git clone --recursive https://github.com/deepseek-ai/DeepGEMM
uv pip install -e DeepGEMM
```

## Usage

```
python benchmark_fp8_block_dense_gemm.py
INFO 02-26 19:12:16 [__init__.py:207] Automatically detected platform cuda.
===== STARTING FP8 GEMM BENCHMARK =====
Using device: NVIDIA H100 80GB HBM3

=== Benchmarking shape: m=8, n=4096, k=7168 ===
Running correctness check...
WARNING 02-26 19:12:19 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=4096,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
DeepGEMM vs Reference difference: 0.000689
vLLM Triton vs Reference difference: 0.000691
vLLM CUTLASS vs Reference difference: 0.000691
vLLM Triton vs DeepGEMM difference: 0.000011
vLLM CUTLASS vs DeepGEMM difference: 0.000011
DeepGEMM: 0.129 ms, 3.64 TFLOPS
vLLM Triton: 0.074 ms, 6.35 TFLOPS
vLLM CUTLASS: 0.034 ms, 13.71 TFLOPS
DeepGEMM is 1.74x faster than vLLM Triton
DeepGEMM is 3.76x faster than vLLM CUTLASS
vLLM CUTLASS is 2.16x faster than vLLM Triton

=== Benchmarking shape: m=8, n=7168, k=18432 ===
Running correctness check...
INFO 02-26 19:12:19 [fp8_utils.py:449] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=7168,K=18432,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json for W8A8 Block FP8 kernel.
DeepGEMM vs Reference difference: 0.000680
vLLM Triton vs Reference difference: 0.000680
vLLM CUTLASS vs Reference difference: 0.000680
vLLM Triton vs DeepGEMM difference: 0.000010
vLLM CUTLASS vs DeepGEMM difference: 0.000010
DeepGEMM: 0.114 ms, 18.48 TFLOPS
vLLM Triton: 0.091 ms, 23.14 TFLOPS
vLLM CUTLASS: 0.082 ms, 25.86 TFLOPS
DeepGEMM is 1.25x faster than vLLM Triton
DeepGEMM is 1.40x faster than vLLM CUTLASS
vLLM CUTLASS is 1.12x faster than vLLM Triton

=== Benchmarking shape: m=8, n=18432, k=7168 ===
Running correctness check...
WARNING 02-26 19:12:19 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=18432,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
DeepGEMM vs Reference difference: 0.000682
vLLM Triton vs Reference difference: 0.000682
vLLM CUTLASS vs Reference difference: 0.000682
vLLM Triton vs DeepGEMM difference: 0.000005
vLLM CUTLASS vs DeepGEMM difference: 0.000005
DeepGEMM: 0.113 ms, 18.68 TFLOPS
vLLM Triton: 0.117 ms, 18.03 TFLOPS
vLLM CUTLASS: 0.082 ms, 25.76 TFLOPS
DeepGEMM is 0.97x slower than vLLM Triton
DeepGEMM is 1.38x faster than vLLM CUTLASS
vLLM CUTLASS is 1.43x faster than vLLM Triton

=== Benchmarking shape: m=128, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000682
vLLM Triton vs Reference difference: 0.000682
vLLM CUTLASS vs Reference difference: 0.000682
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.114 ms, 65.79 TFLOPS
vLLM Triton: 0.091 ms, 82.65 TFLOPS
vLLM CUTLASS: 0.039 ms, 191.25 TFLOPS
DeepGEMM is 1.26x faster than vLLM Triton
DeepGEMM is 2.91x faster than vLLM CUTLASS
vLLM CUTLASS is 2.31x faster than vLLM Triton

=== Benchmarking shape: m=128, n=7168, k=18432 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000008
vLLM CUTLASS vs DeepGEMM difference: 0.000008
DeepGEMM: 0.115 ms, 293.95 TFLOPS
vLLM Triton: 0.143 ms, 236.69 TFLOPS
vLLM CUTLASS: 0.093 ms, 363.23 TFLOPS
DeepGEMM is 0.81x slower than vLLM Triton
DeepGEMM is 1.24x faster than vLLM CUTLASS
vLLM CUTLASS is 1.53x faster than vLLM Triton

=== Benchmarking shape: m=128, n=18432, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000684
vLLM Triton vs Reference difference: 0.000684
vLLM CUTLASS vs Reference difference: 0.000684
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.112 ms, 301.67 TFLOPS
vLLM Triton: 0.228 ms, 148.41 TFLOPS
vLLM CUTLASS: 0.086 ms, 395.53 TFLOPS
DeepGEMM is 0.49x slower than vLLM Triton
DeepGEMM is 1.31x faster than vLLM CUTLASS
vLLM CUTLASS is 2.67x faster than vLLM Triton

=== Benchmarking shape: m=1024, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.171 ms, 351.94 TFLOPS
vLLM Triton: 0.241 ms, 249.66 TFLOPS
vLLM CUTLASS: 0.101 ms, 598.08 TFLOPS
DeepGEMM is 0.71x slower than vLLM Triton
DeepGEMM is 1.70x faster than vLLM CUTLASS
vLLM CUTLASS is 2.40x faster than vLLM Triton

=== Benchmarking shape: m=1024, n=18432, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000684
vLLM Triton vs Reference difference: 0.000684
vLLM CUTLASS vs Reference difference: 0.000684
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.347 ms, 780.08 TFLOPS
vLLM Triton: 0.898 ms, 301.38 TFLOPS
vLLM CUTLASS: 0.331 ms, 817.56 TFLOPS
DeepGEMM is 0.39x slower than vLLM Triton
DeepGEMM is 1.05x faster than vLLM CUTLASS
vLLM CUTLASS is 2.71x faster than vLLM Triton

=== Benchmarking shape: m=2048, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.321 ms, 374.33 TFLOPS
vLLM Triton: 0.461 ms, 261.05 TFLOPS
vLLM CUTLASS: 0.200 ms, 601.60 TFLOPS
DeepGEMM is 0.70x slower than vLLM Triton
DeepGEMM is 1.61x faster than vLLM CUTLASS
vLLM CUTLASS is 2.30x faster than vLLM Triton

===== BENCHMARK SUMMARY =====
Matrix multiplication: C[m,n] = A[m,k] @ B[n,k].T

Average speedups:
DeepGEMM vs vLLM Triton: 1.32x faster
DeepGEMM vs vLLM CUTLASS: 0.64x slower
vLLM CUTLASS vs vLLM Triton: 2.07x faster

Average TFLOPS:
DeepGEMM: 245.40 TFLOPS
vLLM Triton: 147.48 TFLOPS
vLLM CUTLASS: 336.95 TFLOPS

Average accuracy difference vs reference:
DeepGEMM: 0.000683
vLLM Triton: 0.000684
vLLM CUTLASS: 0.000684
```
