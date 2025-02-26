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
INFO 02-26 19:45:44 [__init__.py:207] Automatically detected platform cuda.
===== STARTING FP8 GEMM BENCHMARK =====
Using device: NVIDIA H100 80GB HBM3

=== Benchmarking shape: m=8, n=4096, k=7168 ===
Running correctness check...
WARNING 02-26 19:45:47 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=4096,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
DeepGEMM vs Reference difference: 0.000689
vLLM Triton vs Reference difference: 0.000691
vLLM CUTLASS vs Reference difference: 0.000691
vLLM Triton vs DeepGEMM difference: 0.000011
vLLM CUTLASS vs DeepGEMM difference: 0.000011
DeepGEMM: 0.111 ms, 4.25 TFLOPS
vLLM Triton: 0.074 ms, 6.39 TFLOPS
vLLM CUTLASS: 0.034 ms, 13.71 TFLOPS
DeepGEMM is 0.66x slower than vLLM Triton
DeepGEMM is 0.31x slower than vLLM CUTLASS
vLLM CUTLASS is 2.15x faster than vLLM Triton

=== Benchmarking shape: m=8, n=7168, k=18432 ===
Running correctness check...
INFO 02-26 19:45:47 [fp8_utils.py:449] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=7168,K=18432,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json for W8A8 Block FP8 kernel.
DeepGEMM vs Reference difference: 0.000680
vLLM Triton vs Reference difference: 0.000680
vLLM CUTLASS vs Reference difference: 0.000680
vLLM Triton vs DeepGEMM difference: 0.000010
vLLM CUTLASS vs DeepGEMM difference: 0.000010
DeepGEMM: 0.112 ms, 18.83 TFLOPS
vLLM Triton: 0.092 ms, 22.86 TFLOPS
vLLM CUTLASS: 0.081 ms, 26.15 TFLOPS
DeepGEMM is 0.82x slower than vLLM Triton
DeepGEMM is 0.72x slower than vLLM CUTLASS
vLLM CUTLASS is 1.14x faster than vLLM Triton

=== Benchmarking shape: m=8, n=18432, k=7168 ===
Running correctness check...
WARNING 02-26 19:45:47 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=18432,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
DeepGEMM vs Reference difference: 0.000682
vLLM Triton vs Reference difference: 0.000682
vLLM CUTLASS vs Reference difference: 0.000682
vLLM Triton vs DeepGEMM difference: 0.000005
vLLM CUTLASS vs DeepGEMM difference: 0.000005
DeepGEMM: 0.109 ms, 19.35 TFLOPS
vLLM Triton: 0.117 ms, 18.06 TFLOPS
vLLM CUTLASS: 0.081 ms, 26.21 TFLOPS
DeepGEMM is 1.07x faster than vLLM Triton
DeepGEMM is 0.74x slower than vLLM CUTLASS
vLLM CUTLASS is 1.45x faster than vLLM Triton

=== Benchmarking shape: m=128, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000682
vLLM Triton vs Reference difference: 0.000682
vLLM CUTLASS vs Reference difference: 0.000682
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.109 ms, 68.76 TFLOPS
vLLM Triton: 0.091 ms, 82.65 TFLOPS
vLLM CUTLASS: 0.039 ms, 190.49 TFLOPS
DeepGEMM is 0.83x slower than vLLM Triton
DeepGEMM is 0.36x slower than vLLM CUTLASS
vLLM CUTLASS is 2.30x faster than vLLM Triton

=== Benchmarking shape: m=128, n=7168, k=18432 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000008
vLLM CUTLASS vs DeepGEMM difference: 0.000008
DeepGEMM: 0.115 ms, 294.42 TFLOPS
vLLM Triton: 0.142 ms, 237.38 TFLOPS
vLLM CUTLASS: 0.093 ms, 361.90 TFLOPS
DeepGEMM is 1.24x faster than vLLM Triton
DeepGEMM is 0.81x slower than vLLM CUTLASS
vLLM CUTLASS is 1.52x faster than vLLM Triton

=== Benchmarking shape: m=128, n=18432, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000684
vLLM Triton vs Reference difference: 0.000684
vLLM CUTLASS vs Reference difference: 0.000684
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.110 ms, 308.47 TFLOPS
vLLM Triton: 0.228 ms, 148.56 TFLOPS
vLLM CUTLASS: 0.086 ms, 394.22 TFLOPS
DeepGEMM is 2.08x faster than vLLM Triton
DeepGEMM is 0.78x slower than vLLM CUTLASS
vLLM CUTLASS is 2.65x faster than vLLM Triton

=== Benchmarking shape: m=1024, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.169 ms, 356.31 TFLOPS
vLLM Triton: 0.241 ms, 249.85 TFLOPS
vLLM CUTLASS: 0.101 ms, 592.45 TFLOPS
DeepGEMM is 1.43x faster than vLLM Triton
DeepGEMM is 0.60x slower than vLLM CUTLASS
vLLM CUTLASS is 2.37x faster than vLLM Triton

=== Benchmarking shape: m=1024, n=18432, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000684
vLLM Triton vs Reference difference: 0.000684
vLLM CUTLASS vs Reference difference: 0.000684
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.347 ms, 779.63 TFLOPS
vLLM Triton: 0.898 ms, 301.41 TFLOPS
vLLM CUTLASS: 0.331 ms, 818.21 TFLOPS
DeepGEMM is 2.59x faster than vLLM Triton
DeepGEMM is 0.95x slower than vLLM CUTLASS
vLLM CUTLASS is 2.71x faster than vLLM Triton

=== Benchmarking shape: m=2048, n=4096, k=7168 ===
Running correctness check...
DeepGEMM vs Reference difference: 0.000683
vLLM Triton vs Reference difference: 0.000683
vLLM CUTLASS vs Reference difference: 0.000683
vLLM Triton vs DeepGEMM difference: 0.000007
vLLM CUTLASS vs DeepGEMM difference: 0.000007
DeepGEMM: 0.320 ms, 376.32 TFLOPS
vLLM Triton: 0.460 ms, 261.25 TFLOPS
vLLM CUTLASS: 0.200 ms, 602.18 TFLOPS
DeepGEMM is 1.44x faster than vLLM Triton
DeepGEMM is 0.62x slower than vLLM CUTLASS
vLLM CUTLASS is 2.30x faster than vLLM Triton

===== BENCHMARK SUMMARY =====
Matrix multiplication: C[m,n] = A[m,k] @ B[n,k].T

Average speedups:
DeepGEMM vs vLLM Triton: 1.35x faster
DeepGEMM vs vLLM CUTLASS: 0.66x slower
vLLM CUTLASS vs vLLM Triton: 2.07x faster

Average TFLOPS:
DeepGEMM: 247.37 TFLOPS
vLLM Triton: 147.60 TFLOPS
vLLM CUTLASS: 336.17 TFLOPS

Average accuracy difference vs reference:
DeepGEMM: 0.000683
vLLM Triton: 0.000684
vLLM CUTLASS: 0.000684
```
