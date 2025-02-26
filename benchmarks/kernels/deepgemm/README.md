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
python benchmark_fp8_block_dense_gemm_table.py
INFO 02-26 21:35:35 [__init__.py:207] Automatically detected platform cuda.
===== STARTING FP8 GEMM BENCHMARK =====
PyTorch version: 2.5.1+cu124
CUDA version: 12.4
Triton version: 3.1.0
Using device: NVIDIA H100 80GB HBM3

===== PERFORMANCE COMPARISON =====

DeepGEMM Implementation:
+------+-------+-------+-----------+--------+--------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   |
+------+-------+-------+-----------+--------+--------+
|    8 |  4096 |  7168 | 85.1      | 5.5    | 346.3  |
|    8 |  7168 | 18432 | 83.9      | 25.2   | 1577.3 |
|    8 | 18432 |  7168 | 84.1      | 25.1   | 1576.0 |
|   64 | 24576 |  1536 | 86.1      | 56.1   | 476.0  |
|   64 | 32768 |   512 | 84.0      | 25.6   | 250.1  |
|   64 |  7168 | 16384 | 120.3     | 124.9  | 992.5  |
|   64 |  4096 |  7168 | 84.5      | 44.5   | 359.3  |
|  128 |  4096 |  7168 | 85.0      | 88.4   | 368.5  |
|  128 |  7168 | 18432 | 88.3      | 383.0  | 1543.5 |
|  128 | 18432 |  7168 | 86.4      | 391.4  | 1594.3 |
| 1024 |  4096 |  7168 | 91.7      | 655.5  | 491.5  |
| 1024 | 18432 |  7168 | 283.3     | 955.0  | 625.4  |
| 2048 |  4096 |  7168 | 177.6     | 677.1  | 342.4  |
| 4096 |  4096 |  7168 | 338.9     | 709.6  | 272.2  |
+------+-------+-------+-----------+--------+--------+

vLLM Triton Implementation:
+------+-------+-------+-----------+--------+--------+--------------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   | vs DeepGEMM  |
+------+-------+-------+-----------+--------+--------+--------------+
|    8 |  4096 |  7168 | 74.4      | 6.3    | 396.4  | 1.14x faster |
|    8 |  7168 | 18432 | 89.6      | 23.6   | 1476.7 | 0.94x slower |
|    8 | 18432 |  7168 | 116.5     | 18.1   | 1137.3 | 0.72x slower |
|   64 | 24576 |  1536 | 37.2      | 129.9  | 1101.8 | 2.31x faster |
|   64 | 32768 |   512 | 38.7      | 55.5   | 542.4  | 2.17x faster |
|   64 |  7168 | 16384 | 86.7      | 173.3  | 1376.5 | 1.39x faster |
|   64 |  4096 |  7168 | 76.9      | 48.8   | 394.4  | 1.10x faster |
|  128 |  4096 |  7168 | 89.2      | 84.2   | 351.0  | 0.95x slower |
|  128 |  7168 | 18432 | 142.9     | 236.8  | 954.2  | 0.62x slower |
|  128 | 18432 |  7168 | 227.5     | 148.7  | 605.5  | 0.38x slower |
| 1024 |  4096 |  7168 | 240.7     | 249.8  | 187.3  | 0.38x slower |
| 1024 | 18432 |  7168 | 901.9     | 300.0  | 196.5  | 0.31x slower |
| 2048 |  4096 |  7168 | 462.6     | 260.0  | 131.5  | 0.38x slower |
| 4096 |  4096 |  7168 | 901.6     | 266.8  | 102.3  | 0.38x slower |
+------+-------+-------+-----------+--------+--------+--------------+

vLLM CUTLASS Implementation:
+------+-------+-------+-----------+--------+--------+--------------+--------------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   | vs DeepGEMM  | vs Triton    |
+------+-------+-------+-----------+--------+--------+--------------+--------------+
|    8 |  4096 |  7168 | 33.9      | 13.9   | 869.7  | 2.51x faster | 2.19x faster |
|    8 |  7168 | 18432 | 78.9      | 26.8   | 1677.7 | 1.06x faster | 1.14x faster |
|    8 | 18432 |  7168 | 80.3      | 26.3   | 1649.8 | 1.05x faster | 1.45x faster |
|   64 | 24576 |  1536 | 28.3      | 170.9  | 1449.8 | 3.05x faster | 1.32x faster |
|   64 | 32768 |   512 | 27.8      | 77.2   | 754.8  | 3.02x faster | 1.39x faster |
|   64 |  7168 | 16384 | 78.5      | 191.6  | 1522.1 | 1.53x faster | 1.11x faster |
|   64 |  4096 |  7168 | 36.4      | 103.2  | 833.4  | 2.32x faster | 2.11x faster |
|  128 |  4096 |  7168 | 39.1      | 192.3  | 801.4  | 2.17x faster | 2.28x faster |
|  128 |  7168 | 18432 | 92.9      | 364.0  | 1467.1 | 0.95x slower | 1.54x faster |
|  128 | 18432 |  7168 | 85.6      | 395.1  | 1609.2 | 1.01x faster | 2.66x faster |
| 1024 |  4096 |  7168 | 100.6     | 597.7  | 448.2  | 0.91x slower | 2.39x faster |
| 1024 | 18432 |  7168 | 329.8     | 820.5  | 537.4  | 0.86x slower | 2.73x faster |
| 2048 |  4096 |  7168 | 198.7     | 605.1  | 306.0  | 0.89x slower | 2.33x faster |
| 4096 |  4096 |  7168 | 393.0     | 612.0  | 234.8  | 0.86x slower | 2.29x faster |
+------+-------+-------+-----------+--------+--------+--------------+--------------+

===== AVERAGE PERFORMANCE =====
+----------------+------------+----------+---------------+
| Implementation | Avg TFLOPS | Avg GB/s | Avg Time (ms) |
+----------------+------------+----------+---------------+
| DeepGEMM       | 297.65     | 772.53   | 0.13          |
| vLLM Triton    | 142.98     | 639.56   | 0.25          |
| vLLM CUTLASS   | 299.75     | 1011.51  | 0.11          |
+----------------+------------+----------+---------------+

===== AVERAGE SPEEDUPS =====
+-----------------------------+--------------+
| Comparison                  | Speedup      |
+-----------------------------+--------------+
| DeepGEMM vs vLLM Triton     | 1.59x faster |
| DeepGEMM vs vLLM CUTLASS    | 0.79x slower |
| vLLM CUTLASS vs vLLM Triton | 1.92x faster |
+-----------------------------+--------------+

===== ACCURACY COMPARISON =====
+----------------+-----------------------+
| Implementation | Avg Diff vs Reference |
+----------------+-----------------------+
| DeepGEMM       | 0.000685              |
| vLLM Triton    | 0.000685              |
| vLLM CUTLASS   | 0.000685              |
+----------------+-----------------------+
```
