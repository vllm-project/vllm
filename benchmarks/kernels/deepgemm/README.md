# DeepSeek DeepGEMM Kernels Benchmark

This directory includes benchmarks between DeepSeek's DeepGEMM block fp8 kernels against vLLM's existing triton and CUTLASS-based kernels.

Currently, this just includes dense GEMMs and only works on Hopper GPUs.

## Setup

You need to install vLLM in your usual fashion, then install DeepGEMM from source in its own directory:

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM
cd DeepGEMM
python setup.py install
uv pip install -e .
```

## Usage

```console
python benchmark_fp8_block_dense_gemm.py
INFO 02-26 21:55:13 [__init__.py:207] Automatically detected platform cuda.
===== STARTING FP8 GEMM BENCHMARK =====
PyTorch version: 2.5.1+cu124
CUDA version: 12.4
Triton version: 3.1.0
Using device: NVIDIA H100 80GB HBM3
WARNING 02-26 21:55:15 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=4096,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
INFO 02-26 21:55:15 [fp8_utils.py:449] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=7168,K=18432,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json for W8A8 Block FP8 kernel.
WARNING 02-26 21:55:16 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=18432,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
WARNING 02-26 21:55:17 [fp8_utils.py:458] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=24576,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
INFO 02-26 21:55:17 [fp8_utils.py:449] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=32768,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json for W8A8 Block FP8 kernel.
INFO 02-26 21:55:17 [fp8_utils.py:449] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=7168,K=16384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json for W8A8 Block FP8 kernel.

===== PERFORMANCE COMPARISON =====

DeepGEMM Implementation:
+------+-------+-------+-----------+--------+--------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   |
+------+-------+-------+-----------+--------+--------+
|    8 |  4096 |  7168 | 102.9     | 4.6    | 286.4  |
|    8 |  7168 | 18432 | 70.8      | 29.8   | 1868.8 |
|    8 | 18432 |  7168 | 69.3      | 30.5   | 1911.8 |
|   64 |  4096 |  7168 | 69.1      | 54.4   | 439.0  |
|   64 |  7168 | 18432 | 69.4      | 243.6  | 1933.6 |
|   64 | 18432 |  7168 | 70.4      | 240.3  | 1917.2 |
|   64 | 24576 |  1536 | 70.1      | 68.9   | 584.6  |
|   64 | 32768 |   512 | 68.4      | 31.4   | 307.1  |
|   64 |  7168 | 16384 | 69.5      | 216.3  | 1718.5 |
|  128 |  4096 |  7168 | 141.1     | 53.3   | 222.1  |
|  128 |  7168 | 18432 | 71.9      | 470.5  | 1896.1 |
|  128 | 18432 |  7168 | 69.3      | 488.2  | 1988.2 |
| 1024 |  4096 |  7168 | 89.7      | 670.1  | 502.5  |
| 1024 | 18432 |  7168 | 279.0     | 969.8  | 635.2  |
| 2048 |  4096 |  7168 | 175.1     | 687.0  | 347.4  |
| 4096 |  4096 |  7168 | 335.4     | 717.0  | 275.1  |
+------+-------+-------+-----------+--------+--------+

vLLM Triton Implementation:
+------+-------+-------+-----------+--------+--------+--------------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   | vs DeepGEMM  |
+------+-------+-------+-----------+--------+--------+--------------+
|    8 |  4096 |  7168 | 74.0      | 6.3    | 398.2  | 1.39x faster |
|    8 |  7168 | 18432 | 89.6      | 23.6   | 1478.1 | 0.79x slower |
|    8 | 18432 |  7168 | 113.2     | 18.7   | 1170.4 | 0.61x slower |
|   64 |  4096 |  7168 | 79.4      | 47.3   | 382.2  | 0.87x slower |
|   64 |  7168 | 18432 | 98.5      | 171.7  | 1363.0 | 0.70x slower |
|   64 | 18432 |  7168 | 119.5     | 141.5  | 1129.4 | 0.59x slower |
|   64 | 24576 |  1536 | 37.6      | 128.4  | 1089.7 | 1.86x faster |
|   64 | 32768 |   512 | 38.7      | 55.5   | 542.6  | 1.77x faster |
|   64 |  7168 | 16384 | 86.1      | 174.5  | 1386.4 | 0.81x slower |
|  128 |  4096 |  7168 | 90.7      | 82.9   | 345.4  | 1.56x faster |
|  128 |  7168 | 18432 | 144.0     | 234.9  | 946.9  | 0.50x slower |
|  128 | 18432 |  7168 | 229.5     | 147.4  | 600.1  | 0.30x slower |
| 1024 |  4096 |  7168 | 242.3     | 248.2  | 186.1  | 0.37x slower |
| 1024 | 18432 |  7168 | 897.8     | 301.4  | 197.4  | 0.31x slower |
| 2048 |  4096 |  7168 | 463.0     | 259.7  | 131.4  | 0.38x slower |
| 4096 |  4096 |  7168 | 901.8     | 266.7  | 102.3  | 0.37x slower |
+------+-------+-------+-----------+--------+--------+--------------+

vLLM CUTLASS Implementation:
+------+-------+-------+-----------+--------+--------+--------------+--------------+
| m    | n     | k     | Time (μs) | TFLOPS | GB/s   | vs DeepGEMM  | vs Triton    |
+------+-------+-------+-----------+--------+--------+--------------+--------------+
|    8 |  4096 |  7168 | 34.6      | 13.6   | 852.3  | 2.98x faster | 2.14x faster |
|    8 |  7168 | 18432 | 78.9      | 26.8   | 1677.3 | 0.90x slower | 1.13x faster |
|    8 | 18432 |  7168 | 81.2      | 26.0   | 1631.1 | 0.85x slower | 1.39x faster |
|   64 |  4096 |  7168 | 36.9      | 101.9  | 822.9  | 1.87x faster | 2.15x faster |
|   64 |  7168 | 18432 | 87.4      | 193.4  | 1535.2 | 0.79x slower | 1.13x faster |
|   64 | 18432 |  7168 | 85.0      | 199.0  | 1587.6 | 0.83x slower | 1.41x faster |
|   64 | 24576 |  1536 | 28.0      | 172.8  | 1465.8 | 2.51x faster | 1.35x faster |
|   64 | 32768 |   512 | 28.8      | 74.5   | 728.5  | 2.37x faster | 1.34x faster |
|   64 |  7168 | 16384 | 77.9      | 193.0  | 1532.8 | 0.89x slower | 1.11x faster |
|  128 |  4096 |  7168 | 39.1      | 192.4  | 802.0  | 3.61x faster | 2.32x faster |
|  128 |  7168 | 18432 | 93.7      | 360.8  | 1454.2 | 0.77x slower | 1.54x faster |
|  128 | 18432 |  7168 | 85.7      | 394.8  | 1608.0 | 0.81x slower | 2.68x faster |
| 1024 |  4096 |  7168 | 99.7      | 603.1  | 452.2  | 0.90x slower | 2.43x faster |
| 1024 | 18432 |  7168 | 331.3     | 816.7  | 534.9  | 0.84x slower | 2.71x faster |
| 2048 |  4096 |  7168 | 198.3     | 606.6  | 306.7  | 0.88x slower | 2.34x faster |
| 4096 |  4096 |  7168 | 392.2     | 613.2  | 235.3  | 0.86x slower | 2.30x faster |
+------+-------+-------+-----------+--------+--------+--------------+--------------+

===== AVERAGE PERFORMANCE =====
+----------------+------------+----------+---------------+
| Implementation | Avg TFLOPS | Avg GB/s | Avg Time (ms) |
+----------------+------------+----------+---------------+
| DeepGEMM       | 310.98     | 1052.10  | 0.11          |
| vLLM Triton    | 144.30     | 715.60   | 0.23          |
| vLLM CUTLASS   | 286.78     | 1076.67  | 0.11          |
+----------------+------------+----------+---------------+

===== AVERAGE SPEEDUPS =====
+-----------------------------+--------------+
| Comparison                  | Speedup      |
+-----------------------------+--------------+
| DeepGEMM vs vLLM Triton     | 1.71x faster |
| DeepGEMM vs vLLM CUTLASS    | 0.94x slower |
| vLLM CUTLASS vs vLLM Triton | 1.84x faster |
+-----------------------------+--------------+

===== ACCURACY COMPARISON =====
+----------------+-----------------------+
| Implementation | Avg Diff vs Reference |
+----------------+-----------------------+
| DeepGEMM       | 0.000684              |
| vLLM Triton    | 0.000684              |
| vLLM CUTLASS   | 0.000684              |
+----------------+-----------------------+
```
