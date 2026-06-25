# Notes:
evaulate if helion is faster than cutlass, with the following assumptions:
1. gemv(batch_size=1)
2. as a quick PoC- i didn't added it to the helion repo and instead I imported as it is
   - that means that the kernel might not be optimized(we need to autotune it).
3. tested for now on dgx spark(sm121) build with 12.1a
4. testing on this model 
Observations: 
1. correctness test for  compare to cutlass- pass
2. benchmark (kernel level) results show helion is faster 
    - nvfp4-noquant means the kerenel itself
    - the other options(without the noquant ) means arranging the data with the SWIZZLE_32_4_4 layout
3. coldstart with helion is slower than cutlass

benchmark (kernel level)- showing just the releavent data(batch-size=1):
```
F16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.163452                            0.831522                                    0.878304                                       0.970006                                               1.063832
meta-llama/Llama-3.1-8B-Instruct, N=4096 K=4096, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.182544                            0.790985                                    0.845340                                       1.000162                                               1.038173
meta-llama/Llama-3.1-8B-Instruct, N=28672 K=4096, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.237491                            0.745372                                    0.753888                                       0.761133                                               0.790657
meta-llama/Llama-3.1-8B-Instruct, N=4096 K=14336, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.230543                            0.787372                                    0.809048                                       0.865911                                               0.851426
```

benchmark(model level):
```
  vllm bench latency \
    --model RedHatAI/Qwen3-30B-A3B-NVFP4 \
    --linear-backend helion #\ or cutlass
    --input-len 128 \
    --output-len 128 \
    --batch-size 1 \
```
helion:
10% percentile latency: 1.8168933711014688 seconds
25% percentile latency: 1.8200808880501427 seconds
50% percentile latency: 1.8222514321096241 seconds
75% percentile latency: 1.8232573161367327 seconds
90% percentile latency: 1.8247000063303858 seconds
99% percentile latency: 1.825230823212769 seconds

cutlass:
Avg latency: 1.8150567370932549 seconds
10% percentile latency: 1.8117690255632624 seconds
25% percentile latency: 1.8134913760004565 seconds
50% percentile latency: 1.8150849200319499 seconds
75% percentile latency: 1.8167916560778394 seconds
90% percentile latency: 1.8177597534842789 seconds
99% percentile latency: 1.8234028382529506 seconds

helion (1.2.1.dev2+gf98a204f6)
commit f98a204f62e038b66269047b4fcb42fdf77f05cf
Author: Shangdi Yu <shangdiy@meta.com>
Date:   Mon Jun 22 10:17:10 2026 -0700


vllm bench latency \
--model RedHatAI/Llama-3.1-8B-Instruct-NVFP4 \
--linear-backend helion \
--input-len 128 \
--output-len 128 \
--batch-size 1 \
#--enforce-eager

imporvmenet of 0.2%-0.7% 
"helion":
Avg latency: 3.386302546132356 seconds
10% percentile latency: 3.374913340853527 seconds
25% percentile latency: 3.3774117599241436 seconds
50% percentile latency: 3.3837158320238814 seconds
75% percentile latency: 3.388888300047256 seconds
90% percentile latency: 3.4036725999088957 seconds
99% percentile latency: 3.4229529345431366 seconds
"cutlass":
Avg latency: 3.394636163720861 seconds
10% percentile latency: 3.3792949807597323 seconds
25% percentile latency: 3.3830675678909756 seconds
50% percentile latency: 3.3901969039579853 seconds
75% percentile latency: 3.398571068129968 seconds
90% percentile latency: 3.414220852870494 seconds
99% percentile latency: 3.4485593167413024 seconds

insights here:
(lkesem) redhat-et@dgx-spark-3:~/src/lkesem/vllm$ grep "backend cute" logs_helion_bench.txt | sort -u
(EngineCore pid=491781)  gemv_helion path: M=1,N=28672,K=2048 with backend cute
(EngineCore pid=491781)  gemv_helion path: M=1,N=4096,K=2048 with backend cute
(EngineCore pid=491781)  gemv_helion path: M=1,N=6144,K=2048 with backend cute
(lkesem) redhat-et@dgx-spark-3:~/src/lkesem/vllm$ grep "backend triton" logs_helion_bench.txt | sort -u
(EngineCore pid=491781)  gemv_helion path: M=1,N=4096,K=7168 with backend triton


vllm bench latency \
--model RedHatAI/Qwen3-30B-A3B-NVFP4 \
--linear-backend cutlass #\ or cutlass
--input-len 128 \
--output-len 128 \
--batch-size 1 \
--enforce-eager

helion:
Avg latency: 4.183671469862262 seconds
10% percentile latency: 4.071383041585795 seconds
25% percentile latency: 4.085507431998849 seconds
50% percentile latency: 4.167008064105175 seconds
75% percentile latency: 4.264231440029107 seconds
90% percentile latency: 4.309902897640131 seconds
99% percentile latency: 4.453812835721765 seconds

cutlass:
Avg latency: 4.266653785629509 seconds
10% percentile latency: 4.11550148637034 seconds
25% percentile latency: 4.190905123949051 seconds
50% percentile latency: 4.25785802397877 seconds
75% percentile latency: 4.355307903955691 seconds
90% percentile latency: 4.4704073904780675 seconds
99% percentile latency: 4.551459094893653 seconds

___ not inculde (not running batch_size=1 ever)
why it is just 

vllm bench throughput \
--model RedHatAI/Llama-3.1-8B-Instruct-NVFP4 \
--linear-backend cutlass \
--input-len 128 \
--output-len 128 \
--num-prompts 100


helion:
Throughput: 3.99 requests/s, 4591.71 total tokens/s, 510.19 output tokens/s
Total num prompt tokens:  102400
Total num output tokens:  12800

cutlass:
Throughput: 3.76 requests/s, 4329.26 total tokens/s, 481.03 output tokens/s
Total num prompt tokens:  102400
Total num output tokens:  12800
__
 requests/s (3.99 - 3.76) / 3.76 = 6.1% faster
 output tokens/s (510.19 - 481.03)/481.03=6.1% faster
__
why? 

vllm serve RedHatAI/Qwen3-30B-A3B-NVFP4 \
--linear-backend cutlass \
--moe-backend cutlass \
--tensor-parallel-size 1 \
--port 8000 \
--max-num-seqs 1


vllm bench serve \
  --backend vllm \
  --model RedHatAI/Qwen3-30B-A3B-NVFP4 \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 100 \
  --max-concurrency 1 \
  --input-len 128 \
  --output-len 128

rm -rf ~/.cache/torch_extensions/*
pkill -f "vllm serve"
  00%|█████████████████████████████████████████████████| 100/100 [03:16<00:00,  1.97s/it]
=========== Helion Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  195.47    
Total input tokens:                      12800     
Total generated tokens:                  12800     
Request throughput (req/s):              0.51      
Output token throughput (tok/s):         65.48     
Peak output token throughput (tok/s):    68.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          130.97    
---------------Time to First Token----------------
Mean TTFT (ms):                          71.04     
Median TTFT (ms):                        72.29     
P99 TTFT (ms):                           78.61     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.83     
Median TPOT (ms):                        14.83     
P99 TPOT (ms):                           14.90     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.83     
Median ITL (ms):                         14.85     
P99 ITL (ms):                            16.09     

rm -rf ~/.cache/torch_extensions/*
pkill -f "vllm serve"
============ Cutlass Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  196.14    
Total input tokens:                      12800     
Total generated tokens:                  12800     
Request throughput (req/s):              0.51      
Output token throughput (tok/s):         65.26     
Peak output token throughput (tok/s):    68.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          130.52    
---------------Time to First Token----------------
Mean TTFT (ms):                          80.81     
Median TTFT (ms):                        74.64     
P99 TTFT (ms):                           120.37    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.81     
Median TPOT (ms):                        14.80     
P99 TPOT (ms):                           14.90     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.81     
Median ITL (ms):                         14.83     
P99 ITL (ms):                            16.09    