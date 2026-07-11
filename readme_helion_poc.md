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
0          1.0                                 0.161752                            0.815279                                    0.877524                                       1.450914                                               1.528096
1         16.0                                 3.580709                           13.479930                                   14.827917                                      13.353627                                              14.588392
2         64.0                                13.365884                           76.621407                                   85.477586                                      73.389320                                              79.804941
3        128.0                                26.168854                          296.761776                                  342.231580                                     264.445609                                             296.511651
4        256.0                                47.702834                          315.769493                                  349.012000                                     198.599701                                             305.996366
5        512.0                                79.831797                          289.700652                                  372.813647                                     182.493388                                             230.350182
6       1024.0                                99.110894                          247.594494                                  320.314702                                     177.494994                                             222.376221
7       2048.0                                96.516882                          236.083331                                  310.980728                                     169.949044                                             205.861989
8       4096.0                                95.760475                          235.141426                                  314.979346                                     158.841068                                             190.584700
9       8192.0                                99.195134                          170.838864                                  201.721279                                     125.628779                                             137.056171
10     16384.0                               101.934594                          239.682987                                  321.940883                                     159.089559                                             182.992172
meta-llama/Llama-3.1-8B-Instruct, N=4096 K=4096, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.187590                            0.772946                                    0.843370                                       1.347994                                               1.476193
1         16.0                                 3.887617                           12.933233                                   14.565841                                      12.701804                                              14.225251
2         64.0                                15.182655                           72.250372                                   80.383774                                      69.175095                                              76.039464
3        128.0                                27.723389                          215.176612                                  246.461507                                     195.241182                                             223.906695
4        256.0                                50.171508                          226.664909                                  259.058060                                     212.480869                                             234.090786
5        512.0                                82.755717                          240.226565                                  261.974821                                     184.345582                                             222.521420
6       1024.0                                98.484803                          237.123238                                  324.534105                                     167.555157                                             229.423086
7       2048.0                                90.118775                          200.963337                                  281.834260                                     147.877478                                             199.068022
8       4096.0                                93.166970                          207.796195                                  310.055826                                     154.688710                                             188.499618
9       8192.0                                99.010778                          213.648609                                  325.756002                                     152.390271                                             194.673371
10     16384.0                               101.663549                          219.095180                                  341.001313                                     149.730641                                             191.709128
meta-llama/Llama-3.1-8B-Instruct, N=28672 K=4096, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.234866                            0.753434                                    0.761050                                       0.929634                                               0.973020
1         16.0                                 3.664251                           11.329284                                   11.641321                                      11.286220                                              11.364329
2         64.0                                13.599812                           43.190127                                   43.951350                                      40.213362                                              40.357697
3        128.0                                26.302357                           80.086245                                   82.238056                                      71.700003                                              72.875616
4        256.0                                46.761919                          141.446211                                  147.362645                                     116.002433                                             119.159954
5        512.0                                79.569655                          236.867910                                  237.383004                                     167.075597                                             175.136580
6       1024.0                                87.883868                          290.775870                                  315.405413                                     188.267567                                             198.757119
7       2048.0                                93.584410                          286.664003                                  306.181594                                     175.034617                                             183.969835
8       4096.0                                97.091346                          284.806224                                  307.640640                                     176.130670                                             184.900569
9       8192.0                                99.306260                          195.069502                                  200.977962                                     135.438031                                             136.203474
10     16384.0                               101.292172                           80.138264                                   81.255643                                      67.866110                                              67.554301
meta-llama/Llama-3.1-8B-Instruct, N=4096 K=14336, BF16 vs NVFP4 GEMMs TFLOP/s:
BF16 vs NVFP4 GEMMs:
    batch_size  torch-bf16 (TFLOP/s (larger is better))  nvfp4 (TFLOP/s (larger is better))  nvfp4-noquant (TFLOP/s (larger is better))  helion-gemv-w4a4 (TFLOP/s (larger is better))  helion-gemv-w4a4-noquant (TFLOP/s (larger is better))
0          1.0                                 0.229241                            0.756272                                    0.772649                                       1.011900                                               1.081111
1         16.0                                 3.572851                           11.935613                                   12.176134                                      11.671092                                              11.858010
2         64.0                                13.363322                           45.240429                                   47.714171                                      44.732699                                              47.296806
3        128.0                                26.932985                           78.691128                                   92.890414                                      73.679590                                              86.591871
4        256.0                                49.099030                          126.208541                                  159.504582                                     119.315915                                             141.052717
5        512.0                                73.347255                          169.144161                                  219.240787                                     157.432522                                             190.927226
6       1024.0                                96.077062                          241.258916                                  340.634437                                     215.488065                                             287.590853
7       2048.0                                87.748292                          232.586051                                  335.092451                                     203.341741                                             285.641590
8       4096.0                                91.285726                          167.987295                                  218.050061                                     147.922090                                             193.824331
9       8192.0                                94.957175                          159.535990                                  211.365013                                     144.446679                                             187.965708
10     16384.0                                96.254391                          161.209098                                  213.970619                                     146.130458                                             187.939760
Benchmark finished!
```


(lkesem) redhat-et@dgx-spark-3:~/src/lkesem/vllm$ 
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