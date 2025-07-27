
## w/ kvcached

```txt
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  105.74
Total input tokens:                      215196
Total generated tokens:                  129675
Request throughput (req/s):              9.46
Output token throughput (tok/s):         1226.41
Total Token throughput (tok/s):          3261.64
---------------Time to First Token----------------
Mean TTFT (ms):                          42.38
Median TTFT (ms):                        32.91
P99 TTFT (ms):                           158.48
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.33
Median TPOT (ms):                        14.95
P99 TPOT (ms):                           35.76
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.22
Median ITL (ms):                         11.19
P99 ITL (ms):                            73.25
==================================================
```

## w/o kvcached

```txt
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  103.70
Total input tokens:                      215196
Total generated tokens:                  130762
Request throughput (req/s):              9.64
Output token throughput (tok/s):         1260.95
Total Token throughput (tok/s):          3336.12
---------------Time to First Token----------------
Mean TTFT (ms):                          19.45
Median TTFT (ms):                        17.91
P99 TTFT (ms):                           35.52
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.61
Median TPOT (ms):                        7.54
P99 TPOT (ms):                           8.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.57
Median ITL (ms):                         7.34
P99 ITL (ms):                            17.76
==================================================
```
