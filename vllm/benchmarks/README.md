We support high-throughput benchmark.

[Benchmark multiprocessing]
Background: TBD

Implementation: TBD

[Benchmark trimming]
Background:
To measure the high-throughput of the benchmark, the beginning (where decoding batches gradually increase as they are filled) and the end (where decoding batches gradually decrease as they are completed) must be excluded from the overall benchmark duration.

Implementation:
Trimming is implemented based on the response metadata for each request collected from the benchmark. The response metadata includes the request transmission time, the first token generation delay, and the inter-token generation delay. Based on this, the generation time of all tokens is reversed, and token information within the user-specified time interval is collected based on that time.





Test result:
The benchmark execution script set the warmup-time to 150 seconds and the cooldown-time to 120 seconds. The trimmed experimental results are listed at the bottom of the benchmark output, allowing us to confirm the following metrics: 1) the benchmark execution time after trimming, 2) the number of tokens generated within the defined time interval, 3) the output token throughput, and 4) the Inter-Token Latency (ITL).
Furthermore, the graph below, which visually compares the difference resulting from the trimming, shows the specific time interval from which token information was aggregated. Additionally, since the width of the bars in the graph represents 1 second, the bar height can be interpreted as tokens per second, which directly represents the output token throughput.


