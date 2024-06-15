
## Serving tests

This test suite aims to test vllm's real serving metrics, while keeping the setup deterministic to reduce the variance of these numbers.

Concretely, we sample 200 (input, output) pairs from ShareGPT dataset using a fixed random seed. Then, for each (input, output) pair, we prepare a dummy user request with same input length, and request the server to generate the same amount of tokens as the output.

To simulate the arrival time of different requests, we vary the average QPS (query per second) and then determine the arrival time of each request by using a random Poisson process (with fixed random seed). And we assume all requests arrive at the same time if QPS is inf.

We cover llama-3 8B model and 70B model, together with mixtral 8x7B model.


We evaluate the performance of this test using standard serving metrics:
- Throughput (how many requests can be processed per second),
- TTFT (time to the first token),
- ITL (inter-token latency).
