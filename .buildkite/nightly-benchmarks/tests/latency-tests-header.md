
## Latency tests

This test suite aims to test vllm's end-to-end latency under a controlled setup, where we repeatedly test how long it takes for vllm to process fixed number of input tokens (32) and generate fixed number of output tokens (128) using a fixed batch size (8).

We cover llama-3 8B model and 70B model, together with mixtral 8x7B model.

We evaluate the performance of this test using the mean, the median, and the p99 of latency.

