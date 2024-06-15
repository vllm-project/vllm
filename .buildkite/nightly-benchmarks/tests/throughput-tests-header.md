

## Throughput tests

This test suite aims to test vllm's throughput.


Concretely, we sample 200 (input, output) pairs from ShareGPT dataset using a fixed random seed. Then, for each (input, output) pair, we create a dummy input with the same length, and request vllm to generate the same amount of tokens as the output.
We will then forward all these inputs to vllm, and see how fast can vllm finish processing these inputs.

We cover llama-3 8B model and 70B model, together with mixtral 8x7B model.


We use throughput (number of requests processed per second) as the evaluation metric.
