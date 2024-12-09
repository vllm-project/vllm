# SwiftKV on vLLM

SwiftKV is a technique developed by Snowflake AI Research that reduces computational overhead during prompt processing by combining model rewiring and knowledge-preserving self-distillation.

For more details, see:

- [Blog post](https://www.snowflake.com/engineering-blog/swiftkv-llm-compute-reduction)
- [Paper](https://arxiv.org/abs/2410.03960)
- [Huggingface](https://huggingface.co/collections/Snowflake/swiftkv-models-674f7d7474eb789e185d31cb)

## Quickstart

Run an example conversation using [Snowflake/Llama-3.1-SwiftKV-8B-Instruct](https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct):
```console
$ python examples/swiftkv/offline_inference_swiftkv.py

...

The Importance of Higher Education

Higher education is a vital component of an individual's life, providing numerous benefits that extend beyond the acquisition of knowledge and skills. It plays a significant role in shaping an individual's future, career prospects, and overall well-being. In this essay, we will explore the importance of higher education and its far-reaching implications on individuals, society, and the economy.

...
```

## Running Accuracy Evaluations

To evaluate the Llama-3.1-SwiftKV models, we use the [LM-Eval fork by NeuralMagic](https://github.com/neuralmagic/lm-evaluation-harness.git):

```console
$ pip install git+https://github.com/neuralmagic/lm-evaluation-harness.git@llama_3.1_instruct
```

Run evaluation on Llama-3.1-SwiftKV-8B-Instruct:

```console
$ bash examples/swiftkv/run_eval_8b.sh
```

Run evaluation on Llama-3.1-SwiftKV-405B-Instruct-FP8:

```console
$ bash examples/swiftkv/run_eval_405b_fp8.sh
```

## Running Performance Benchmarks

Llama-3.1-SwiftKV-8B-Instruct

```console
$ python benchmarks/benchmark_throughput.py \
    --input-len 2000 --output-len 256 \
    --model Snowflake/Llama-3.1-SwiftKV-8B-Instruct \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 512 

...

Throughput: 11.36 requests/s, 25635.51 total tokens/s, 2908.99 output tokens/s
```

Llama-3.1-SwiftKV-405B-Instruct-FP8

```console
$ python benchmarks/benchmark_throughput.py \
    --input-len 2000 --output-len 256 \
    --model Snowflake/Llama-3.1-SwiftKV-405B-Instruct-FP8 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 512 \
    --tensor-parallel-size 8

...

Throughput: 3.21 requests/s, 7233.37 total tokens/s, 820.81 output tokens/s
```
