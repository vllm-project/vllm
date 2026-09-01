# vLLM CLI Guide

The vllm command-line tool is used to run and manage vLLM models. You can start by viewing the help message with:

```bash
vllm --help
```

Available Commands:

```bash
vllm {chat,complete,serve,launch,bench,collect-env,run-batch}
```

## serve

Starts the vLLM OpenAI Compatible API server.

Start with a model:

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

Specify the port:

```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8100
```

Serve over a Unix domain socket:

```bash
vllm serve meta-llama/Llama-2-7b-hf --uds /tmp/vllm.sock
```

Check with --help for more options:

```bash
# To list all flags
vllm serve --help=all

# To view an argument group
vllm serve --help=ModelConfig

# To view a single argument
vllm serve --help=max-num-seqs

# To search by keyword or flag name
vllm serve --help=max
```

!!! tip "Human-readable integer arguments"
    Many integer arguments accept human-readable suffixes for convenience. For example:

    - `1k` = 1,000 (decimal kilo)
    - `1K` = 1,024 (binary kibibyte)
    - `1m` = 1,000,000 (decimal mega)
    - `1M` = 1,048,576 (binary mebibyte)
    - `1g` / `1G` = 1 billion / 1 gibibyte
    - `1t` / `1T` = 1 trillion / 1 tebibyte
    
    Decimal suffixes (`k`, `m`, `g`, `t`) also accept floating point: `25.6k` = 25,600.
    Binary suffixes (`K`, `M`, `G`, `T`) require integers: `32K` = 32,768.
    
    Supported arguments include: `--max-model-len`, `--max-num-batched-tokens`, `--max-num-scheduled-tokens`, `--kv-cache-memory-bytes`, `--safetensors-prefetch-block-size`.

See [vllm serve](./serve.md) for the full reference of all available arguments.

## launch

Launch individual vLLM components.

```bash
# Launch the rendering server component
vllm launch render meta-llama/Llama-3.2-1B-Instruct

# Inspect all available flags for the render component
vllm launch render --help=all
```

See [vllm launch render](./launch/render.md) for the current launch
component reference.

## chat

Generate chat completions via the running API server.

```bash
# Directly connect to localhost API without arguments
vllm chat

# Specify API url
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick chat with a single prompt
vllm chat --quick "hi"

# Print TTFT and throughput statistics after each response
vllm chat --stats
```

See [vllm chat](./chat.md) for the full reference of all available arguments.

## complete

Generate text completions based on the given prompt via the running API server.

```bash
# Directly connect to localhost API without arguments
vllm complete

# Specify API url
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick complete with a single prompt
vllm complete --quick "The future of AI is"

# Print TTFT and throughput statistics after each response
vllm complete --stats
```

See [vllm complete](./complete.md) for the full reference of all available arguments.

## bench

Run benchmark tests for latency, online serving throughput, offline inference throughput, startup time, multimodal processor latency, and parameter sweeps.

To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

Available Commands:

```bash
vllm bench {latency, serve, throughput, startup, mm-processor, sweep}
```

### latency

Benchmark the latency of a single batch of requests.

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

See [vllm bench latency](./bench/latency.md) for the full reference of all available arguments.

### serve

Benchmark the online serving throughput.

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

See [vllm bench serve](./bench/serve.md) for the full reference of all available arguments.

### throughput

Benchmark offline inference throughput.

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

See [vllm bench throughput](./bench/throughput.md) for the full reference of all available arguments.

### startup

Benchmark the startup time of vLLM models.

```bash
vllm bench startup \
    --model meta-llama/Llama-3.2-1B-Instruct
```

See [vllm bench startup](./bench/startup.md) for the full reference of all available arguments.

### mm-processor

Benchmark multimodal processor latency across different configurations.

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 50 \
    --random-input-len 300 \
    --random-output-len 40 \
    --random-mm-base-items-per-request 2 \
    --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
    --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

See [vllm bench mm-processor](./bench/mm-processor.md) for the full reference of all available arguments.

### sweep

Benchmark for a parameter sweep.

Available Commands:

```bash
vllm bench sweep {serve, serve_workload, startup, plot, plot_pareto}
```

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --output-dir benchmarks/results \
    --experiment-name demo
```

See [vllm bench sweep](./bench/sweep.md) for the full reference of all available arguments.

## collect-env

Start collecting environment information.

```bash
vllm collect-env
```

## run-batch

Run batch prompts and write results to file.

Running with a local file:

```bash
vllm run-batch \
    -i examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

Using remote file:

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

See [vllm run-batch](./run-batch.md) for the full reference of all available arguments.

## More Help

For detailed options of any subcommand, use:

```bash
vllm <subcommand> --help
```
