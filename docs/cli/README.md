# vLLM CLI Guide

The vllm command-line tool is used to run and manage vLLM models. You can start by viewing the help message with:

```
vllm --help
```

Available Commands:

```
vllm {chat,complete,serve,bench,collect-env,run-batch}
```

## serve

Start the vLLM OpenAI Compatible API server.

Examples:

```bash
# Start with a model
vllm serve meta-llama/Llama-2-7b-hf

# Specify the port
vllm serve meta-llama/Llama-2-7b-hf --port 8100

# Check with --help for more options
# To list all groups
vllm serve --help=listgroup

# To view a argument group
vllm serve --help=ModelConfig

# To view a single argument
vllm serve --help=max-num-seqs

# To search by keyword
vllm serve --help=max
```

## chat

Generate chat completions via the running API server.

Examples:

```bash
# Directly connect to localhost API without arguments
vllm chat

# Specify API url
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick chat with a single prompt
vllm chat --quick "hi"
```

## complete

Generate text completions based on the given prompt via the running API server.

Examples:

```bash
# Directly connect to localhost API without arguments
vllm complete

# Specify API url
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick complete with a single prompt
vllm complete --quick "The future of AI is"
```

## bench

Run benchmark tests for latency online serving throughput and offline inference throughput.

To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

Available Commands:

```bash
vllm bench {latency, serve, throughput}
```

### latency

Benchmark the latency of a single batch of requests.

Example:

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

### serve

Benchmark the online serving throughput.

Example:

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

### throughput

Benchmark offline inference throughput.

Example:

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

## collect-env

Start collecting environment information.

```bash
vllm collect-env
```

## run-batch

Run batch prompts and write results to file.

Examples:

```bash
# Running with a local file
vllm run-batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Using remote file
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## More Help

For detailed options of any subcommand, use:

```bash
vllm <subcommand> --help
```
