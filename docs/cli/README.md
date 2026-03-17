# vLLM CLI Guide

The vllm command-line tool is used to run and manage vLLM models. You can start by viewing the help message with:

```bash
vllm --help
```

Available Commands:

```bash
vllm {pull,run,ls,list,aliases,inspect,serve,ps,stop,logs,rm,chat,complete,bench,collect-env,run-batch}
```

## Local Runtime

The local-runtime commands are designed to make a source checkout feel more like a local model runner:

```bash
./scripts/install.sh
vllm aliases
vllm pull deepseek-r1:8b
vllm run deepseek-r1:8b
vllm serve deepseek-r1:8b
vllm ps
```

### Main Changes

Compared with the older server-first CLI flow, the main additions are:

- A one-script repo install path via `./scripts/install.sh`
- A lightweight launcher so help and local metadata commands do not have to pull in the full runtime stack
- Direct shell usage with `vllm run` instead of requiring a separate running server first
- Built-in model aliases such as `deepseek-r1:8b`
- Background local service management through `vllm serve`, `vllm ps`, `vllm stop`, and `vllm logs`

The advanced vLLM commands still exist; this changes the default user journey rather than removing the underlying engine/server features.

### Fast Path

For a terminal-first workflow, the intended path is:

```bash
./scripts/install.sh
vllm aliases
vllm pull deepseek-r1:8b
vllm run deepseek-r1:8b
```

### aliases

List the built-in easy-model names that resolve to Hugging Face repos.

```bash
vllm aliases
```

### pull

Resolve a model alias or exact Hugging Face repo and pre-download the model locally.

```bash
vllm pull deepseek-r1:8b
vllm pull meta-llama/Llama-3.1-8B-Instruct
```

### run

Run a model directly in your shell without starting a separate API server first.

```bash
vllm run deepseek-r1:8b
vllm run deepseek-r1:8b --prompt "Explain KV cache in one paragraph."
vllm run meta-llama/Llama-3.1-8B-Instruct --complete --prompt "The future of inference is"
```

### ls, list, and inspect

Show pulled model metadata and how aliases resolve.

```bash
vllm ls
vllm list
vllm inspect deepseek-r1:8b
```

### Built-in Easy Models

The built-in aliases currently include:

- `deepseek-r1:1.5b`, `deepseek-r1:7b`, `deepseek-r1:8b`, `deepseek-r1:14b`, `deepseek-r1:32b`, `deepseek-r1:70b`, `deepseek-v3`
- `llama3.2:1b-instruct`, `llama3.2:3b-instruct`, `llama3.1:8b-instruct`, `llama3.1:70b-instruct`, `llama3.3:70b-instruct`
- `qwen2.5:0.5b-instruct`, `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`, `qwen2.5:32b-instruct`, `qwen2.5:72b-instruct`
- `qwen2.5-coder:1.5b-instruct`, `qwen2.5-coder:7b-instruct`, `qwen2.5-coder:32b-instruct`
- `mistral:7b-instruct`, `ministral:8b-instruct`, `mistral-nemo:12b-instruct`, `mixtral:8x7b-instruct`
- `gemma2:2b-it`, `gemma2:9b-it`, `gemma2:27b-it`
- `phi3.5:mini-instruct`, `phi3.5:moe-instruct`, `phi4`
- `smollm2:360m-instruct`, `smollm2:1.7b-instruct`

## serve

Starts the vLLM OpenAI Compatible API server.

Start with a model:

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

By default, `vllm serve` now starts a managed background service. Use `--foreground` to keep the previous blocking behavior:

```bash
vllm serve deepseek-r1:8b
vllm serve meta-llama/Llama-3.1-8B-Instruct --foreground
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
# To list all groups
vllm serve --help=listgroup

# To view a argument group
vllm serve --help=ModelConfig

# To view a single argument
vllm serve --help=max-num-seqs

# To search by keyword
vllm serve --help=max

# To view full help with pager (less/more)
vllm serve --help=page
```

See [vllm serve](./serve.md) for the full reference of all available arguments.

### ps, stop, logs, rm

Manage local background services and pulled model metadata.

```bash
vllm ps
vllm logs deepseek-r1-8b
vllm stop deepseek-r1-8b
vllm rm deepseek-r1:8b
```

## chat

Generate chat completions via the running API server.

```bash
# Directly connect to localhost API without arguments
vllm chat

# Specify API url
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick chat with a single prompt
vllm chat --quick "hi"
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
```

See [vllm complete](./complete.md) for the full reference of all available arguments.

## bench

Run benchmark tests for latency online serving throughput and offline inference throughput.

To use benchmark commands, please install with extra dependencies using `pip install vllm[bench]`.

Available Commands:

```bash
vllm bench {latency, serve, throughput}
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
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

Using remote file:

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

See [vllm run-batch](./run-batch.md) for the full reference of all available arguments.

## More Help

For detailed options of any subcommand, use:

```bash
vllm <subcommand> --help
```

Tracked follow-up items for the local-runtime UX live in [local_runtime_followups.md](./local_runtime_followups.md).
