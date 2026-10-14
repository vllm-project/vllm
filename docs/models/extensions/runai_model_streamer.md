# Loading models with Run:ai Model Streamer

Run:ai Model Streamer is a library to read tensors in concurrency, while streaming it to GPU memory.
Further reading can be found in [Run:ai Model Streamer Documentation](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md).

vLLM supports loading weights in Safetensors format using the Run:ai Model Streamer.
You first need to install vLLM RunAI optional dependency:

```bash
pip3 install vllm[runai]
```

To run it as an OpenAI-compatible server, add the `--load-format runai_streamer` flag:

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer
```

To run model from AWS S3 object store run:

```bash
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

To run model from Google Cloud Storage run:

```bash
vllm serve gs://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

To run model from Azure Blob Storage run:

```bash
AZURE_STORAGE_ACCOUNT_NAME=<account> \
vllm serve az://<container>/<model-path> \
    --load-format runai_streamer
```

Authentication uses `DefaultAzureCredential`, which supports `az login`, managed identity, environment variables (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`), and other methods.

To run model from a S3 compatible object store run:

```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=true \
AWS_ENDPOINT_URL=https://storage.googleapis.com \
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

## Tunable parameters

You can tune parameters using `--model-loader-extra-config`:

You can tune `distributed` that controls whether distributed streaming should be used. This is currently only possible on CUDA and ROCM devices. This can significantly improve loading times from object storage or high-throughput network fileshares.
You can read further about Distributed streaming [here](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/usage.md#distributed-streaming)

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"distributed":true}'
```

You can tune `concurrency` that controls the level of concurrency and number of OS threads reading tensors from the file to the CPU buffer.
For reading from S3, it will be the number of client instances the host is opening to the S3 server.

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16}'
```

You can control the size of the CPU Memory buffer to which tensors are read from the file, and limit this size.
You can read further about CPU buffer memory limiting [here](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit).

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"memory_limit":5368709120}'
```

!!! note
    For further instructions about tunable parameters and additional parameters configurable through environment variables, read the [Environment Variables Documentation](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md).

### Experimental DeepSeek V4.1 shared CPU loading

The mixed loading path keeps ordinary weights on the distributed GPU streamer
and reads oversized Engram shards on CPU. With CPU Engram storage shared across
DP replicas, one rank reads the shared tables and scales. Other ranks receive
metadata placeholders and participate in the existing weight-loader barriers.
Non-shared tensors in those shards are broadcast as CPU bytes and loaded on each
rank. The CPU staging limit is restored after this phase.

This experimental path requires Run:ai Model Streamer 0.17.0 and the
[non-distributed rank-discovery fix](https://github.com/dsx-ai-factory/model-streamer/pull/252).
It was validated through startup with TP1/DP8/EP8 on eight co-located H200 GPUs.
Request correctness, broader topologies and a generic configuration API remain
pending. The wrapper and DSpark changes in this patch retain streamed tensor
lifetimes and use the target's object-storage source for the draft.

For checkpoint revision `5254448`, verify that the checkpoint index places the
Engram tables in the two filenames below before enabling these settings:

```bash
export RUNAI_STREAMER_PARTITION_POLICY=spans
export VLLM_RUNAI_CPU_WEIGHT_FILES=model-00047-of-00048.safetensors,model-00048-of-00048.safetensors
export VLLM_RUNAI_CPU_MEMORY_LIMIT=107374182400
export VLLM_RUNAI_SINGLE_ENGRAM_READER=1
export VLLM_RUNAI_BORROWED_WEIGHTS=1
export VLLM_RUNAI_USE_LOCAL_WEIGHT_INDEX=1
vllm serve s3://bucket/deepseek-v4-1-flash/ \
    --load-format runai_streamer \
    --model-loader-extra-config '{"distributed":true,"memory_limit":34359738368}' \
    --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
    --engram-config '{"cpu_offload":true,"embedding_across_dp":true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

In streamer 0.17.0, 32 GiB is the distributed node staging budget, giving eight
readers 4 GiB each. The CPU phase uses a separate 100 GiB budget. These limits
control checkpoint staging, independently of KV cache and KV offload settings.
Borrowed GPU tensors still receive vLLM's normal clone; shared CPU tensors are
consumed synchronously. The local index must match the actual checkpoint and its
shard headers. These environment flags are experimental; they are disabled
unless explicitly set.

## Sharded Model Loading

vLLM also supports loading sharded models using Run:ai Model Streamer. This is particularly useful for large models that are split across multiple files. To use this feature, use the `--load-format runai_streamer_sharded` flag:

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

The sharded loader expects model files to follow the same naming pattern as the regular sharded state loader: `model-rank-{rank}-part-{part}.safetensors`. You can customize this pattern using the `pattern` parameter in `--model-loader-extra-config`:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"pattern":"custom-model-rank-{rank}-part-{part}.safetensors"}'
```

To create sharded model files, you can use the script provided in [examples/features/sharded_state/save_sharded_state_offline.py](../../../examples/features/sharded_state/save_sharded_state_offline.py). This script demonstrates how to save a model in the sharded format that is compatible with the Run:ai Model Streamer sharded loader.

The sharded loader supports all the same tunable parameters as the regular Run:ai Model Streamer, including `concurrency` and `memory_limit`. These can be configured in the same way:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

!!! note
    The sharded loader is particularly efficient for tensor or pipeline parallel models where each worker only needs to read its own shard rather than the entire checkpoint.
