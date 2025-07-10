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

To create sharded model files, you can use the script provided in <gh-file:examples/offline_inference/save_sharded_state.py>. This script demonstrates how to save a model in the sharded format that is compatible with the Run:ai Model Streamer sharded loader.

The sharded loader supports all the same tunable parameters as the regular Run:ai Model Streamer, including `concurrency` and `memory_limit`. These can be configured in the same way:

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

!!! note
    The sharded loader is particularly efficient for tensor or pipeline parallel models where each worker only needs to read its own shard rather than the entire checkpoint.
