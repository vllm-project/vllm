(runai-model-streamer)=

# Loading models with Run:ai Model Streamer

Run:ai Model Streamer is a library to read tensors in concurrency, while streaming it to GPU memory.
Further reading can be found in [Run:ai Model Streamer Documentation](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md).

vLLM supports loading weights in Safetensors format using the Run:ai Model Streamer.
You first need to install vLLM RunAI optional dependency:

```console
pip3 install vllm[runai]
```

To run it as an OpenAI-compatible server, add the `--load-format runai_streamer` flag:

```console
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer
```

To run model from AWS S3 object store run:

```console
vllm serve s3://core-llm/Llama-3-8b --load-format runai_streamer
```

To run model from a S3 compatible object store run:

```console
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_EC2_METADATA_DISABLED=true AWS_ENDPOINT_URL=https://storage.googleapis.com vllm serve s3://core-llm/Llama-3-8b --load-format runai_streamer
```

## Tunable parameters

You can tune parameters using `--model-loader-extra-config`:

You can tune `concurrency` that controls the level of concurrency and number of OS threads reading tensors from the file to the CPU buffer.
For reading from S3, it will be the number of client instances the host is opening to the S3 server.

```console
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"concurrency":16}'
```

You can control the size of the CPU Memory buffer to which tensors are read from the file, and limit this size.
You can read further about CPU buffer memory limiting [here](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit).

```console
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"memory_limit":5368709120}'
```

```{note}
For further instructions about tunable parameters and additional parameters configurable through environment variables, read the [Environment Variables Documentation](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md).
```
