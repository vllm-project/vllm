# LoRA Adapters

This document shows you how to use [LoRA adapters](https://arxiv.org/abs/2106.09685) with vLLM on top of a base model.

LoRA adapters can be used with any vLLM model that implements [SupportsLoRA][vllm.model_executor.models.interfaces.SupportsLoRA].

Adapters can be efficiently served on a per-request basis with minimal overhead. First we download the adapter(s) and save
them locally with

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
```

Then we instantiate the base model and pass in the `enable_lora=True` flag:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)
```

We can now submit the prompts and call `llm.generate` with the `lora_request` parameter. The first parameter
of `LoRARequest` is a human identifiable name, the second parameter is a globally unique ID for the adapter and
the third parameter is the path to the LoRA adapter.

??? code

    ```python
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )
    ```

Check out [examples/offline_inference/multilora_inference.py](../../examples/offline_inference/multilora_inference.py) for an example of how to use LoRA adapters with the async engine and how to use more advanced configuration options.

## Serving LoRA Adapters

LoRA adapted models can also be served with the Open-AI compatible vLLM server. To do so, we use
`--lora-modules {name}={path} {name}={path}` to specify each LoRA module when we kick off the server:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

The server entrypoint accepts all other LoRA configuration parameters (`max_loras`, `max_lora_rank`, `max_cpu_loras`,
etc.), which will apply to all forthcoming requests. Upon querying the `/models` endpoint, we should see our LoRA along
with its base model (if `jq` is not installed, you can follow [this guide](https://jqlang.org/download/) to install it.):

??? console "Command"

    ```bash
    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }
    ```

Requests can specify the LoRA adapter as if it were any other model via the `model` request parameter. The requests will be
processed according to the server-wide LoRA configuration (i.e. in parallel with base model requests, and potentially other
LoRA adapter requests if they were provided and `max_loras` is set high enough).

The following is an example request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## Dynamically serving LoRA Adapters

In addition to serving LoRA adapters at server startup, the vLLM server supports dynamically configuring LoRA adapters at runtime through dedicated API endpoints and plugins. This feature can be particularly useful when the flexibility to change models on-the-fly is needed.

Note: Enabling this feature in production environments is risky as users may participate in model adapter management.

To enable dynamic LoRA configuration, ensure that the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
is set to `True`.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### Using API Endpoints

Loading a LoRA Adapter:

To dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary
details of the adapter to be loaded. The request payload should include the name and path to the LoRA adapter.

Example request to load a LoRA adapter:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

Upon a successful request, the API will respond with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' added successfully`. If an error occurs, such as if the adapter
cannot be found or loaded, an appropriate error message will be returned.

Unloading a LoRA Adapter:

To unload a LoRA adapter that has been previously loaded, send a POST request to the `/v1/unload_lora_adapter` endpoint
with the name or ID of the adapter to be unloaded.

Upon a successful request, the API responds with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' removed successfully`.

Example request to unload a LoRA adapter:

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### Using Plugins

Alternatively, you can use the LoRAResolver plugin to dynamically load LoRA adapters. LoRAResolver plugins enable you to load LoRA adapters from both local and remote sources such as local file system and S3. On every request, when there's a new model name that hasn't been loaded yet, the LoRAResolver will try to resolve and load the corresponding LoRA adapter.

You can set up multiple LoRAResolver plugins if you want to load LoRA adapters from different sources. For example, you might have one resolver for local files and another for S3 storage. vLLM will load the first LoRA adapter that it finds.

You can either install existing plugins or implement your own. By default, vLLM comes with a [resolver plugin to load LoRA adapters from a local directory.](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)
To enable this resolver, set `VLLM_ALLOW_RUNTIME_LORA_UPDATING` to True, set `VLLM_PLUGINS` to include `lora_filesystem_resolver`, and then set `VLLM_LORA_RESOLVER_CACHE_DIR` to a local directory. When vLLM receives a request using a LoRA adapter `foobar`,
it will first look in the local directory for a directory `foobar`, and attempt to load the contents of that directory as a LoRA adapter. If successful, the request will complete as normal and
that adapter will then be available for normal use on the server.

Alternatively, follow these example steps to implement your own plugin:

1. Implement the LoRAResolver interface.

    ??? code "Example of a simple S3 LoRAResolver implementation"

        ```python
        import os
        import s3fs
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver

        class S3LoRAResolver(LoRAResolver):
            def __init__(self):
                self.s3 = s3fs.S3FileSystem()
                self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
                self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

            async def resolve_lora(self, base_model_name, lora_name):
                s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
                local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

                # Download the LoRA from S3 to the local path
                await self.s3._get(
                    s3_path, local_path, recursive=True, maxdepth=1
                )

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=local_path,
                    lora_int_id=abs(hash(lora_name)),
                )
                return lora_request
        ```

2. Register `LoRAResolver` plugin.

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    For more details, refer to the [vLLM's Plugins System](../design/plugin_system.md).

### In-Place LoRA Reloading

When dynamically loading LoRA adapters, you may need to replace an existing adapter with updated weights while keeping the same name. The `load_inplace` parameter enables this functionality. This commonly occurs in asynchronous reinforcement learning setups, where adapters are continuously updated and swapped in without interrupting ongoing inference.

When `load_inplace=True`, vLLM will replace the existing adapter with the new one.

Example request to load or replace a LoRA adapter with the same name:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter/v2",
    "load_inplace": true
}'
```

## New format for `--lora-modules`

In the previous version, users would provide LoRA modules via the following format, either as a key-value pair or in JSON format. For example:

```bash
--lora-modules  sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

This would only include the `name` and `path` for each LoRA module, but did not provide a way to specify a `base_model_name`.
Now, you can specify a base_model_name alongside the name and path using JSON format. For example:

```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

To provide the backward compatibility support, you can still use the old key-value format (name=path), but the `base_model_name` will remain unspecified in that case.

## LoRA model lineage in model card

The new format of `--lora-modules` is mainly to support the display of parent model information in the model card. Here's an explanation of how your current response supports this:

- The `parent` field of LoRA model `sql-lora` now links to its base model `meta-llama/Llama-3.2-3B-Instruct`. This correctly reflects the hierarchical relationship between the base model and the LoRA adapter.
- The `root` field points to the artifact location of the lora adapter.

??? console "Command output"

    ```bash
    $ curl http://localhost:8000/v1/models

    {
        "object": "list",
        "data": [
            {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "meta-llama/Llama-3.2-3B-Instruct",
            "parent": null,
            "permission": [
                {
                .....
                }
            ]
            },
            {
            "id": "sql-lora",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "jeeejeee/llama32-3b-text2sql-spider",
            "parent": "meta-llama/Llama-3.2-3B-Instruct",
            "permission": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```

## LoRA Support for Tower and Connector of Multi-Modal Model

Currently, vLLM experimentally supports LoRA for the Tower and Connector components of multi-modal models. To enable this feature, you need to implement the corresponding token helper functions for the tower and connector. For more details on the rationale behind this approach, please refer to [PR 26674](https://github.com/vllm-project/vllm/pull/26674). We welcome contributions to extend LoRA support to additional models' tower and connector. Please refer to [Issue 31479](https://github.com/vllm-project/vllm/issues/31479) to check the current model support status.

## Default LoRA Models For Multimodal Models

Some models, e.g., [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) and [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) multimodal, contain LoRA adapter(s) that are expected to always be applied when a given modality is present. This can be a bit tedious to manage with the above approaches, as it requires the user to send the `LoRARequest` (offline) or to filter requests between the base model and LoRA model (server) depending on the content of the request's multimodal data.

To this end, we allow registration of default multimodal LoRAs to handle this automatically, where users can map each modality to a LoRA adapter to automatically apply it when the corresponding inputs are present. Note that currently, we only allow one LoRA per prompt; if several modalities are provided, each of which are registered to a given modality, none of them will be applied.

??? code "Example usage for offline inference"

    ```python
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    model_id = "ibm-granite/granite-speech-3.3-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_prompt(question: str, has_audio: bool):
        """Build the input prompt to send to vLLM."""
        if has_audio:
            question = f"<|audio|>{question}"
        chat = [
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False)


    llm = LLM(
        model=model_id,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
        # Will always pass a `LoRARequest` with the `model_id`
        # whenever audio is contained in the request data.
        default_mm_loras = {"audio": model_id},
        enforce_eager=True,
    )

    question = "can you transcribe the speech into a written format?"
    prompt_with_audio = get_prompt(
        question=question,
        has_audio=True,
    )
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

    inputs = {
        "prompt": prompt_with_audio,
        "multi_modal_data": {
            "audio": audio,
        }
    }


    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=0.2,
            max_tokens=64,
        ),
    )
    ```

You can also pass a json dictionary of `--default-mm-loras` mapping modalities to LoRA model IDs. For example, when starting the server:

```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --max-model-len 2048 \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

Note: Default multimodal LoRAs are currently only available for `.generate` and chat completions.

## Using Tips

### Configuring `max_lora_rank`

The `--max-lora-rank` parameter controls the maximum rank allowed for LoRA adapters. This setting affects memory allocation and performance:

- **Set it to the maximum rank** among all LoRA adapters you plan to use
- **Avoid setting it too high** - using a value much larger than needed wastes memory and can cause performance issues

For example, if your LoRA adapters have ranks [16, 32, 64], use `--max-lora-rank 64` rather than 256

```bash
# Good: matches actual maximum rank
vllm serve model --enable-lora --max-lora-rank 64

# Bad: unnecessarily high, wastes memory
vllm serve model --enable-lora --max-lora-rank 256
```
