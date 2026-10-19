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

Check out [examples/features/lora/multilora_offline.py](../../examples/features/lora/multilora_offline.py) for an example of how to use LoRA adapters with the async engine and how to use more advanced configuration options.

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

!!! warning
    This feature comes with security risks. It should not be used in production unless it is an isolated, fully trusted environment.

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

You can either install existing plugins or implement your own. By default, vLLM comes with a [resolver plugin to load LoRA adapters from a local directory, as well as a resolver plugin to load LoRA adapters from repositories on Hugging Face Hub](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)
To enable either of these resolvers, you must `set VLLM_ALLOW_RUNTIME_LORA_UPDATING` to True.

- To leverage a local directory, set `VLLM_PLUGINS` to include `lora_filesystem_resolver` and set `VLLM_LORA_RESOLVER_CACHE_DIR` to a local directory. When vLLM receives a request using a LoRA adapter `foobar`,
it will first look in the local directory for a directory `foobar`, and attempt to load the contents of that directory as a LoRA adapter. If successful, the request will complete as normal and that adapter will then be available for normal use on the server.
- To leverage repositories on Hugging Face Hub, set `VLLM_PLUGINS` to include `lora_hf_hub_resolver` and set `VLLM_LORA_RESOLVER_HF_REPO_LIST` to a comma separated list of repository IDs on Hugging Face Hub. When vLLM receives a request for the LoRA adapter `my/repo/subpath`, it will download the adapter at the `subpath` of `my/repo` if it exists and contains an `adapter_config.json`, then build a request to the cached dir for the adapter, similar to the `lora_filesystem_resolver`. Please note that enabling remote downloads is insecure and not intended for use in production environments.

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

### Registering an Adapter from Tensors

An adapter can also be registered without a checkpoint on disk. Inside the worker, typically from a worker extension (the `worker_extension_cls` engine argument), build the `LoRAModel` with `LoRAModel.from_lora_tensors`, the step the path loader runs after reading the files, and add it to the manager:

```python
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper


class LoRARegistrar:  # in an importable module, passed as worker_extension_cls
    def register_lora(self, lora_id: int, adapter_config: dict) -> None:
        tensors = ...  # PEFT-named lora_A/lora_B tensors, produced in the worker
        manager = self.model_runner.get_model().lora_manager
        peft_helper = PEFTHelper.from_dict(adapter_config)
        peft_helper.validate_legal(manager.lora_config)
        mapper = getattr(manager.model, "hf_to_vllm_mapper", None)
        lora = LoRAModel.from_lora_tensors(
            lora_id,
            tensors,
            peft_helper,
            device="cpu",
            dtype=manager.lora_config.lora_dtype,
            model_vocab_size=manager.vocab_size,
            weights_mapper=mapper.get_rename_mapper() if mapper else None,
            skip_prefixes=getattr(manager.model, "lora_skip_prefixes", None),
        )
        manager.add_adapter(lora)
        manager.pin_adapter(lora_id)
```

Requests then reference the adapter as `LoRARequest(name, lora_id, lora_path)` with any non-empty placeholder path: an adapter already registered under that ID is used as is, and the path is only read if the adapter has to be loaded again. Pin it, as above, so it is never evicted, and do not set `load_inplace` on its requests.

### Updating LoRA Weights from Tensors

The LoRA manager of each worker can also replace an adapter's weights from tensors, without writing a checkpoint, or expose the GPU slot that holds them. These calls run inside the worker, typically from a worker extension (the `worker_extension_cls` engine argument) that obtains the tensors itself, e.g. generates them from a seed or receives them through a weight-transfer channel; `collective_rpc` then only carries small arguments:

```python
import torch


class LoRAWriter:  # in an importable module, passed as worker_extension_cls
    def set_down_proj(self, lora_id: int, layer: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        lora_a = torch.randn(8, 3072, generator=g)  # (rank, in_features)
        lora_b = torch.zeros(1024, 8)  # (out_features, rank), scaling folded in
        self.model_runner.lora_manager.update_adapter_weights(
            lora_id, {f"model.layers.{layer}.mlp.down_proj": (lora_a, lora_b)}
        )


llm = LLM(model, enable_lora=True, worker_extension_cls="my_module.LoRAWriter")
llm.collective_rpc("set_down_proj", args=(lora_id, 0, 1234))
```

- `update_adapter_weights(lora_id, weights)` takes a mapping from module name to `(lora_a, lora_b)`, with one entry per slice for packed modules such as `qkv_proj` (`None` keeps a slice). Modules held by another pipeline-parallel rank are skipped, so every worker can receive the same mapping. The cached copy of the adapter is updated too, so the weights survive GPU-slot eviction for as long as the adapter stays registered; an adapter removed from the CPU cache is reloaded from its `lora_path`. Passing `update_cpu_cache=False` writes only the GPU slot, avoiding a device-to-host copy for frequent updates, and requires the adapter to be pinned (`pin_lora`).
- `get_adapter_slot(lora_id)` returns the slot index, and `get_adapter_slot_weights(lora_id)` returns per-module views of the slot's A and B buffers (local to the rank, padded to `max_lora_rank`) for reading or in-place updates under `torch.inference_mode()`. These are views, not checkpoint-format weights: on TP workers, do not pass them as full unsharded inputs to `update_adapter_weights`.

Both APIs support dense and MoE LoRA layers. MoE updates use stacked A `(experts, rank, in_features)` and B `(experts, out_features, rank)` factors, with scaling folded into B:

| MoE layer | Slice order |
| --- | --- |
| Gated, per-expert format | `w1`, `w2`, `w3` |
| Non-gated | `w1`, `w2` |
| 3D fused format | `w13`, `w2` |

`update_adapter_weights` preserves the loaded rank and expects full TP dimensions. For EP, it accepts global-expert tensors and selects this worker's contiguous expert range, or accepts already local-expert tensors. Shared factors keep an expert axis of size one. Slot views expose the actual local buffers, including TP sharding and rank padding. Metadata errors are rejected before any writes; sources aliasing a destination are cloned before the slot is reset.

### Frequent GPU-Resident Updates (ZO / ES)

Zeroth-order optimization and evolution strategies can write new perturbations before every probe forward. Repeated checkpoint loading, CPU-cache updates, and clearing a slot before replacing all of its factors add work to every probe. Pin an active adapter and obtain its views once, then generate directions directly into those views or copy prepared GPU factors into them. This avoids checkpoint I/O, device-to-host copies, intermediate adapter allocation, repeated slot lookup, and reset-then-copy work. The buffers and their addresses stay the same, including when CUDA graphs read them.

For example, the following worker-extension methods prepare a rank-local write plan and overwrite B directly. The adapter must already be loaded and active; its A buffers must contain the intended factors.

```python
class ProbeWriter:
    def prepare_probe(self, lora_id: int, module_names: list[str]) -> None:
        manager = self.model_runner.lora_manager
        manager.pin_adapter(lora_id)
        weights = manager.get_adapter_slot_weights(lora_id, module_names)
        self.probe_b = [b for _, bs in weights.values() for b in bs]

    def zero_probe_b(self) -> None:
        with torch.inference_mode():
            torch._foreach_zero_(self.probe_b)
```

Direct writes affect only the GPU copy. Keep the adapter pinned while retaining views; removal, or eviction after unpinning, allows another adapter to reuse the slot. Overwrite the whole intended region, including unused padded rank entries, so previous perturbations cannot leak into a probe. The adapter's CPU copy stays unchanged; use `update_adapter_weights` when weights must survive eviction. Cache persistence can synchronize CPU/GPU transfers. `update_cpu_cache=False` is a validated GPU-only replacement API, while retained views let a caller generate factors in place without a replacement/reset cycle. Use GPU-resident inputs to avoid host transfers on these paths.

Perform writes between forwards on the worker stream. A worker extension must join any separate producer stream before writing and must not modify buffers while a forward is using them. These methods do not quiesce in-flight requests or implement a concurrent weight-transfer protocol.

Prefix-cache entries computed with an adapter's previous weights are not invalidated by these calls. Call `llm.reset_prefix_cache()` after an update if requests for that adapter may share cached prefixes.

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

## Mixing 2D and 3D MoE LoRA Adapters

To serve 2D-format(based on `megatron`) and 3D-format (based on `peft`) adapters from the same engine instance, start the server with `--enable-mixed-moe-lora-format`
and declare the layout of each adapter explicitly via the `is_3d_lora_weight` field.

Server startup (static modules):

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --enable-lora \
    --enable-mixed-moe-lora-format \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --lora-modules \
        '{"name": "lora-2d", "path": "jeeejeee/qwen36-35ba3b-2d-weights-poken-lora", "is_3d_lora_weight": false}' \
        '{"name": "lora-3d", "path": "jeeejeee/qwen36-35ba3b-moe-all-linear-poken-lora", "is_3d_lora_weight": true}'
```

Dynamic load via `/v1/load_lora_adapter`:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "lora-3d",
    "lora_path": "/path/to/3d-format-lora",
    "is_3d_lora_weight": true
}'
```

!!! warning "You must know your adapter's layout"
    Under `--enable-mixed-moe-lora-format`, vLLM trusts whatever
    `is_3d_lora_weight` the caller declares. Loading checks reject fused 3D
    expert weights routed to a 2D wrapper without the required flags, but
    do not validate every layout mismatch. A wrong declaration can still
    cause loading failures or incorrect outputs. Confirm the layout before serving:

    - **2D (per-expert, megatron-style)** → set `is_3d_lora_weight: false`.
      Adapter keys look like `...experts.{idx}.gate_proj.lora_A.weight`,
      `...experts.{idx}.up_proj.lora_A.weight`,
      `...experts.{idx}.down_proj.lora_A.weight` — one set per expert.
    - **3D (fused, peft-style)** → set `is_3d_lora_weight: true`.
      Adapter keys look like `...experts.gate_up_proj.lora_A.weight`,
      `...experts.down_proj.lora_A.weight` — a single tensor that stacks
      all experts on the leading dim.

When `--enable-mixed-moe-lora-format` is **not** set, `is_3d_lora_weight`
is ignored: vLLM picks the wrapper from the base model's
`is_3d_moe_weight` and the adapter is required to match. The field is
also ignored for non-MoE models.

For a model using a 2D MoE wrapper (such as Qwen3-MoE), loading fused 3D
expert adapter weights requires both `enable_mixed_moe_lora_format=True`
on the engine and `is_3d_lora_weight=True` on the request. Otherwise, adapter
loading raises a `ValueError` naming these settings before activation.

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

## Sequence-Classification LoRA Adapters

vLLM supports PEFT sequence-classification adapters that save a complete, single-layer linear classification head through `modules_to_save`. The saved module must be named `score` or `classifier`.

See [classification_with_lora_offline.py](../../examples/pooling/classify/classification_with_lora_offline.py) for an offline classification example using a LoRA adapter.

To batch adapters with different `num_labels`, set the maximum number of labels:

```bash
vllm serve model --enable-lora --max-lora-cls-labels 8
```

The equivalent `LLM` argument is `max_lora_cls_labels`. It defaults to the base model's `num_labels`, and each request returns its adapter's number of labels.

This support has the following limitations:

- A classification head stored as float32 is converted to the runtime head dtype when it is loaded.
- Token-classification adapters are not supported by this feature.

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

### Restricting LoRA to Specific Modules

The `--lora-target-modules` parameter allows you to restrict which model modules have LoRA applied at deployment time. This is useful for performance tuning when you only need LoRA on specific layers:

```bash
# Apply LoRA only to output projection layers
vllm serve model --enable-lora --lora-target-modules o_proj

# Apply LoRA to multiple specific modules
vllm serve model --enable-lora --lora-target-modules o_proj qkv_proj down_proj
```

When `--lora-target-modules` is not specified, LoRA will be applied to all supported modules in the model. This parameter accepts module suffixes (the last component of the module name), such as `o_proj`, `qkv_proj`, `gate_proj`, etc.
