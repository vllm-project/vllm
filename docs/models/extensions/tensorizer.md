---
title: Loading models with CoreWeave's Tensorizer
---
[](){ #tensorizer }

vLLM supports loading models with [CoreWeave's Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer).
vLLM model tensors that have been serialized to disk, an HTTP/HTTPS endpoint, or S3 endpoint can be deserialized
at runtime extremely quickly directly to the GPU, resulting in significantly
shorter Pod startup times and CPU memory usage. Tensor encryption is also supported.

vLLM fully integrates Tensorizer in to its model loading machinery. The 
following will give a brief overview on how to get started with using 
Tensorizer on vLLM.

## The basics
To load a model using Tensorizer, it first needs to be serialized by Tensorizer.
The example script in [examples/others/tensorize_vllm_model.py] takes care of 
this process.
(https://docs.vllm.ai/en/latest/examples/others/tensorize_vllm_model.html)

Let's walk through a basic example by serializing `facebook/opt-125m` using the
script, and then loading it for inference.

## Saving a vLLM model with Tensorizer
To save a model with Tensorizer, call the example script with the necessary
CLI arguments. The docstring for the script itself explains the CLI args
and how to use it properly in great detail, and we'll use one of the 
examples from the docstring directly, assuming we want to save our model at 
our S3 bucket example `s3://my-bucket`:

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

This saves the model tensors at `s3://my-bucket/vllm/facebook/opt-125m/v1`. If 
you intend on applying a LoRA adapter to your tensorized model, you can pass 
the HF id of the LoRA adapter in the above command, and the artifacts will be 
saved there too:

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

## Serving the model using Tensorizer
Once the model is serialized where you want it, you can load the model using 
`vllm serve` or the `LLM` entrypoint. The directory where the 
model artifacts were saved can be passed to the `model` argument for 
`LLM()` and `vllm serve`. For example, to serve the tensorized model 
saved previously with the LoRA adapter, you'd do:

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora 
```

Or, with `LLM()`:

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True
)
```

`tensorizer`'s core objects that serialize and deserialize models are 
`TensorSerializer` and `TensorDeserializer` respectively. In order to 
pass arbitrary kwargs to these, which will configure the serialization 
and deserialization processes, you can provide them as keys to 
`model_loader_extra_config` with `serialization_kwargs` and 
`deserialization_kwargs` respectively. Full docstrings detailing all 
parameters for the aforementioned objects can be found in `tensorizer`'s
[serialization.py](https://github.
com/coreweave/tensorizer/blob/main/tensorizer/serialization.py) file.

As an example, CPU concurrency can be limited when serializing with 
`tensorizer` via the `limit_cpu_concurrency` parameter in the 
initializer for `TensorSerializer`. To set `limit_cpu_concurrency` to 
some arbitrary value, you would do so like this when serializing:

```bash
python examples/others/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --serialization-kwargs '{"limit_cpu_concurrency": 2}' \
   --suffix v1
```

As an example when customizing the loading process via `TensorDeserializer`, 
one could limit the number of concurrency readers during 
deserialization with the `num_readers` parameter in the initializer 
via `model_loader_extra_config` like so:

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora \
    --model-loader-extra-config '{"deserialization_kwargs": {"num_readers": 2}}'
```

Or with `LLM()`:

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
    model_loader_extra_config={"deserialization_kwargs": {"num_readers": 2}}
)
```



!!! note
    Note that to use this feature you will need to install `tensorizer` by running `pip install vllm[tensorizer]`.
