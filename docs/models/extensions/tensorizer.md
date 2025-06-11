---
title: Loading models with CoreWeave's Tensorizer
---
[](){ #tensorizer }

vLLM supports loading models with [CoreWeave's Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer).
vLLM model tensors that have been serialized to disk, an HTTP/HTTPS endpoint, or S3 endpoint can be deserialized
at runtime extremely quickly directly to the GPU, resulting in significantly
lower Pod startup times and CPU memory usage. Tensor encryption is also supported.

vLLM fully integrates Tensorizer in its model loading machinery. The
following will give a brief overview on how to get started with using
Tensorizer on vLLM.

## The basics
To load a model using Tensorizer, it first needs to be serialized by Tensorizer.
The example script in [examples/others/tensorize_vllm_model.py](https://github.com/vllm-project/vllm/blob/main/examples/others/tensorize_vllm_model.py)
takes care of this process.

The `TensorizerConfig` class is used to customize Tensorizer's behaviour,
defined in [vllm/model_executor/model_loader/tensorizer.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/tensorizer.py).
It is passed to any serialization or deserialization operation.
When loading with Tensorizer using the vLLM 
library rather than through a model-serving entrypoint, it gets passed to 
the `LLM` entrypoint class directly. Here's an example of loading a model
saved at `"s3://my-bucket/vllm/facebook/opt-125m/v1/model.tensors"`:

```python
from vllm import LLM
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

path_to_tensors = "s3://my-bucket/vllm/facebook/opt-125m/v1/model.tensors"

model_ref = "facebook/opt-125m"
tensorizer_config = TensorizerConfig(
    tensorizer_uri=path_to_tensors
)

llm = LLM(
    model_ref,
    load_format="tensorizer",
    model_loader_extra_config=tensorizer_config,
)
```

However, the above code will not function until you have successfully
serialized the model tensors with Tensorizer to get the `model.tensors`
file shown. The following section walks through an end-to-end example
of serializing `facebook/opt-125m` with the example script,
and then loading it for inference.

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

This saves the model tensors at 
`s3://my-bucket/vllm/facebook/opt-125m/v1/model.tensors`, as well as all other 
artifacts needed to load the model at that same directory.

## Serving the model using Tensorizer
Once the model is serialized where you want it, all that is needed is to 
pass that directory with the model artifacts to `vllm serve` in the case above, 
one can simply do:

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 --load-format=tensorizer
```

Please note that object storage authentication is still required.
To authenticate, Tensorizer searches for a `~/.s3cfg` configuration file
([s3cmd](https://s3tools.org/kb/item14.htm)'s format),
or `~/.aws/config` and `~/.aws/credentials` files (`boto3` and the
`aws` CLI's format), but authentication can also be configured using
[any normal `boto3` environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables),
such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
and `AWS_ENDPOINT_URL_S3`.

If only the model tensors are saved, you can still provide that as the only 
artifact in your directory to load from, and vLLM will fetch the rest using the 
`--model-loader-extra-config` CLI arg, passing a JSON string with the args 
for `TensorizerConfig`.

```bash
#!/bin/sh

MODEL_LOADER_EXTRA_CONFIG='{
  "tensorizer_uri": "s3://my-bucket/vllm/facebook/opt-125m/v1/model.tensors",
  "stream_kwargs": {"force_http": false},
  "deserialization_kwargs": {"verify_hash": true, "num_readers": 8}
}'

vllm serve facebook/opt-125m \
  --load-format=tensorizer \
  --model-loader-extra-config="$MODEL_LOADER_EXTRA_CONFIG"
```

Note in this case, if the directory to the model artifacts at
`s3://my-bucket/vllm/facebook/opt-125m/v1/` doesn't have all necessary model 
artifacts to load, you'll want to pass `facebook/opt-125m` as the model tag like
it was done in the example script above. In this case, vLLM will take care of
resolving the other model artifacts by pulling them from HuggingFace Hub.

!!! note
    Note that to use this feature you will need to install `tensorizer` by running `pip install vllm[tensorizer]`.
