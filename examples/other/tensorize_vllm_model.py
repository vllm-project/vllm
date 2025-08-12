# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import json
import os
import uuid

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerArgs, TensorizerConfig, tensorize_lora_adapter,
    tensorize_vllm_model)
from vllm.utils import FlexibleArgumentParser

# yapf conflicts with isort for this docstring
# yapf: disable
"""
tensorize_vllm_model.py is a script that can be used to serialize and 
deserialize vLLM models. These models can be loaded using tensorizer 
to the GPU extremely quickly over an HTTP/HTTPS endpoint, an S3 endpoint,
or locally. Tensor encryption and decryption is also supported, although 
libsodium must be installed to use it. Install vllm with tensorizer support 
using `pip install vllm[tensorizer]`. To learn more about tensorizer, visit
https://github.com/coreweave/tensorizer

To serialize a model, install vLLM from source, then run something 
like this from the root level of this repository:

python examples/other/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
   
Which downloads the model from HuggingFace, loads it into vLLM, serializes it,
and saves it to your S3 bucket. A local directory can also be used. This
assumes your S3 credentials are specified as environment variables
in the form of `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, and 
`S3_ENDPOINT_URL`. To provide S3 credentials directly, you can provide 
`--s3-access-key-id` and `--s3-secret-access-key`, as well as `--s3-endpoint` 
as CLI args to this script.

You can also encrypt the model weights with a randomly-generated key by 
providing a `--keyfile` argument.

To deserialize a model, you can run something like this from the root 
level of this repository:

python examples/other/tensorize_vllm_model.py \
   --model EleutherAI/gpt-j-6B \
   --dtype float16 \
   deserialize \
   --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors

Which downloads the model tensors from your S3 bucket and deserializes them.

You can also provide a `--keyfile` argument to decrypt the model weights if 
they were serialized with encryption.

To support distributed tensor-parallel models, each model shard will be
serialized to a separate file. The tensorizer_uri is then specified as a string
template with a format specifier such as '%03d' that will be rendered with the
shard's rank. Sharded models serialized with this script will be named as
model-rank-%03d.tensors

For more information on the available arguments for serializing, run 
`python -m examples.other.tensorize_vllm_model serialize --help`.

Or for deserializing:

`python examples/other/tensorize_vllm_model.py deserialize --help`.

Once a model is serialized, tensorizer can be invoked with the `LLM` class 
directly to load models:

    llm = LLM(model="facebook/opt-125m",
              load_format="tensorizer",
              model_loader_extra_config=TensorizerConfig(
                    tensorizer_uri = path_to_tensors,
                    num_readers=3,
                    )
              )
            
A serialized model can be used during model loading for the vLLM OpenAI
inference server. `model_loader_extra_config` is exposed as the CLI arg
`--model-loader-extra-config`, and accepts a JSON string literal of the
TensorizerConfig arguments desired.

In order to see all of the available arguments usable to configure 
loading with tensorizer that are given to `TensorizerConfig`, run:

`python examples/other/tensorize_vllm_model.py deserialize --help`

under the `tensorizer options` section. These can also be used for
deserialization in this example script, although `--tensorizer-uri` and
`--path-to-tensors` are functionally the same in this case.

Tensorizer can also be used to save and load LoRA adapters. A LoRA adapter
can be serialized directly with the path to the LoRA adapter on HF Hub and
a TensorizerConfig object. In this script, passing a HF id to a LoRA adapter
will serialize the LoRA adapter artifacts to `--serialized-directory`.

You can then use the LoRA adapter with `vllm serve`, for instance, by ensuring 
the LoRA artifacts are in your model artifacts directory and specifying 
`--enable-lora`. For instance:

```
vllm serve <model_path> \
    --load-format tensorizer \
    --model-loader-extra-config '{"tensorizer_uri": "<model_path>.tensors"}' \
    --enable-lora
```
"""


def parse_args():
    parser = FlexibleArgumentParser(
        description="An example script that can be used to serialize and "
        "deserialize vLLM models. These models "
        "can be loaded using tensorizer directly to the GPU "
        "extremely quickly. Tensor encryption and decryption is "
        "also supported, although libsodium must be installed to "
        "use it.")
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--lora-path",
        type=str,
        required=False,
        help="Path to a LoRA adapter to "
        "serialize along with model tensors. This can then be deserialized "
        "along with the model by passing a tensorizer_config kwarg to "
        "LoRARequest with type TensorizerConfig. See the docstring for this "
        "for a usage example."

    )

    subparsers = parser.add_subparsers(dest='command')

    serialize_parser = subparsers.add_parser(
        'serialize', help="Serialize a model to `--serialized-directory`")

    serialize_parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        help=(
            "The suffix to append to the serialized model directory, which is "
            "used to construct the location of the serialized model tensors, "
            "e.g. if `--serialized-directory` is `s3://my-bucket/` and "
            "`--suffix` is `v1`, the serialized model tensors will be "
            "saved to "
            "`s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors`. "
            "If none is provided, a random UUID will be used."))
    serialize_parser.add_argument(
        "--serialized-directory",
        type=str,
        required=True,
        help="The directory to serialize the model to. "
        "This can be a local directory or S3 URI. The path to where the "
        "tensors are saved is a combination of the supplied `dir` and model "
        "reference ID. For instance, if `dir` is the serialized directory, "
        "and the model HuggingFace ID is `EleutherAI/gpt-j-6B`, tensors will "
        "be saved to `dir/vllm/EleutherAI/gpt-j-6B/suffix/model.tensors`, "
        "where `suffix` is given by `--suffix` or a random UUID if not "
        "provided.")

    serialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Encrypt the model weights with a randomly-generated binary key,"
              " and save the key at this path"))

    deserialize_parser = subparsers.add_parser(
        'deserialize',
        help=("Deserialize a model from `--path-to-tensors`"
              " to verify it can be loaded and used."))

    deserialize_parser.add_argument(
        "--path-to-tensors",
        type=str,
        required=True,
        help="The local path or S3 URI to the model tensors to deserialize. ")

    deserialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Path to a binary key to use to decrypt the model weights,"
              " if the model was serialized with encryption"))

    TensorizerArgs.add_cli_args(deserialize_parser)

    return parser.parse_args()



def deserialize():
    if args.lora_path:
        tensorizer_config.lora_dir = tensorizer_config.tensorizer_dir
        llm = LLM(model=args.model,
                  load_format="tensorizer",
                  tensor_parallel_size=args.tensor_parallel_size,
                  model_loader_extra_config=tensorizer_config,
                  enable_lora=True,
        )
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
            stop=["[/assistant]"]
        )

        # Truncating this as the extra text isn't necessary
        prompts = [
            "[user] Write a SQL query to answer the question based on ..."
        ]

        # Test LoRA load
        print(
            llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest("sql-lora",
                                     1,
                                     args.lora_path,
                                     tensorizer_config = tensorizer_config)
            )
        )
    else:
        llm = LLM(model=args.model,
                  load_format="tensorizer",
                  tensor_parallel_size=args.tensor_parallel_size,
                  model_loader_extra_config=tensorizer_config
        )
    return llm


if __name__ == '__main__':
    args = parse_args()

    s3_access_key_id = (getattr(args, 's3_access_key_id', None)
                        or os.environ.get("S3_ACCESS_KEY_ID", None))
    s3_secret_access_key = (getattr(args, 's3_secret_access_key', None)
                            or os.environ.get("S3_SECRET_ACCESS_KEY", None))
    s3_endpoint = (getattr(args, 's3_endpoint', None)
                or os.environ.get("S3_ENDPOINT_URL", None))

    credentials = {
        "s3_access_key_id": s3_access_key_id,
        "s3_secret_access_key": s3_secret_access_key,
        "s3_endpoint": s3_endpoint
    }

    model_ref = args.model

    model_name = model_ref.split("/")[1]

    if args.command == "serialize" or args.command == "deserialize":
        keyfile = args.keyfile
    else:
        keyfile = None

    if args.model_loader_extra_config:
        config = json.loads(args.model_loader_extra_config)
        tensorizer_args = \
            TensorizerConfig(**config)._construct_tensorizer_args()
        tensorizer_args.tensorizer_uri = args.path_to_tensors
    else:
        tensorizer_args = None

    if args.command == "serialize":
        eng_args_dict = {f.name: getattr(args, f.name) for f in
                        dataclasses.fields(EngineArgs)}

        engine_args = EngineArgs.from_cli_args(
            argparse.Namespace(**eng_args_dict)
        )

        input_dir = args.serialized_directory.rstrip('/')
        suffix = args.suffix if args.suffix else uuid.uuid4().hex
        base_path = f"{input_dir}/vllm/{model_ref}/{suffix}"
        if engine_args.tensor_parallel_size > 1:
            model_path = f"{base_path}/model-rank-%03d.tensors"
        else:
            model_path = f"{base_path}/model.tensors"

        tensorizer_config = TensorizerConfig(
            tensorizer_uri=model_path,
            encryption_keyfile=keyfile,
            **credentials)

        if args.lora_path:
            tensorizer_config.lora_dir = tensorizer_config.tensorizer_dir
            tensorize_lora_adapter(args.lora_path, tensorizer_config)

        tensorize_vllm_model(engine_args, tensorizer_config)

    elif args.command == "deserialize":
        if not tensorizer_args:
            tensorizer_config = TensorizerConfig(
                tensorizer_uri=args.path_to_tensors,
                encryption_keyfile = keyfile,
                **credentials
            )
        deserialize()
    else:
        raise ValueError("Either serialize or deserialize must be specified.")
