import argparse
import dataclasses
import json
import os
import time
import uuid
from functools import partial
from typing import Type

import torch
import torch.nn as nn
from tensorizer import DecryptionParams, TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import AutoConfig, PretrainedConfig

from vllm.distributed import initialize_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.model_executor.model_loader.tensorizer import (TensorizerArgs,
                                                         TensorizerConfig,
                                                         serialize_vllm_model)
from vllm.model_executor.models import ModelRegistry

# yapf conflicts with isort for this docstring
# yapf: disable
"""
tensorize_vllm_model.py is a script that can be used to serialize and 
deserialize vLLM models. These models can be loaded using tensorizer 
to the GPU extremely quickly over an HTTP/HTTPS endpoint, an S3 endpoint,
or locally. Tensor encryption and decryption is also supported, although 
libsodium must be installed to use it. Install vllm with tensorizer support 
using `pip install vllm[tensorizer]`.

To serialize a model, install vLLM from source, then run something 
like this from the root level of this repository:

python -m examples.tensorize_vllm_model \
   --model EleutherAI/gpt-j-6B \
   --dtype float16 \
   serialize \
   --serialized-directory s3://my-bucket/ \
   --suffix vllm
   
Which downloads the model from HuggingFace, loads it into vLLM, serializes it,
and saves it to your S3 bucket. A local directory can also be used. This
assumes your S3 credentials are specified as environment variables
in the form of `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, and `S3_ENDPOINT`.
To provide S3 credentials directly, you can provide `--s3-access-key-id` and 
`--s3-secret-access-key`, as well as `--s3-endpoint` as CLI args to this 
script.

You can also encrypt the model weights with a randomly-generated key by 
providing a `--keyfile` argument.

To deserialize a model, you can run something like this from the root 
level of this repository:

python -m examples.tensorize_vllm_model \
   --model EleutherAI/gpt-j-6B \
   --dtype float16 \
   deserialize \
   --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/vllm/model.tensors

Which downloads the model tensors from your S3 bucket and deserializes them.

You can also provide a `--keyfile` argument to decrypt the model weights if 
they were serialized with encryption.

For more information on the available arguments for serializing, run 
`python -m examples.tensorize_vllm_model serialize --help`.

Or for deserializing:

`python -m examples.tensorize_vllm_model deserialize --help`.

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

`python -m examples.tensorize_vllm_model deserialize --help`

under the `tensorizer options` section. These can also be used for
deserialization in this example script, although `--tensorizer-uri` and
`--path-to-tensors` are functionally the same in this case.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="An example script that can be used to serialize and "
        "deserialize vLLM models. These models "
        "can be loaded using tensorizer directly to the GPU "
        "extremely quickly. Tensor encryption and decryption is "
        "also supported, although libsodium must be installed to "
        "use it.")
    parser = EngineArgs.add_cli_args(parser)
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


def _get_vllm_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")



def deserialize():
    config = AutoConfig.from_pretrained(model_ref)

    with no_init_or_tensor():
        model_class = _get_vllm_model_architecture(config)
        model = model_class(config)

    before_mem = get_mem_usage()
    start = time.time()

    if keyfile:
        with _read_stream(keyfile) as stream:
            key = stream.read()
            decryption_params = DecryptionParams.from_key(key)
            tensorizer_args.deserializer_params['encryption'] = \
                decryption_params

    with (_read_stream(model_path)) as stream, TensorDeserializer(
            stream, **tensorizer_args.deserializer_params) as deserializer:
        deserializer.load_into_module(model)
        end = time.time()

    # Brag about how fast we are.
    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    print(
        f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s"
    )
    print(f"Memory usage before: {before_mem}")
    print(f"Memory usage after: {after_mem}")
    del model.vllm_tensorized_marker
    return model


args = parse_args()

s3_access_key_id = (getattr(args, 's3_access_key_id', None)
                    or os.environ.get("S3_ACCESS_KEY_ID", None))
s3_secret_access_key = (getattr(args, 's3_secret_access_key', None)
                        or os.environ.get("S3_SECRET_ACCESS_KEY", None))
s3_endpoint = (getattr(args, 's3_endpoint', None)
               or os.environ.get("S3_ENDPOINT", None))

credentials = {
    "s3_access_key_id": s3_access_key_id,
    "s3_secret_access_key": s3_secret_access_key,
    "s3_endpoint": s3_endpoint
}

_read_stream, _write_stream = (partial(
    stream_io.open_stream,
    mode=mode,
    s3_access_key_id=s3_access_key_id,
    s3_secret_access_key=s3_secret_access_key,
    s3_endpoint=s3_endpoint,
) for mode in ("rb", "wb+"))

model_ref = args.model

model_name = model_ref.split("/")[1]

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8080"

torch.distributed.init_process_group(world_size=1, rank=0)
initialize_model_parallel()

keyfile = args.keyfile if args.keyfile else None

eng_args_dict = {f.name: getattr(args, f.name) for f in
                 dataclasses.fields(EngineArgs)}

engine_args = EngineArgs.from_cli_args(argparse.Namespace(**eng_args_dict))
engine = LLMEngine.from_engine_args(engine_args)

if args.model_loader_extra_config:
    config = json.loads(args.model_loader_extra_config)
    tensorizer_args = TensorizerConfig(**config)._construct_tensorizer_args()
    tensorizer_args.tensorizer_uri = args.path_to_tensors
else:
    tensorizer_args = None

if args.command == "serialize":
    input_dir = args.serialized_directory.rstrip('/')
    suffix = args.suffix if args.suffix else uuid.uuid4().hex
    base_path = f"{input_dir}/vllm/{model_ref}/{suffix}"
    model_path = f"{base_path}/model.tensors"
    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        **credentials)
    serialize_vllm_model(engine, tensorizer_config, keyfile)
elif args.command == "deserialize":
    if not tensorizer_args:
        tensorizer_args = TensorizerConfig(
            tensorizer_uri=args.path_to_tensors,
            encryption_keyfile = keyfile,
            **credentials
        )._construct_tensorizer_args()
        tensorizer_args.tensorizer_uri = args.path_to_tensors
    model_path = args.path_to_tensors
    deserialize()
else:
    raise ValueError("Either serialize or deserialize must be specified.")
