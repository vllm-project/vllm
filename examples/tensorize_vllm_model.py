import argparse
import os
import tempfile
import time
import uuid
from functools import partial
from typing import Type

import torch
import torch.nn as nn
from tensorizer import (DecryptionParams, EncryptionParams, TensorDeserializer,
                        TensorSerializer, stream_io)
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from vllm.config import _get_and_verify_dtype
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.model_executor.tensorizer_loader import TensorizerArgs


def parse_args():
    parser = argparse.ArgumentParser(
        description="An example script that can be used to serialize and "
        "deserialize vLLM models. These models "
        "can be loaded using tensorizer directly to the GPU "
        "extremely quickly. Tensor encryption and decryption is "
        "also supported, although libsodium must be installed to "
        "use it.")
    parser = TensorizerArgs.add_cli_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model reference name to serialize or deserialize. "
        "This should be a HuggingFace ID for the model, e.g. "
        "EleutherAI/gpt-j-6B.")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        required=False,
                        help="The dtype to cast the tensors to. ")

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

    return parser.parse_args()


def make_model_contiguous(model):
    # Ensure tensors are saved in memory contiguously
    for param in model.parameters():
        param.data = param.data.contiguous()


def _get_vllm_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def serialize():
    model = AutoModelForCausalLM.from_pretrained(model_ref)
    config = AutoConfig.from_pretrained(model_ref)
    make_model_contiguous(model)
    to_dtype = _get_and_verify_dtype(config, dtype=dtype)
    model = model.to(dtype=to_dtype)
    with tempfile.TemporaryDirectory() as tmpdir:
        DOWNLOAD_DIR = os.path.join(tmpdir, model_name)
        model.save_pretrained(DOWNLOAD_DIR)
        del model
        model_class = _get_vllm_model_architecture(config)
        model = model_class(config).to(dtype=to_dtype)
        model.load_weights(model_ref, cache_dir=DOWNLOAD_DIR)

    encryption_params = EncryptionParams.random() if keyfile else None
    if keyfile:
        with _write_stream(keyfile) as stream:
            stream.write(encryption_params.key)

    with _write_stream(model_path) as stream:
        serializer = TensorSerializer(stream, encryption=encryption_params)
        serializer.write_module(model)
        serializer.close()

    print("Serialization complete. Model tensors saved to", model_path)
    if keyfile:
        print("Key saved to", keyfile)


def deserialize():
    config = AutoConfig.from_pretrained(model_ref)

    with no_init_or_tensor():
        model_class = _get_vllm_model_architecture(config)
        model = model_class(config)

    before_mem = get_mem_usage()
    # Lazy load the tensors from S3 into the model.
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

    return model


args = parse_args()

s3_access_key_id = (args.s3_access_key_id or os.environ.get("S3_ACCESS_KEY_ID")
                    or None)
s3_secret_access_key = (args.s3_secret_access_key
                        or os.environ.get("S3_SECRET_ACCESS_KEY") or None)

s3_endpoint = (args.s3_endpoint or os.environ.get("S3_ENDPOINT_URL") or None)

_read_stream, _write_stream = (partial(
    stream_io.open_stream,
    mode=mode,
    s3_access_key_id=s3_access_key_id,
    s3_secret_access_key=s3_secret_access_key,
    s3_endpoint=s3_endpoint,
) for mode in ("rb", "wb+"))

dtype = args.dtype

model_ref = args.model

model_name = model_ref.split("/")[1]

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8080"

torch.distributed.init_process_group(world_size=1, rank=0)
initialize_model_parallel()

keyfile = args.keyfile if args.keyfile else None

if args.command == "serialize":
    input_dir = args.serialized_directory.rstrip('/')
    suffix = args.suffix if args.suffix else uuid.uuid4().hex
    base_path = f"{input_dir}/vllm/{model_ref}/{suffix}"
    model_path = f"{base_path}/model.tensors"
    args.tensorizer_uri = args.serialized_directory
    tensorizer_args = TensorizerArgs.from_cli_args(args)
    serialize()
elif args.command == "deserialize":
    tensorizer_args = TensorizerArgs.from_cli_args(args)
    model_path = args.path_to_tensors
    deserialize()
else:
    raise ValueError("Either serialize or deserialize must be specified.")
