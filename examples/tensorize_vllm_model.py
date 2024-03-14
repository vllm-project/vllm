import os
import time
from typing import Type
import argparse
import torch
import torch.nn as nn
from tensorizer import TensorDeserializer, TensorSerializer, \
    EncryptionParams, DecryptionParams
from tensorizer import stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoConfig, PretrainedConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.tensorizer_loader import TensorizerArgs
from vllm.config import _get_and_verify_dtype
from functools import partial
import uuid

from vllm.model_executor.parallel_utils.parallel_state import \
    initialize_model_parallel

s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID") or None
s3_secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY") or None
s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or None


_read_stream, _write_stream = (
    partial(
        stream_io.open_stream,
        mode=mode,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_endpoint=s3_endpoint,
    )
    for mode in ("rb", "wb+")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for serializing and deserializing vLLM "
                    "models. These models can be loaded using "
                    "tensorizer directly to the GPU extremely quickly")
    parser = TensorizerArgs.add_cli_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model reference name to serialize or deserialize"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=False,
        help="The dtype to use for the model"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="The directory to serialize or deserialize the model to. "
             "For serialization, model tensors will be saved to this "
             "directory under a directory given by the model ref set "
             "in --model and a unique identifier. For instance, if "
             "serializing EleutherAI/pythia-1.4b, in S3 bucket "
             "BUCKET, tensors and encryption key if applicable will "
             "be saved to "
             "s3://BUCKET/EleutherAI/pythia-1.4b/HASH/model.tensors. "
             "for this reason, --input-dir is recommended to specify "
             "a bucket name if using object storage rather than a "
             "local dir. If deserializing, model.tensors and "
             "model.key will be looked for in --input-dir. In the "
             "previous example, the --input-dir to use would be "
             "s3://BUCKET/EleutherAI/pythia-1.4b/HASH"

    )

    subparsers = parser.add_subparsers(dest='command')

    serialize_parser = subparsers.add_parser(
        'serialize',
        help="Serialize a model to `--input-dir`")

    serialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("File to write 32 bytes of randomly-generated binary data used as"
              " an encryption key"
              )
    )

    deserialize_parser = subparsers.add_parser(
        'deserialize',
        help=(
            "Deserialize a model from `--input-dir`"
            " to verify it can be loaded and used."
        )
    )

    deserialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=True,
        help="Decryption keyfile to use to decrypt the model"
    )

    return parser.parse_args()


def make_model_contiguous(model):
    # Ensure tensors are saved in memory contiguously
    for param in model.parameters():
        param.data = param.data.contiguous()


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
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
    model.save_pretrained(DOWNLOAD_DIR)
    del model

    model_class = _get_model_architecture(config)
    model = model_class(config).to(dtype=to_dtype)
    print(f"Loading from {DOWNLOAD_DIR}")
    model.load_weights(model_ref)

    encryption_params = EncryptionParams.random() if keyfile else None
    if keyfile:
        with _write_stream(keyfile_path) as stream:
            stream.write(encryption_params.key)
    print("Is encryption being used?", encryption_params is not None)

    with _write_stream(model_path) as stream:
        serializer = TensorSerializer(stream, encryption=encryption_params)
        serializer.write_module(model)
        serializer.close()

    print("Serialization complete.")


def deserialize():
    config = AutoConfig.from_pretrained(model_ref)

    with no_init_or_tensor():
        model_class = _get_model_architecture(config)
        model = model_class(config)

    before_mem = get_mem_usage()
    # Lazy load the tensors from S3 into the model.
    start = time.time()

    if keyfile:
        with _read_stream(keyfile_path) as stream:
            key = stream.read()
            decryption_params = DecryptionParams.from_key(key)
            tensorizer_args.deserializer_params['encryption'] = \
                decryption_params

    with (_read_stream(
            model_path,
    )) as stream, TensorDeserializer(
        stream,
        **tensorizer_args.deserializer_params) as deserializer:
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
tensorizer_args = TensorizerArgs.from_cli_args(args)


dtype = args.dtype if args.dtype else "float16"

model_ref = args.model

model_name = model_ref.split("/")[1]
input_dir = args.input_dir.rstrip('/')

DOWNLOAD_DIR = f"/tmp/{model_name}"

os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "8080"

torch.distributed.init_process_group(world_size=1, rank=0)
initialize_model_parallel()

keyfile = args.keyfile if args.keyfile else None

base_path = f"{input_dir}/{model_ref}/{uuid.uuid4().hex}" \
    if args.command == "serialize" else input_dir
model_path = f"{base_path}/model.tensors"
keyfile_path = f"{base_path}/{keyfile}"

if args.command == "serialize":
    serialize()
elif args.command == "deserialize":
    deserialize()
else:
    raise ValueError("Either serialize or deserialize must be specified.")
