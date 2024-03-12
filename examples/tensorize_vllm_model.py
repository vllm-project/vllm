import os
import time
from typing import Type
import argparse
import torch
import torch.nn as nn
from tensorizer import TensorDeserializer, TensorSerializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import AutoModelForCausalLM, AutoConfig, PretrainedConfig
from vllm.model_executor.models  import ModelRegistry
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.tensorizer_loader import TensorizerArgs

from vllm.model_executor.parallel_utils.parallel_state import \
    initialize_model_parallel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for serializing and deserializing models. "
                    "in this example script.")
    ## TODO: Decide if below two are worth including besides --model and --tensorizer-uri
    parser = AsyncEngineArgs.add_cli_args(parser)
    parser = TensorizerArgs.add_cli_args(parser)
    parser.add_argument("--serialize",
                        default = False,
                        action="store_true",
                        help="If specified, serialize the model. "
                             "Which will be saved to --tensorizer-uri.")
    parser.add_argument("--deserialize",
                        default = False,
                        action="store_true",
                        help="If specified, deserialize the model. "
                             "Which will be loaded from --tensorizer-uri")
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

s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID") or None
s3_secret_access_key = os.environ.get(
    "S3_SECRET_ACCESS_KEY") or None

args = parse_args()
tensorizer_args = TensorizerArgs.from_cli_args(args)

MODEL_REF = args.model

MODEL_NAME = MODEL_REF.split("/")[1]
S3_URI = args.tensorizer_uri

DOWNLOAD_DIR = f"/tmp/{MODEL_NAME}"

os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "8080"

torch.distributed.init_process_group(world_size=1, rank=0)
initialize_model_parallel()

def serialize():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REF
    )

    make_model_contiguous(model)
    model.save_pretrained(DOWNLOAD_DIR)
    config = AutoConfig.from_pretrained(MODEL_REF)
    model_class = _get_model_architecture(config)
    model = model_class(config)
    print(f"Loading from {DOWNLOAD_DIR}")
    model.load_weights(
        MODEL_REF,
        DOWNLOAD_DIR
    )

    stream = stream_io.open_stream(S3_URI,
                                   "wb",
                                   s3_access_key_id=s3_access_key_id,
                                   s3_secret_access_key=s3_secret_access_key)
    serializer = TensorSerializer(stream)

    print(
        f"Writing serialized tensors for model {MODEL_REF} to {S3_URI}. "
        "Type given as {next(model.parameters()).dtype}")

    serializer.write_module(model)
    serializer.close()
    print("Serialization complete.")


def deserialize():
    config = AutoConfig.from_pretrained(MODEL_REF)

    with no_init_or_tensor():
        model_class = _get_model_architecture(config)
        model = model_class(config)

    before_mem = get_mem_usage()
    # Lazy load the tensors from S3 into the model.
    start = time.time()
    stream = stream_io.open_stream(S3_URI,
                                   "rb",
                                   s3_access_key_id=s3_access_key_id,
                                   s3_secret_access_key=s3_secret_access_key
    )
    deserializer = TensorDeserializer(stream,
                                      **tensorizer_args.deserializer_params
                                      )
    deserializer.load_into_module(model)
    end = time.time()

    # Brag about how fast we are.
    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    deserializer.close()
    print(f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s")
    print(f"Memory usage before: {before_mem}")
    print(f"Memory usage after: {after_mem}")

    return model

if args.serialize:
    serialize()
if args.deserialize:
    deserialize()

