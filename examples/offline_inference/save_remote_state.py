# SPDX-License-Identifier: Apache-2.0
"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python save_remote_state.py \
    --model /path/to/load \
    --tensor-parallel-size 8 \
    --remote-model-save-url [protocol]://[host]:[port]/[model_name] \

Then, the model can be loaded with

llm = LLM(
    model="/path/to/save",
    --remote-model-url [protocol]://[host]:[port]/[model_name] \
    tensor_parallel_size=8,
)
"""
import dataclasses
from pathlib import Path

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)

parser.add_argument("--remote-model-save-url",
                    required=True,
                    type=str,
                    help="remote address to store model weights")


def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.enable_lora:
        raise ValueError("Saving with enable_lora=True is not supported!")
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    # Create LLM instance from arguments
    llm = LLM(**dataclasses.asdict(engine_args))
    # Dump worker states to output directory
    model_executor = llm.llm_engine.model_executor
    model_executor.save_remote_state(url=args.remote_model_save_url)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
