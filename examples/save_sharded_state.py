import argparse
import dataclasses
import os
import shutil
from pathlib import Path

from vllm import LLM, EngineArgs
from vllm.model_executor.model_loader.loader import ShardedStateLoader

"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python save_sharded_state.py \
    --model /path/to/load \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/save

Then, the model can be loaded with

llm = LLM(
    model="/path/to/save",
    load_format="sharded_state",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
)
"""

parser = argparse.ArgumentParser()
EngineArgs.add_cli_args(parser)
parser.add_argument("--output",
                    "-o",
                    required=True,
                    type=str,
                    help="path to output checkpoint")
parser.add_argument("--pattern",
                    type=str,
                    default=ShardedStateLoader.DEFAULT_PATTERN,
                    help="string pattern of saved filenames")


def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    # Create LLM instance from arguments
    llm = LLM(**dataclasses.asdict(engine_args))
    # Prepare output directory
    Path(args.output).mkdir(exist_ok=True)
    # Dump worker states to output directory
    model_executor = llm.llm_engine.model_executor
    model_executor._run_workers("save_sharded_state",
                                path=args.output,
                                pattern=args.pattern,
                                max_size=5 * 1024**3)
    # Copy metadata files to output directory
    for file in os.listdir(model_path):
        if not any(
                file.endswith(ext) for ext in (".bin", ".pt", ".safetensors")):
            shutil.copy(f"{model_path}/{file}", args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
