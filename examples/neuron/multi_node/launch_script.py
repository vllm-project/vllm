# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import subprocess
import sys

from vllm.logger import init_logger

logger = init_logger("vllm.neuron.multi-node")

NEURON_RT_ROOT_COMM_ID_PORT = 63423


def error_exit(message: str) -> None:
    logger.error(message)
    sys.exit(1)


def arg_parser():
    parser = argparse.ArgumentParser(description="vLLM multi-node launcher")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model or model path")
    parser.add_argument("--world-size",
                        type=int,
                        required=True,
                        help="World size for distributed inference")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        required=True,
                        help="Maximum number of sequences (or batch size)")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=8192,
                        help="Maximum sequence length")
    parser.add_argument("--max-context-length",
                        type=int,
                        help="Maximum context length")
    parser.add_argument("--compiled-model-path",
                        help="Path to the compiled model. If not present, "
                        "model artifacts will be created in local-models "
                        "folder")
    parser.add_argument("--local-ranks-size",
                        type=int,
                        default=32,
                        help="Local ranks size")
    parser.add_argument("--on-device-sampling-config",
                        type=json.loads,
                        help="On-device sampling configuration")
    parser.add_argument("--quantized",
                        type=bool,
                        default=False,
                        help="Enable quantized mode (default: False)")
    parser.add_argument("--quantized-checkpoints-path",
                        type=str,
                        help="Path to quantized checkpoints "
                        "(required if --quantized is True)")
    parser.add_argument("--port",
                        type=int,
                        default=8080,
                        help="Port for the API server")

    args = parser.parse_args()
    if args.quantized and not args.quantized_checkpoints_path:
        parser.error("--quantized-checkpoints-path is required when "
                     "--quantized is enabled.")
    return args


def make_override_config(args, rank):
    if rank < 0:
        error_exit("rank must be a non-negative integer")
    start_rank_id = rank * args.local_ranks_size
    override_config = {
        "world_size": args.world_size,
        "tp_degree": args.local_ranks_size,
        "local_ranks_size": args.local_ranks_size,
        "start_rank_id": start_rank_id,
    }

    if args.max_context_length:
        override_config["max_context_length"] = args.max_context_length
    if args.on_device_sampling_config:
        override_config[
            "on_device_sampling_config"] = args.on_device_sampling_config
    if args.quantized:
        override_config[
            "quantized_checkpoints_path"] = args.quantized_checkpoints_path
        override_config["quantized"] = args.quantized

    return override_config


def main() -> None:
    args = arg_parser()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    mpi_world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    master_addr = os.environ.get("MASTER_ADDR")
    # TODO: this script can be extended to support TnX
    os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"
    if args.compiled_model_path:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = args.compiled_model_path
    os.environ.update({
        "ENABLE_NEURON_MULTI_NODE": "true",
        "WORLD_SIZE": str(mpi_world_size),
        "NEURON_RT_ROOT_COMM_ID":
        f"{master_addr}:{NEURON_RT_ROOT_COMM_ID_PORT}",
        "NEURON_LOCAL_TP": str(args.local_ranks_size),
        "NEURON_RANK_ID": str(rank)
    })

    override_config = make_override_config(args, rank)
    if rank == 0:
        logger.info("Starting vLLM API server on rank 0...")
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            f"--model={args.model}", f"--port={args.port}", "--device=neuron",
            f"--max-num-seqs={args.max_num_seqs}",
            f"--max-model-len={args.max_model_len}",
            f"--override-neuron-config={json.dumps(override_config)}"
        ]
        logger.debug("Command ran: %s", cmd)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            error_exit(f"Failed to start vLLM API server on rank {rank}")
    else:
        logger.info("Starting worker on rank: %s", rank)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_file_path = os.path.join(current_script_dir, "worker.py")
        cmd = [
            "python", worker_file_path, f"--model={args.model}",
            "--device=neuron", f"--max-num-seqs={args.max_num_seqs}",
            f"--max-model-len={args.max_model_len}",
            f"--override-neuron-config={json.dumps(override_config)}"
        ]
        logger.debug("Command ran: %s", cmd)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            error_exit(f"Failed to start worker on rank {rank}")


if __name__ == "__main__":
    main()
