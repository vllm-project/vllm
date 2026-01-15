# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            -dp=2 \
            -tp=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    -dp=2 \
                    -tp=2 \
                    --dp-num-nodes=2 \
                    --dp-node-rank=0 \
                    --dp-master-addr=10.99.48.128 \
                    --dp-master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    -dp=2 \
                    -tp=2 \
                    --dp-num-nodes=2 \
                    --dp-node-rank=1 \
                    --dp-master-addr=10.99.48.128 \
                    --dp-master-port=13345
"""

import os
from time import sleep

from vllm import LLM, EngineArgs, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port


def create_parser():
    parser = FlexibleArgumentParser(description="Data Parallel Inference")

    # Add all engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="ibm-research/PowerMoE-3b",
        enable_expert_parallel=True,
    )

    # Add DP-specific args (separate from engine args to avoid conflicts)
    parser.add_argument(
        "--dp-num-nodes",
        type=int,
        default=1,
        help="Total number of nodes for data parallel.",
    )
    parser.add_argument(
        "--dp-node-rank",
        type=int,
        default=0,
        help="Rank of the current node for data parallel.",
    )
    parser.add_argument(
        "--dp-master-addr",
        type=str,
        default="",
        help="Master node IP address for DP coordination.",
    )
    parser.add_argument(
        "--dp-master-port",
        type=int,
        default=0,
        help="Master node port for DP coordination.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Number of seconds before unresponsive process is killed.",
    )

    return parser


def main(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    engine_args,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 100

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=[16, 20][global_dp_rank % 2]
    )

    # Create an LLM.
    llm = LLM(**engine_args)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())

    # Extract DP-specific args (pop to remove from engine_args)
    dp_size = args.pop("data_parallel_size")
    dp_num_nodes = args.pop("dp_num_nodes")
    dp_node_rank = args.pop("dp_node_rank")
    dp_master_addr = args.pop("dp_master_addr")
    dp_master_port = args.pop("dp_master_port")
    timeout = args.pop("timeout")

    # Remaining args are engine args
    engine_args = args

    if dp_num_nodes == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port_val = get_open_port()
    else:
        dp_master_ip = dp_master_addr
        dp_master_port_val = dp_master_port

    assert dp_size % dp_num_nodes == 0, "dp_size should be divisible by dp_num_nodes"
    dp_per_node = dp_size // dp_num_nodes

    from multiprocessing import Process

    if current_platform.is_rocm():
        from multiprocessing import set_start_method

        set_start_method("spawn", force=True)

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port_val,
                engine_args,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
