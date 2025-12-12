# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-research/PowerMoE-3b",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help=("Maximum number of sequences to be processed in a single iteration."),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help=("Maximum number of tokens to be processed in a single iteration."),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=("Number of seconds before unresponsive process is killed."),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=("Fraction of GPU memory vLLM is allowed to allocate (0.0, 1.0]."),
    )
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help=("Enable microbatched execution"),
    )
    parser.add_argument(
        "--compilation-config",
        type=int,
        help=("Compilation optimization (O) mode 0-3."),
    )
    parser.add_argument(
        "--quantization",
        type=str,
    )
    parser.add_argument(
        "--disable-expert-parallel",
        dest="enable_expert_parallel",
        action="store_false",
        help="Disable expert parallel (default: enabled).",
    )
    parser.set_defaults(enable_expert_parallel=True)
    return parser.parse_args()


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    enable_expert_parallel,
    trust_remote_code,
    max_num_seqs,
    max_model_len,
    compilation_config,
    gpu_memory_utilization,
    enable_dbo,
    quantization,
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
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_dbo=enable_dbo,
        quantization=quantization,
        compilation_config=compilation_config,
    )
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
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    if current_platform.is_rocm():
        from multiprocessing import set_start_method

        set_start_method("spawn", force=True)

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enforce_eager,
                args.enable_expert_parallel,
                args.trust_remote_code,
                args.max_num_seqs,
                args.max_model_len,
                args.compilation_config,
                args.gpu_memory_utilization,
                args.enable_dbo,
                args.quantization,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
