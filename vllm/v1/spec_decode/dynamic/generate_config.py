import json
import pprint
import time
from pathlib import Path

from vllm.benchmarks.sweep.param_sweep import ParameterSweep
from vllm.benchmarks.sweep.serve import SweepServeArgs, run_main
from vllm.config.speculative import DynamicSpeculativeConfig
from vllm.v1.spec_decode.offline import main as spec_decode_main
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.benchmarks.datasets import add_dataset_parser


def build_serve_params(method, draft_dir, tp,
                       num_speculative_tokens_list,
                       prompt_lookup_max, prompt_lookup_min):
    """Build serve parameter sweep for vanilla + speculative decode configs.

    Each entry becomes a separate server configuration in the sweep.
    The sweep framework starts/stops the server for each serve config.
    """
    records = []

    # Vanilla config (no speculative decoding)
    records.append({"_benchmark_name": "vanilla"})

    # Speculative decoding configs with varying num_speculative_tokens
    if method == "ngram":
        for k in num_speculative_tokens_list:
            records.append({
                "_benchmark_name": f"ngram-k-{k}",
                "speculative_config": {
                    "method": "ngram",
                    "num_speculative_tokens": k,
                    "prompt_lookup_max": prompt_lookup_max,
                    "prompt_lookup_min": prompt_lookup_min,
                },
            })
    elif method in ("eagle", "eagle3"):
        for k in num_speculative_tokens_list:
            records.append({
                "_benchmark_name": f"{method}-k-{k}",
                "speculative_config": {
                    "method": method,
                    "model": draft_dir,
                    "num_speculative_tokens": k,
                    "draft_tensor_parallel_size": tp,
                },
            })
    elif method == "mtp":
        for k in num_speculative_tokens_list:
            records.append({
                "_benchmark_name": f"mtp-k-{k}",
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": k,
                },
            })

    return ParameterSweep.from_records(records)


def build_bench_params(batch_size_list, num_batches):
    """Build benchmark parameter sweep for different concurrencies.

    Each entry varies max_concurrency (batch size) and num_prompts.
    """
    records = []
    for bs in batch_size_list:
        num_prompts = num_batches * bs
        records.append({
            "_benchmark_name": f"bs-{bs}",
            "max_concurrency": bs,
            "num_prompts": num_prompts,
        })
    return ParameterSweep.from_records(records)


def parse_itl_from_dataframe(result_df):
    """Parse ITL from sweep result DataFrame into batch_stats format.

    Returns:
        batch_stats: dict of {batch_size: {num_drafts: median_itl_ms}}
        where num_drafts=0 corresponds to vanilla (no speculation).
    """
    batch_stats = {}
    for _, row in result_df.iterrows():
        bs = int(row["max_concurrency"])

        # Determine k (num speculative tokens) from speculative_config.
        # For vanilla rows, speculative_config is NaN (missing from serve
        # params); for spec decode rows, it's a dict or JSON string.
        spec_config = row.get("speculative_config")
        if isinstance(spec_config, dict):
            k = int(spec_config["num_speculative_tokens"])
        elif isinstance(spec_config, str):
            k = int(json.loads(spec_config)["num_speculative_tokens"])
        else:
            k = 0  # vanilla (NaN or None)

        if bs not in batch_stats:
            batch_stats[bs] = {}
        batch_stats[bs][k] = row["median_itl_ms"]

    return batch_stats


def run_profiling_sweep(args):
    """Run profiling benchmarks using vllm bench sweep serve.

    This replaces the custom profiling_client/profiling_server by leveraging
    the existing vllm bench sweep serve utility which handles:
    - Server lifecycle management (start, wait-for-ready, stop)
    - Cartesian product of serve_params x bench_params
    - Result saving and aggregation
    """
    # Base serve command (static params shared across all serve configs)
    serve_cmd = [
        "vllm", "serve", args.model_dir,
        "--disable-log-requests",
        "--gpu-memory-utilization", "0.95",
        "--max-num-seqs", str(args.max_vllm_batch_size),
        "--tensor-parallel-size", str(args.tp),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ]

    # Base bench command (static params shared across all bench configs)
    bench_cmd = [
        "vllm", "bench", "serve",
        "--model", args.model_dir,
        "--backend", "openai-chat",
        "--endpoint", "/v1/chat/completions",
        "--dataset-name", args.dataset_name,
        "--dataset-path", args.dataset_path,
        f"--temperature={args.temp}",
        f"--top-p={args.top_p}",
        f"--top-k={args.top_k}",
    ]

    # Build parameter sweeps
    serve_params = build_serve_params(
        method=args.method,
        draft_dir=args.draft_dir,
        tp=args.tp,
        num_speculative_tokens_list=args.num_speculative_tokens_list,
        prompt_lookup_max=args.prompt_lookup_max,
        prompt_lookup_min=args.prompt_lookup_min,
    )

    bench_params = build_bench_params(
        batch_size_list=args.batch_size_list,
        num_batches=args.num_batches,
    )

    # Run the sweep
    sweep_args = SweepServeArgs(
        serve_cmd=serve_cmd,
        bench_cmd=bench_cmd,
        after_bench_cmd=[],
        show_stdout=True,
        serve_params=serve_params,
        bench_params=bench_params,
        output_dir=Path(args.result_dir),
        num_runs=1,
        dry_run=False,
        resume=None,
        link_vars=[],
    )

    result_df = run_main(sweep_args)
    return result_df


def main():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="eagle",
                        choices=["ngram", "eagle", "eagle3", "mtp"])
    parser.add_argument(
        "--num-speculative-tokens-list", nargs="*", type=int,
        default=[1, 3, 5]
    )
    parser.add_argument(
        "--batch-size-list", nargs="*", type=int,
        default=[1, 4, 16, 64, 256]
    )
    parser.add_argument(
        "--max-vllm-batch-size",
        type=int,
        help="Max vllm server batch size (max concurrency)",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--result-dir", type=str, default="./log/dynamic_sd")
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of batches to run for each benchmark.")
    parser.add_argument("--custom-mm-prompts", action="store_true")

    args = parser.parse_args()
    args.enable_chunked_prefill = True
    args.enforce_eager = False
    args.print_output = False
    args.num_spec_tokens = max(args.num_speculative_tokens_list)
    args.eagle_dir = args.draft_dir
    args.result_dir = (f"{args.result_dir}/tp-{args.tp}_temp-{args.temp}"
                       f"_top_p-{args.top_p}_top_k-{args.top_k}"
                       f"/{args.dataset_path}/")

    assert args.max_vllm_batch_size == max(args.batch_size_list), (
        "max_vllm_batch_size must be equal to max of batch_size_list"
    )

    pprint.pprint(vars(args))
    start = time.time()

    # Step 1: get acceptance_rate_per_pos
    acceptance_length, acceptance_rate_per_pos = spec_decode_main(args)
    print(f"Acceptance length: {acceptance_length}")
    print(f"Acceptance rate per position: {acceptance_rate_per_pos}")
    print("✅ Step 1: obtained acceptance rate per position.")

    # Step 2: generate benchmark data using vllm bench sweep serve
    # This runs the Cartesian product of:
    #   serve_params: [vanilla, {method}-k-1, {method}-k-3, {method}-k-5]
    #   bench_params: [bs-1, bs-4, bs-16, bs-64, bs-256]
    # The sweep framework handles server start/stop for each serve config.
    result_df = run_profiling_sweep(args)
    print(f"✅ Step 2: benchmark data generated for vanilla and {args.method}.")

    # Step 3: parse batch_stats from benchmark data
    batch_stats = parse_itl_from_dataframe(result_df)
    print("✅ Step 3: parsed batch statistics from benchmark data.")
    print(json.dumps(batch_stats, indent=4))

    # Step 4: Save DynamicSpeculativeConfig to a json file
    dynamic_config = DynamicSpeculativeConfig(
        is_online=False,
        max_num_speculative_tokens=len(acceptance_rate_per_pos),
        acceptance_rate_per_pos=acceptance_rate_per_pos,
        batch_stats=batch_stats,
    )

    config_path = f"{args.result_dir}/dynamic_speculative_config.json"
    with open(config_path, "w") as f:
        json.dump(dynamic_config.model_dump(), f, indent=4)

    print(f"✅ Step 4: config saved to {config_path}")

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")


"""
time python3 vllm/v1/spec_decode/dynamic/generate_config.py \
    --method eagle \
    --model-dir 'meta-llama/Llama-3.1-8B-Instruct' \
    --draft-dir 'yuhuili/EAGLE-LLaMA3.1-Instruct-8B' \
    --tp 1 \
    --temp 0 \
    --top-p 1.0 \
    --top-k -1 \
    --max-vllm-batch-size 256 \
    --batch-size-list 1 4 16 64 256 \
    --num-speculative-tokens-list 1 3 5 \
    --num-batches 20 \
    --dataset-name hf \
    --dataset-path 'philschmid/mt-bench' \
    --no-oversample \
    --result-dir './log/dynamic_sd_test'

# shorter version:
time python3 vllm/v1/spec_decode/dynamic/generate_config.py \
    --method eagle \
    --model-dir 'meta-llama/Llama-3.1-8B-Instruct' \
    --draft-dir 'yuhuili/EAGLE-LLaMA3.1-Instruct-8B' \
    --tp 1 \
    --temp 0 \
    --top-p 1.0 \
    --top-k -1 \
    --max-vllm-batch-size 256 \
    --batch-size-list 1 256 \
    --num-speculative-tokens-list 1 5 \
    --num-batches 5 \
    --dataset-name hf \
    --dataset-path 'philschmid/mt-bench' \
    --no-oversample \
    --result-dir './log/dynamic_sd_test_short'
"""
if __name__ == "__main__":
    main()