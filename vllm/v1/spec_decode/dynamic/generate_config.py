from vllm.config.speculative import DynamicSpeculativeConfig
from vllm.v1.spec_decode.dynamic.process_benchmark_results import parse_itl
from examples.offline_inference.spec_decode import main as spec_decode_main
from vllm.v1.spec_decode.dynamic.process_benchmark_results import parse_itl
from vllm.v1.spec_decode.dynamic.profiling_client import run_benchmarks
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.benchmarks.datasets import add_dataset_parser, get_samples

def main():
    parser = FlexibleArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument(
        "--num-speculative-tokens-list", nargs="*", type=int, default=[1, 3, 5]
    )
    parser.add_argument(
        "--batch-size-list", nargs="*", type=int, default=[1, 4, 16, 64, 256]
    )
    parser.add_argument(
        "--max-vllm-batch-size",
        type=int,
        help="Max vllm server batch size (max concurrency)",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--result-dir", type=str, default="./log/dynamic_sd")
    parser.add_argument("--extra-log-arg", type=str, default="")


    args = parser.parse_args()

    # Step 1: get acceptance_rate_per_pos
    acceptance_length, acceptance_rate_per_pos = spec_decode_main(args)
    
    # Step 2: generate benchmark data for vanilla and specified method
    for method in ["vanilla", args.method]:
        run_benchmarks(
            dry_run = False,
            model_dir = args.model_dir,
            draft_dir = args.draft_dir,
            method = method,
            num_speculative_tokens_list = args.num_speculative_tokens_list,
            batch_size_list = args.batch_size_list,
            max_vllm_batch_size = args.max_vllm_batch_size,
            tp = args.tp,
            result_dir = args.result_dir,
            extra_log_arg = args.extra_log_arg
        )

    # Step 3: parse batch_stats from benchmark data
    batch_stats = parse_itl(args.result_dir)

    # Step 4: create DynamicSpeculativeConfig
    dynamic_config = DynamicSpeculativeConfig(
        is_online=False,
        max_num_speculative_tokens=len(acceptance_rate_per_pos) + 1,
        acceptance_rate_per_pos=acceptance_rate_per_pos,
        batch_stats=batch_stats,
    )

    # Step 5: save dynamic_config to a json file
    import json
    with open(f"{args.result_dir}/dynamic_speculative_config.json", "w") as f:
        dynamic_config.model_dump_json(f, indent=4)