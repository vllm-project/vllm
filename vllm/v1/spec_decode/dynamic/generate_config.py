import json
import time
    
from vllm.config.speculative import DynamicSpeculativeConfig
from vllm.v1.spec_decode.dynamic.process_benchmark_results import parse_itl
from vllm.v1.spec_decode.offline import main as spec_decode_main
from vllm.v1.spec_decode.dynamic.process_benchmark_results import parse_itl
from vllm.v1.spec_decode.dynamic.profiling_client import run_benchmarks
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.benchmarks.datasets import add_dataset_parser

def main():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="eagle", choices=["ngram", "eagle", "eagle3", "mtp"])
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
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=20, help="Number of batches to run for each benchmark.")
    parser.add_argument("--custom-mm-prompts", action="store_true")
    

    args = parser.parse_args()
    args.enable_chunked_prefill = True
    args.enforce_eager = False
    args.print_output = False
    args.num_spec_tokens = max(args.num_speculative_tokens_list)
    args.eagle_dir = args.draft_dir
    args.result_dir = f"{args.result_dir}/tp-{args.tp}_temp-{args.temp}_top_p-{args.top_p}_top_k-{args.top_k}/{args.dataset_path}/"
    
    # print the args in pretty format
    import pprint
    pprint.pprint(vars(args))
    start = time.time()
    

    # Step 1: get acceptance_rate_per_pos
    acceptance_length, acceptance_rate_per_pos = spec_decode_main(args)
    print(f"Acceptance length: {acceptance_length}")
    print(f"Acceptance rate per position: {acceptance_rate_per_pos}")
    print(f"✅ Step 1: obtained acceptance rate per position.")

    # Step 2: generate benchmark data for vanilla and specified method
    for method in ["vanilla", args.method]:
        run_benchmarks(
            dry_run = False,
            model_dir = args.model_dir,
            draft_dir = args.draft_dir,
            method = method,
            prompt_lookup_max = args.prompt_lookup_max,
            prompt_lookup_min = args.prompt_lookup_min,
            num_speculative_tokens_list = args.num_speculative_tokens_list,
            batch_size_list = args.batch_size_list,
            max_vllm_batch_size = args.max_vllm_batch_size,
            tp = args.tp,
            temp = args.temp,
            top_p = args.top_p,
            top_k = args.top_k,
            num_batches = args.num_batches,
            dataset_name = args.dataset_name,
            dataset_path = args.dataset_path,
            result_dir = args.result_dir,
            extra_log_arg = args.extra_log_arg
        )
    print(f"✅ Step 2: benchmark data generated for vanilla and {args.method}.")

    # Step 3: parse batch_stats from benchmark data
    batch_stats = parse_itl(method=args.method, benchmark_path_parent=args.result_dir)
    print(f"✅ Step 3: parsed batch statistics from benchmark data.")
    
    # Step 4: Save DynamicSpeculativeConfig to a json file
    dynamic_config = DynamicSpeculativeConfig(
        is_online=False,
        max_num_speculative_tokens=len(acceptance_rate_per_pos),
        acceptance_rate_per_pos=acceptance_rate_per_pos,
        batch_stats=batch_stats,
    )

    with open(f"{args.result_dir}/dynamic_speculative_config.json", "w") as f:
        json.dump(dynamic_config.model_dump(), f, indent=4)
    
    print(f"✅ Step 4: config saved to {args.result_dir}/dynamic_speculative_config.json")

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
    --num-batches 20 \
    --dataset-name hf \
    --dataset-path 'philschmid/mt-bench' \
    --no-oversample \
    --result-dir './log/dynamic_sd_test'
"""
if __name__ == "__main__":
    main()