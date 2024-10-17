import inspect
import json
import os
import sys
from argparse import RawTextHelpFormatter
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.profiler import layerwise_profile
from vllm.utils import FlexibleArgumentParser

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
OUTPUT_LEN_DEFAULT = 2


@dataclass
class ProfileContext:
    engine_args: EngineArgs
    prompt_len: int
    output_len: int
    batch_size: int
    save_chrome_traces_folder: Optional[str]


def get_dtype(dtype: str):
    if dtype == "torch.float":
        return torch.float
    else:
        return dtype


def run_profile(context: ProfileContext, csv_output: Optional[str],
                json_output: Optional[str]):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=args.output_len,
                                     ignore_eos=True)

    # Create LLM
    llm = LLM(**asdict(context.engine_args))
    batch_size = context.batch_size
    prompt_len = context.prompt_len
    output_len = context.output_len

    scheduler_config = llm.llm_engine.scheduler_config
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_seqs = scheduler_config.max_num_seqs

    if batch_size * prompt_len > max_num_batched_tokens:
        print(f"ERROR: chosen batch_size * prompt_len "
              f"({batch_size} * {prompt_len} = {batch_size * prompt_len}) is  "
              f"larger than max_num_batched_tokens ({max_num_batched_tokens}) "
              f"and therefore cannot be run in a single profile step, please "
              f"choose a smaller batch size or prompt length, or increase "
              f"--max-num-batched-tokens")
        sys.exit(-1)
    if batch_size >= max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than "
            f"max_num_seqs ({max_num_seqs}) and therefore cannot be run in a "
            f"single profile step, please choose a smaller batch size")
        sys.exit(-1)
    print("llm.llm_engine.model_config.max_model_len: ",
          llm.llm_engine.model_config.max_model_len)
    if prompt_len + output_len > llm.llm_engine.model_config.max_model_len:
        print(
            f"ERROR: chosen prompt_len + output_len ({prompt_len} + "
            f"{output_len} = {prompt_len + output_len}) is larger than the "
            f"model's max_model_len ({max_model_len}), please choose a smaller "
            f"prompt_len or output_len, or increase --max-model-len")
        sys.exit(-1)

    def add_requests():
        for i in range(batch_size):
            prompt_token_ids = torch.randint(
                llm.llm_engine.model_config.get_vocab_size(),
                size=(prompt_len, )).tolist()

            llm.llm_engine.add_request(
                request_id=f"seq{i}",
                prompt={'prompt_token_ids': prompt_token_ids},
                params=sampling_params)

    def abort_requests():
        for i in range(batch_size):
            llm.llm_engine.abort_request(f"seq{i}")

    # Warm up run
    print("Warm up run ...")
    add_requests()
    llm.llm_engine.step()  # Prefill
    llm.llm_engine.step()  # Decode
    abort_requests()

    print("Profile run ...")
    add_requests()

    with layerwise_profile() as prefill_prof:
        llm.llm_engine.step()  # First step is prefill

    decode_profs = []
    for x in range(args.output_len - 1):
        with layerwise_profile() as decode_prof:
            llm.llm_engine.step()
        decode_profs.append(decode_prof)

    decode_results_list = [prof.results for prof in decode_profs]
    prefill_results = prefill_prof.results
    has_decode = len(decode_results_list) > 0

    LINE_WIDTH = 80
    print("=" * LINE_WIDTH)
    print(f"= Prefill Model Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * LINE_WIDTH)
    print()
    prefill_results.print_model_table()

    if has_decode:
        print()
        print("=" * LINE_WIDTH)
        print(f"= First Decode Step Model Table "
              f"(prompt_len={prompt_len}, batch_size={batch_size})")
        print("=" * LINE_WIDTH)
        print()
        decode_results_list[0].print_model_table()

    print()
    print("=" * LINE_WIDTH)
    print(f"= Prefill Summary Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * LINE_WIDTH)
    print()
    prefill_results.print_summary_table()

    if has_decode:
        print()
        print("=" * LINE_WIDTH)
        print(f"= First Decode Step Summary Table "
              f"(prompt_len={prompt_len}, batch_size={batch_size})")
        print("=" * LINE_WIDTH)
        print()
        decode_results_list[0].print_summary_table()

    if csv_output:
        csv_filename_base = csv_output.rstrip(".csv")
        prefill_results.export_model_stats_table_csv(
            csv_filename_base + "_prefill_model_table.csv")
        prefill_results.export_summary_stats_table_csv(
            csv_filename_base + "_prefill_summary_table.csv")

        if has_decode:
            decode_results_list[0].export_model_stats_table_csv(\
                csv_filename_base + "_decode_model_table.csv")
            decode_results_list[0].export_summary_stats_table_csv(
                csv_filename_base + "_decode_summary_table.csv")

    if json_output:
        cuda_devices = [
            torch.cuda.get_device_properties(dev_idx)
            for dev_idx in range(torch.cuda.device_count())
        ]

        json_dict = {
            "context": {
                "python_version": f"{sys.version}",
                "torch_version": f"{torch.__version__}",
                "torch_cuda_version": f"{torch.version.cuda}",
                "cuda_devices": f"{cuda_devices}",
                **asdict(context)
            },
            "prefill": prefill_results.convert_stats_to_dict(),
        }

        if has_decode:
            for idx, dr in enumerate(decode_results_list):
                json_dict[f"decode_{idx + 1}"] = dr.convert_stats_to_dict()

        for idx, dr in enumerate(decode_results_list[1:]):
            json_dict[f"decode_{idx + 1}"] = dr.convert_stats_to_dict()

        with open(json_output.rstrip(".json") + ".json", "w+") as f:
            json.dump(json_dict, f, indent=2)
        pass

    if context.save_chrome_traces_folder is not None:
        os.makedirs(context.save_chrome_traces_folder, exist_ok=True)
        prefill_prof.profiler.export_chrome_trace(
            context.save_chrome_traces_folder + "/prefill.json")
        for idx, decode_prof in enumerate(decode_profs):
            decode_prof.profiler.export_chrome_trace(
                context.save_chrome_traces_folder + f"/decode_{idx + 1}.json")
        print("Traces saved as prefill.json and decode_1.json, etc."
              f" in folder {context.save_chrome_traces_folder}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="""
Profile a model

    example:
    ```
    python examples/offline_profile.py \\
        --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --batch-size 4 \\
        --prompt-len 512 --max-num-batched-tokens 8196 --json Llama31-8b-FP8 \\
        --enforce-eager
    ```

    then you can use various tools to analyze the json output
    terminal ascii tables:
        ```
        python tools/profiler/print_layerwise_table.py \\
            --json-trace Llama31-8b-FP8.json --phase prefill --table summary
        ```
    or create matplotlib stacked bar charts:
        ```
        python tools/profiler/visualize_layerwise_profile.py \\
            --json-trace Llama31-8b-FP8.json \\
            --output-directory profile_breakdown --plot-metric pct_cuda_time
        ```
""",
                                    formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export the results as multiple csv file. This should be the root "
        "filename, will create <filename>_prefill_model_table.csv, "
        "<filename>_prefill_summary_table.csv, "
        "<filename>_decode_model_table.csv, and "
        "<filename>_decode_summary_table.csv")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export the results as a json file. This should be the filename")
    parser.add_argument("--save-chrome-traces-folder",
                        type=str,
                        help="Save chrome traces for the prefill and decode "
                        "will save traces as prefill.json and decode_1.json, "
                        "etc. inside this folder")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=PROMPT_LEN_DEFAULT,
        help=f"Length of the random prompt to use when profiling, all batched "
        f"requests use the same prompt_len, default={PROMPT_LEN_DEFAULT}")
    parser.add_argument("--batch-size",
                        type=int,
                        default=BATCH_SIZE_DEFAULT,
                        help=f"Number of requests to run as a single batch, "
                        f"default={BATCH_SIZE_DEFAULT}")
    parser.add_argument(
        "--output-len",
        type=int,
        default=OUTPUT_LEN_DEFAULT,
        help="Number of llm steps to run (includes prefill and decode) "
        "- default={OUTPUT_LEN_DEFAULT}")

    EngineArgs.add_cli_args(parser)

    args = parser.parse_args()

    context = ProfileContext(
        engine_args=EngineArgs.from_cli_args(args),
        **{
            k: v
            for k, v in vars(args).items()
            if k in inspect.signature(ProfileContext).parameters
        })
    run_profile(context, csv_output=args.csv, json_output=args.json)
