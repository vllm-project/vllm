# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import os
import sys
from argparse import RawTextHelpFormatter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, TypeAlias

import torch
import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.profiler import layerwise_profile
from vllm.utils import FlexibleArgumentParser

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256


@dataclass
class ProfileContext:
    engine_args: EngineArgs
    prompt_len: int
    batch_size: int

    # The profiler can run in 2 modes,
    # 1. Run profiler for user specified num_steps
    num_steps: Optional[int] = None
    # 2. Run profiler until all requests complete
    complete_num_requests_per_step: Optional[int] = None

    save_chrome_traces_folder: Optional[str] = None


def get_dtype(dtype: str):
    if dtype == "torch.float":
        return torch.float
    else:
        return dtype


OutputLen_NumReqs_Map: TypeAlias = Dict[int, int]
def compute_request_output_lengths(batch_size: int, step_requests: List[int]) \
      -> OutputLen_NumReqs_Map:
    """
    Given the number of requests, batch_size, and the number of requests
    that each engine-step should process, step_requests, determine the
    output lengths of the requests such that step_request is honoured.

    Example: 
    if batch size = 128 and step_request = [128, 128, 96, 64, 32, 1]
    then return,
    {2 : 32, 3 : 32, 4 : 32, 5 : 31, 6 : 1}, meaning,
    32 requests should have output length 2,
    32 requests should have output length 3,
    32 requests should have output length 4,
    31 requests should have output length 5,
    1 request should have output length 6.

    Args:
        batch_size (int): Number of requests submitted for profile. This is
            args.batch_size.
        step_requests (List[int]): step_requests[i] is the number of requests
            that the ith engine step should process.

    Returns:
        OutputLen_NumReqs_Map : A dictionary with output-length as keys and the
            number of requests required to have that output-length as values.
    """
    ol_nr: OutputLen_NumReqs_Map = {}

    # Number of request that are assigned an output-length
    num_reqs_assigned: int = 0
    num_steps: int = len(step_requests)

    # sanity check. The first step (prefill-step), must process all requests.
    assert step_requests[0] == batch_size

    # Begin assignments from the last step.
    output_length: int = num_steps
    for num_requests_at_step in reversed(step_requests):
        if num_reqs_assigned == batch_size:
            break

        assert num_reqs_assigned < batch_size

        # Remove the number of requests that have been determined
        # to participate in this step and beyond.
        num_reqs_unassigned_at_step = num_requests_at_step - num_reqs_assigned
        assert num_reqs_unassigned_at_step >= 0

        if num_reqs_unassigned_at_step > 0:
            ol_nr[output_length] = num_reqs_unassigned_at_step
            num_reqs_assigned += num_reqs_unassigned_at_step

        output_length -= 1

    # sanity checks.
    assert sum(ol_nr.values()) == batch_size, \
            ("Number of requests in output-length assignment does not match "
             f"batch-size.\n batch size {batch_size} - "
             f"step requests {step_requests} - assignments {ol_nr}")

    # Check that the output-length is in [1, num-steps]. Output length must be
    # at least 1 as all requests must participate in the prefill-step.
    assert all(ol >= 1 and ol <= num_steps for ol in ol_nr), \
            ("Output lengths of requests should be in range "
             f"[1, num-engine-steps].\n batch size {batch_size} - "
             f"step requests {step_requests} - assignments {ol_nr}")

    return ol_nr


def determine_requests_per_step(context: ProfileContext) -> List[int]:
    """
    Determine number of requests each engine step should process.
    If context.num_steps is set, then all engine steps process the
    same number of requests and the output list is of length
    context.num_steps.

    If context.complete_num_requests_per_step is set, then each decode step
    processes fewer and fewer requests until there are no requests to process.
    In this case, the output list is as big as the number of steps
    required to process all requests.

    Args:
        context: ProfileContext object.

    Returns:
        List[int]: Number of requests to process for all engine-steps. 
         output[i], contains the number of requests that the ith step
         should process.
    """
    if context.num_steps:
        # All requests must run until num_engine_steps. This implies
        # that their output lengths must be equal to num_engine_steps.
        return [context.batch_size] * context.num_steps

    assert context.complete_num_requests_per_step and \
                context.complete_num_requests_per_step > 0, \
        (f"Expected a positive complete_num_requests_per_step argument."
         f"Instead got {context.complete_num_requests_per_step}")

    # We start dropping after the first decode step.
    step_requests = [
        context.batch_size,  # prefill
        context.batch_size,  # decode
    ]

    num_running_requests = context.batch_size
    num_running_requests -= context.complete_num_requests_per_step
    while num_running_requests > 0:
        step_requests.append(num_running_requests)
        num_running_requests -= context.complete_num_requests_per_step

    if step_requests[-1] != 1:
        # have 1 request running at the last step. This is often
        # useful
        step_requests.append(1)

    return step_requests


def run_profile(context: ProfileContext, csv_output: Optional[str],
                json_output: Optional[str]):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    requests_per_step: List[int] = determine_requests_per_step(context)

    ol_nr: OutputLen_NumReqs_Map = compute_request_output_lengths(
        context.batch_size, requests_per_step)

    num_steps_to_profile: int = len(requests_per_step)
    max_output_len: int = max(ol_nr.keys())
    assert max_output_len >= 1

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        # max_tokens is set on a per-request basis.
        max_tokens=None,
        ignore_eos=True)

    # Create LLM
    llm = LLM(**asdict(context.engine_args))
    batch_size = context.batch_size
    prompt_len = context.prompt_len

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
    if batch_size > max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than "
            f"max_num_seqs ({max_num_seqs}) and therefore cannot be run in a "
            f"single profile step, please choose a smaller batch size")
        sys.exit(-1)
    print("llm.llm_engine.model_config.max_model_len: ",
          llm.llm_engine.model_config.max_model_len)
    if prompt_len + max_output_len > llm.llm_engine.model_config.max_model_len:
        print(f"ERROR: chosen prompt_len + max_output_len ({prompt_len} + "
              f"{max_output_len} = {prompt_len + max_output_len}) is larger "
              f"than the model's max_model_len ({max_model_len}), please "
              f"choose a smaller prompt_len or max_output_len, or increase "
              f"--max-model-len")
        sys.exit(-1)

    def add_requests():

        def get_output_len_generator() -> Generator[int, Any, Any]:
            for output_len, num_reqs in ol_nr.items():
                for _ in range(num_reqs):
                    yield output_len

        output_len_generator = get_output_len_generator()
        for i in range(batch_size):
            sampling_params.max_tokens = next(output_len_generator)
            assert isinstance(sampling_params.max_tokens, int)

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
    for _ in tqdm.tqdm(range(num_steps_to_profile - 1)):
        num_running_seqs = llm.llm_engine.scheduler[
            0].get_num_unfinished_seq_groups()
        with layerwise_profile(
                num_running_seqs=num_running_seqs) as decode_prof:
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
        csv_filename_base = csv_output[:-4] \
                if csv_output.endswith('.csv') else csv_output
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

        # Add .json to json_output filename if it doesn't exist already.
        json_output_file = json_output if json_output.endswith(
            '.json') else json_output + '.json'
        with open(json_output_file, "w+") as f:
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
    python examples/offline_inference/profiling.py \\
        --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --batch-size 4 \\
        --prompt-len 512 --max-num-batched-tokens 8196 --json Llama31-8b-FP8 \\
        --enforce-eager run_num_steps -n 2
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

    subparsers = parser.add_subparsers(dest="cmd")

    run_num_steps_parser = subparsers.add_parser(
        "run_num_steps",
        help="This variation profiles n engine.step() invocations.")
    run_num_steps_parser.add_argument(
        '-n',
        '--num-steps',
        type=int,
        help="Number of engine steps to profile.\n"
        "Setting it to 1, profiles only the prefill step.\n"
        "Setting it to 2, profiles the prefill and first decode step\n"
        "Setting it to 3, profiles the prefill, 1st and 2nd decode steps\n"
        "and so on ...")

    run_to_completion_parser = subparsers.add_parser(
        "run_to_completion",
        help="This variation profiles all the engine.step() invocations"
        "until the engine exhausts all submitted requests.")
    run_to_completion_parser.add_argument(
        '-n',
        '--complete-num-requests-per-step',
        type=int,
        help=
        "Complete complete_num_requests_per_step requests every decode step."
        "For e.g., with batch_size 128 and complete_num_requests_per_step 32,"
        "the profiler is run for 6 engine steps, with the steps processing, "
        "128, 128, 96, 64, 32, 1 requests respectively.\n"
        "Note that we tack-on a one-request step at the end as it is often "
        "useful.")

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
