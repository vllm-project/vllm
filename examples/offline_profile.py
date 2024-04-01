import argparse
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.profiler import nm_profile

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
MAX_SEQ_LEN_DEFAULT = 1024


@dataclass
class ProfileContext:
    model: str
    model_revision: str
    sparsity: str
    quantization: str
    max_seq_len: int
    max_num_batched_tokens: int
    prompt_len: int
    batch_size: int
    tensor_parallel_size: int
    allow_cuda_graphs: bool


def run_profile(context: ProfileContext, csv_output: Optional[str],
                json_output: Optional[str]):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8)

    # Create LLM
    llm = LLM(
        model=context.model,
        revision=context.model_revision,
        sparsity=context.sparsity,
        enforce_eager=not context.allow_cuda_graphs,
        tensor_parallel_size=context.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=context.max_seq_len,
        quantization=context.quantization,
        max_num_batched_tokens=context.max_num_batched_tokens,
    )

    batch_size = context.batch_size
    prompt_len = context.prompt_len

    scheduler_config = llm.llm_engine.scheduler_config
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_seqs = scheduler_config.max_num_seqs

    if batch_size * prompt_len > max_num_batched_tokens:
        print(f"ERROR: chosen batch_size * prompt_len "
              f"({batch_size} * {prompt_len} = {batch_size * prompt_len}) is  "
              f"larger than max_num_batched_tokens ({max_num_batched_tokens}) "
              f"and therefore cannot be run in a single profile step, please "
              f"choose a smaller batch size or prompt length, or increase "
              f"--max_num_batched_tokens")
        sys.exit(-1)
    if batch_size >= max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than "
            f"max_num_seqs ({max_num_seqs}) and therefore cannot be run in a "
            f"single profile step, please choose a smaller batch size")
        sys.exit(-1)

    for i in range(batch_size):
        llm.llm_engine.add_request(
            request_id=f"seq{i}",
            prompt=None,
            prompt_token_ids=torch.randint(
                128,  # 128 to skip over special tokens
                llm.llm_engine.model_config.get_vocab_size() // 2,
                size=(prompt_len, )).tolist(),
            sampling_params=sampling_params)

    with nm_profile() as prefill_prof:
        llm.llm_engine.step()  # First step is prefill

    with nm_profile() as decode_prof:
        llm.llm_engine.step()

    prefill_results = prefill_prof.results
    decode_results = decode_prof.results

    print("=" * 80)
    print(f"= Prefill Model Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * 80)
    print()
    prefill_results.print_model_table()
    print()
    print("=" * 80)
    print(f"= Decode Model Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * 80)
    print()
    decode_results.print_model_table()
    print()
    print("=" * 80)
    print(f"= Prefill Summary Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * 80)
    print()
    prefill_results.print_summary_table()
    print()
    print("=" * 80)
    print(f"= Decode Summary Table "
          f"(prompt_len={prompt_len}, batch_size={batch_size})")
    print("=" * 80)
    print()
    decode_results.print_summary_table()

    if csv_output:
        csv_filename_base = csv_output.rstrip(".csv")
        prefill_results.export_model_stats_table_csv(
            csv_filename_base + "_prefill_model_table.csv")
        prefill_results.export_summary_stats_table_csv(
            csv_filename_base + "_prefill_summary_table.csv")
        decode_results.export_model_stats_table_csv(\
            csv_filename_base + "_decode_model_table.csv")
        decode_results.export_summary_stats_table_csv(
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
            "decode": decode_results.convert_stats_to_dict()
        }

        with open(json_output.rstrip(".json") + ".json", "w+") as f:
            json.dump(json_dict, f, indent=2)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='The name or path of a HuggingFace Transformers model.')
    parser.add_argument("--model-revision", type=str, default=None)
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
    parser.add_argument(
        "--sparsity",
        "-s",
        type=str,
        choices=[None, 'sparse_w16a16', 'semi_structured_sparse_w16a16'],
        help="Method used to compress sparse weights. If "
        "None, we first check the `sparsity_config` attribute"
        "in the model config file. If that is None we assume"
        "the model weights are dense")
    parser.add_argument(
        "--quantization",
        "-q",
        type=str,
        choices=['awq', 'gptq', 'squeezellm', 'marlin', None],
        default=None,
        help="The method used to quantize the model weights, "
        "options are \"marlin\", \"awq\", \"gptq\" and \"squeezellm\"")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN_DEFAULT,
        help=f"Maximum length of a sequence (including prompt and output), "
        f"default={MAX_SEQ_LEN_DEFAULT}")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to be processed in a single iteration. "
        " Should be greater than batch-size * prompt-len so the prefill can "
        " run in a single iteration.")
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
    parser.add_argument("--tensor-parallel-size",
                        "-tp",
                        type=int,
                        default=1,
                        help="Number of GPUs to use i.e. tensor parallelism, "
                        "default=1")
    parser.add_argument(
        "--allow-cuda-graphs",
        action='store_true',
        help="Enables cuda graphs to be used, well remove a lot of the module "
        "level info in the profiler results since almost everything runs in "
        "the graph where we do not have access to an informative stack trace")

    args = parser.parse_args()

    context = ProfileContext(
        **{
            k: v
            for k, v in vars(args).items()
            if k in inspect.signature(ProfileContext).parameters
        })
    run_profile(context, csv_output=args.csv, json_output=args.json)
