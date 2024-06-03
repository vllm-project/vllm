"""
Benchmark offline inference throughput.

NOTE: This script is a modified version of benchmarks/benchmark_serving.py from
 the upstream vllm repo at commit a4211a4dc.
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import AutoTokenizer

from .common import (generate_synthetic_requests, num_available_gpus,
                     print_request_outputs, warmup_vllm_engine)
from .datasets_registry import DatasetArgs, get_dataset
# yapf: disable
from .logging.benchmark_result import BenchmarkResult
from .logging.benchmark_result import (
    BenchmarkThroughputResultMetricTemplates as ResultMetricTemplates)

# yapf: enable


def get_tensor_parallel_size(args: argparse.Namespace) -> int:
    tensor_parallel_size = num_available_gpus() \
        if args.use_all_available_gpus_ else args.tensor_parallel_size_
    assert tensor_parallel_size > 0 and \
           tensor_parallel_size <= num_available_gpus()
    return tensor_parallel_size


def run_vllm(requests: List[Tuple[str, int, int]],
             model: str,
             tokenizer: str,
             quantization: Optional[str],
             tensor_parallel_size: int,
             seed: int,
             n: int,
             use_beam_search: bool,
             trust_remote_code: bool,
             dtype: str,
             max_model_len: Optional[int],
             enforce_eager: bool,
             sparsity: Optional[str],
             num_warmup_prompts: int,
             log_model_io: bool = False) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    warmup_vllm_engine(engine=llm, model=model, num_prompts=num_warmup_prompts)

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            # TODO (varun) Make temperature configurable
            #temperature=0.0 if use_beam_search else 1.0,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.
    outputs = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()

    if log_model_io:
        print_request_outputs(outputs)

    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    if args.dataset:
        # Get dataset from registry.
        requests = get_dataset(name=args.dataset,
                               tokenizer=tokenizer,
                               dataset_args=DatasetArgs(
                                   num_samples=args.num_prompts,
                                   max_len=2048,
                                   seed=42,
                                   fixed_output_len=args.output_len))
    else:
        # Make a synthetic dataset.
        requests = generate_synthetic_requests(args.input_len, args.output_len,
                                               args.num_prompts, tokenizer)

    elapsed_time = run_vllm(requests,
                            args.model,
                            args.tokenizer,
                            args.quantization,
                            get_tensor_parallel_size(args),
                            args.seed,
                            args.n,
                            args.use_beam_search,
                            args.trust_remote_code,
                            args.dtype,
                            args.max_model_len,
                            args.enforce_eager,
                            sparsity=args.sparsity,
                            num_warmup_prompts=args.num_warmup_prompts,
                            log_model_io=args.log_model_io)

    total_prompt_tokens = sum(prompt_len for _, prompt_len, _ in requests)
    total_output_tokens = sum(output_len for _, _, output_len in requests)
    total_num_tokens = total_prompt_tokens + total_output_tokens

    request_throughput = len(requests) / elapsed_time
    token_throughput = total_num_tokens / elapsed_time
    print(f"total prompt tokens {total_prompt_tokens}")
    print(f"total output tokens {total_output_tokens}")
    print(f"total num tokens {total_num_tokens}")
    print(f"Throughput: {request_throughput:.2f} requests/s, "
          f"{token_throughput:.2f} tokens/s")

    # Save config and results to json
    save_result = args.save_directory is not None
    if save_result:

        # Setup
        current_dt = datetime.now()

        result = BenchmarkResult(
            description=args.description,
            date=current_dt,
            script_name=Path(__file__).name,
            script_args=vars(args),
            tensor_parallel_size=get_tensor_parallel_size(args),
            model=args.model,
            tokenizer=args.tokenizer,
            dataset=args.dataset)
        result.add_metric(ResultMetricTemplates.request_throughput,
                          request_throughput)
        result.add_metric(ResultMetricTemplates.token_throughput,
                          token_throughput)

        model_id = args.model.replace('/', '_')
        # Save to file
        current_dt_str = current_dt.strftime("%Y%m%d-%H%M%S")
        file_name = Path(
            args.save_directory
        ) / f"benchmark_throughput-{args.backend}-{model_id}-{current_dt_str}.json"  # noqa: E501
        result.store(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--description",
        type=str,
        default="benchmark-throughput",
        help="Benchmark description. This is primarily useful when "
        "we log the benchmark results and process them for plotting charts")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm"],
                        default="vllm")
    parser.add_argument("--sparsity", type=str, default=None)
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--num-warmup-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to do warmups with.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--log-model-io", action="store_true")
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument("--save-directory",
                        type=str,
                        default=None,
                        help="Output directory to store result file")

    tp_group = parser.add_mutually_exclusive_group(required=True)
    tp_group.add_argument("--tensor-parallel-size_", type=int, default=None)
    tp_group.add_argument("--use-all-available-gpus_", action="store_true")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    main(args)
