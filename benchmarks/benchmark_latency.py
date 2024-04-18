"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


def main(args: argparse.Namespace):
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(model=args.model,
              tokenizer=args.tokenizer,
              quantization=args.quantization,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=args.trust_remote_code,
              dtype=args.dtype,
              enforce_eager=args.enforce_eager,
              kv_cache_dtype=args.kv_cache_dtype,
              quantization_param_path=args.quantization_param_path,
              device=args.device,
              ray_workers_use_nsight=args.ray_workers_use_nsight,
              enable_chunked_prefill=args.enable_chunked_prefill,
              download_dir=args.download_dir,
              block_size=args.block_size)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompt_token_ids = dummy_prompt_token_ids.tolist()

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "."
            ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    args = parser.parse_args()
    main(args)
