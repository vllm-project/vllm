import argparse
import dataclasses
import os
import time
from typing import List

import numpy as np
import torch_xla.debug.profiler as xp
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    _ = xp.start_server(9012)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    # Warmup
    print("Warming up...")
    warmup_latencies = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        warmup_latencies.append(run_to_completion())
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")

    # Profile
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    # Enable tracing on server
    xp.trace_detached("localhost:9012",
                      profile_dir,
                      delay_ms=DELAY_MS,
                      duration_ms=DURATION_MS)
    if DELAY_MS == 0:
        time.sleep(1.0)
    profile_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        profile_latencies.append(run_to_completion())
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")

    return


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=5,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=1,
                        help='Number of iterations to run for profiling.')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default="profiles",
        help=
        ('path to save the pytorch profiler output. Can be visualized '
         'with ui.perfetto.dev or Tensorboard '
         '(https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm).'
         ))

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
