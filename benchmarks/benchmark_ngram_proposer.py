# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from unittest import mock

import numpy as np
from benchmark_utils import TimeCollector
from tabulate import tabulate

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def benchmark_propose(args):
    rows = []
    for max_ngram in args.max_ngram:
        collector = TimeCollector(TimeCollector.US)

        model_config = ModelConfig(
            model="facebook/opt-125m",
            max_model_len=args.num_token + args.num_spec_token,
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            dtype="auto",
            seed=0,
            trust_remote_code=False,
        )
        proposer = NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=args.min_ngram,
                    prompt_lookup_max=max_ngram,
                    num_speculative_tokens=args.num_spec_token,
                    method="ngram",
                ),
            )
        )

        # Warm up
        proposer.propose(np.random.randint(0, 20, (args.num_token,)))

        gc.collect()
        for _ in range(args.num_iteration):
            tokens = np.random.randint(0, 20, (args.num_req, args.num_token))
            with collector:
                for i in range(args.num_req):
                    proposer.propose(tokens[i, :])
        rows.append(
            [args.num_req, args.num_token, args.min_ngram, max_ngram]
            + collector.dump_avg_max()
        )

    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Min Ngram",
                "Max Ngram",
                "Avg (us)",
                "Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def benchmark_batched_propose(args):
    NUM_SPECULATIVE_TOKENS_NGRAM = 10
    PROMPT_LOOKUP_MIN = 5
    PROMPT_LOOKUP_MAX = 15
    MAX_MODEL_LEN = int(1e7)
    DEVICE = current_platform.device_type

    model_config = ModelConfig(model="facebook/opt-125m", runner="generate")

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="ngram",
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS_NGRAM,
        prompt_lookup_max=PROMPT_LOOKUP_MAX,
        prompt_lookup_min=PROMPT_LOOKUP_MIN,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    # monkey patch vllm.v1.worker.gpu_model_runner.get_pp_group
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    with mock.patch(
        "vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=mock_pp_group
    ):
        runner = GPUModelRunner(vllm_config, DEVICE)

        # hack max model len
        runner.max_model_len = MAX_MODEL_LEN
        runner.drafter.max_model_len = MAX_MODEL_LEN

        dummy_input_batch = InputBatch(
            max_num_reqs=args.num_req,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=args.num_req * args.num_token,
            device=DEVICE,
            pin_memory=False,
            vocab_size=256000,
            block_sizes=[16],
        )
        dummy_input_batch._req_ids = list(str(id) for id in range(args.num_req))
        dummy_input_batch.num_tokens_no_spec = [args.num_token] * args.num_req
        dummy_input_batch.token_ids_cpu = np.random.randint(
            0, 20, (args.num_req, args.num_token)
        )

        runner.input_batch = dummy_input_batch

        sampled_token_ids = [[0]] * args.num_req

        print("Starting benchmark")
        # first run is warmup so ignore it
        for _ in range(args.num_iteration):
            start = time.time()
            runner.drafter.propose(
                sampled_token_ids,
                dummy_input_batch.num_tokens_no_spec,
                dummy_input_batch.token_ids_cpu,
            )
            end = time.time()
            print(f"Iteration time (s): {end - start}")


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of N-gram speculative decode drafting"
    )
    parser.add_argument(
        "--batched", action="store_true", help="consider time to prepare batch"
    )
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=100,
        help="Number of iterations to run to stabilize final data readings",
    )
    parser.add_argument(
        "--num-req", type=int, default=128, help="Number of requests in the batch"
    )
    parser.add_argument(
        "--num-token", type=int, default=1500, help="Number of tokens for each request"
    )
    parser.add_argument(
        "--min-ngram",
        type=int,
        default=3,
        help="Minimum n-gram to match",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        nargs="*",
        default=[5, 7, 10, 15, 20],
        help="Maximum n-gram to match",
    )
    parser.add_argument(
        "--num-spec-token",
        type=int,
        default=3,
        help="Number of speculative tokens to generate",
    )
    args = parser.parse_args()

    if not args.batched:
        benchmark_propose(args)
    else:
        benchmark_batched_propose(args)


"""
# Example command lines:
# time python3 benchmarks/benchmark_ngram_proposer.py
# time python3 benchmarks/benchmark_ngram_proposer.py --batched --num-iteration 4 --num-token 1000000 --num-req 128
"""  # noqa: E501
if __name__ == "__main__":
    invoke_main()  # pragma: no cover
