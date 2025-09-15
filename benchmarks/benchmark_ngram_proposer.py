# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import numpy as np
from tabulate import tabulate

from benchmark_utils import TimeCollector
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def main(args):
    rows = []
    for max_ngram in args.max_ngram:
        collector = TimeCollector(TimeCollector.US)

        model_config = ModelConfig(
            model="facebook/opt-125m",
            task="generate",
            max_model_len=args.num_token + args.num_spec_token,
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            dtype="auto",
            seed=None,
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


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of N-gram speculative decode drafting"
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
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
