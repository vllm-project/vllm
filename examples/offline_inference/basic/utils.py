# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace

from vllm import SamplingParams
from vllm.utils import FlexibleArgumentParser


def add_sampling_params_args(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    sampling_params = SamplingParams()
    parser.add_argument("--max-tokens",
                        type=int,
                        default=sampling_params.max_tokens)
    parser.add_argument("--temperature",
                        type=float,
                        default=sampling_params.temperature)
    parser.add_argument("--top-p", type=float, default=sampling_params.top_p)
    parser.add_argument("--top-k", type=int, default=sampling_params.top_k)
    return parser


def del_sampling_params_args(args: Namespace) -> Namespace:
    del args.max_tokens
    del args.temperature
    del args.top_p
    del args.top_k
    return args
