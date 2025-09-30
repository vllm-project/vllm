# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TODO: add benchmark for sampler
import argparse
import json
import os
import time

import numpy as np
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--num_runs", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--all_random", action="store_true", default=False)
    parser.add_argument("--all_greedy", action="store_true", default=False)
    parser.add_argument("--flashinfer", action="store_true", default=False)
    parser.add_argument("--output_json", type=str, default="results.json")
    args = parser.parse_args()

    # whether use flashinfer
    if args.flashinfer:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    else:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    # create Sampler instance
    sampler = Sampler()
    results = {
        "latencies": [],
        "avg_latency": 0,
        "percentiles": {"p50": 0, "p90": 0, "p99": 0},
        "extra_info": {},
    }
    for i in range(100):
        logits = torch.randn(
            args.batch_size, args.vocab_size, device=torch.device("cuda")
        )
        top_p = torch.ones(args.batch_size, device=torch.device("cuda")) * args.top_p
        top_k = (
            torch.ones(args.batch_size, device=torch.device("cuda"), dtype=torch.int)
            * args.top_k
        )
        temperature = (
            torch.ones(args.batch_size, device=torch.device("cuda")) * args.temperature
        )
        all_random = args.all_random
        all_greedy = args.all_greedy
        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            all_random=all_random,
            all_greedy=all_greedy,
            max_num_logprobs=None,
            prompt_token_ids=None,
            allowed_token_ids_mask=None,
            output_token_ids=[],
            presence_penalties=torch.zeros(
                args.batch_size, device=torch.device("cuda")
            ),
            frequency_penalties=torch.zeros(
                args.batch_size, device=torch.device("cuda")
            ),
            repetition_penalties=torch.ones(
                args.batch_size, device=torch.device("cuda")
            ),
            no_penalties=True,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
            generators={},
        )
        sampler(logits, sampling_metadata)

    torch.cuda.synchronize()
    top_p = torch.ones(args.batch_size, device=torch.device("cuda")) * args.top_p
    top_k = (
        torch.ones(args.batch_size, device=torch.device("cuda"), dtype=torch.int)
        * args.top_k
    )
    temperature = (
        torch.ones(args.batch_size, device=torch.device("cuda")) * args.temperature
    )
    for i in range(args.num_runs):
        logits = torch.randn(
            args.batch_size, args.vocab_size, device=torch.device("cuda")
        )
        all_random = args.all_random
        all_greedy = args.all_greedy
        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            top_p=None if args.top_p >= 1 else top_p,
            top_k=top_k,
            all_random=all_random,
            all_greedy=all_greedy,
            max_num_logprobs=None,
            prompt_token_ids=None,
            allowed_token_ids_mask=None,
            output_token_ids=[],
            presence_penalties=torch.zeros(
                args.batch_size, device=torch.device("cuda")
            ),
            frequency_penalties=torch.zeros(
                args.batch_size, device=torch.device("cuda")
            ),
            repetition_penalties=torch.ones(
                args.batch_size, device=torch.device("cuda")
            ),
            no_penalties=True,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
            generators={},
        )
        t1 = time.perf_counter()
        sampler(logits, sampling_metadata)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        log_message = (
            f"batch_size: {args.batch_size}, "
            f"vocab_size: {args.vocab_size}, "
            f"Time: {t2 - t1} seconds"
        )
        print(log_message)
        results["latencies"].append(t2 - t1)
    tmp = vars(args)
    for key, value in tmp.items():
        results["extra_info"][key] = value
    results["avg_latency"] = np.mean(np.array(results["latencies"]))
    results["percentiles"]["p50"] = np.percentile(np.array(results["latencies"]), 50)
    results["percentiles"]["p90"] = np.percentile(np.array(results["latencies"]), 90)
    results["percentiles"]["p99"] = np.percentile(np.array(results["latencies"]), 99)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
