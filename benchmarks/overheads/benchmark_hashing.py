# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cProfile
import pstats

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

# A very long prompt, total number of tokens is about 15k.
LONG_PROMPT = ["You are an expert in large language models, aren't you?"] * 1000
LONG_PROMPT = " ".join(LONG_PROMPT)


def main(args):
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)
    profiler = cProfile.Profile()

    print("------warm up------")
    for i in range(3):
        output = llm.generate(LONG_PROMPT, sampling_params)
        print(output[0].outputs[0].text)

    print("------start generating------")
    for i in range(3):
        profiler.runctx(
            "llm.generate(LONG_PROMPT, sampling_params)", globals(), locals()
        )

    # analyze the runtime of hashing function
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    total_time = 0
    total_calls = 0
    for func in stats.stats:
        if "hash_of_block" in func[2]:
            total_time = stats.stats[func][3]
            total_calls = stats.stats[func][0]
    percentage = (total_time / stats.total_tt) * 100
    print(
        f"Hashing took {total_time:.2f} seconds,{percentage:.2f}% of the total runtime."
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of hashing function in"
        "automatic prefix caching."
    )
    parser.add_argument("--model", type=str, default="lmsys/longchat-7b-16k")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument(
        "--enable-prefix-caching", action="store_true", help="enable prefix caching"
    )
    args = parser.parse_args()
    main(args)
