# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import dataclasses

# from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    # tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
    # inputs = tokenizer('Hello, world!', return_tensors='pt').input_ids
    inputs = [
        "Hello, my name is",
        "The president of the United States is",
        ("1 + " * 50) + " 1 = ",  # Longer prompt.
        "The capital of France is",
    ]
    # Prompt 0: 'Hello, my name is',
    # Generated text: ' John and I am a 30-year-old man from the United States. I am a software engineer by profession and I have been working in the tech industry for about 5 years now. I am married to a wonderful woman named Sarah, and we have two beautiful children together. We live in a cozy little house in the suburbs, and we love spending time outdoors and exploring new places.\n\nI am a bit of a introvert and I enjoy spending time alone, reading books, watching movies, and playing video games. I am also a bit of a foodie and I love trying out new recipes and experimenting with different cuisines. I'   # noqa: E501
    # Prompt 1: 'The president of the United States is',
    # Generated text: ' the head of state and head of government of the United States. The president directs the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces.\nThe president is elected by the people through the Electoral College to a four-year term, and is one of only two nationally elected federal officers, the other being the Vice President of the United States. The Twenty-second Amendment to the United States Constitution prohibits anyone from being elected to the presidency more than twice.\nThe president is both the head of state and head of government of the United States, and is the leader of the executive branch of the federal government. The president'   # noqa: E501
    # Prompt 2: '1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 +  1 = ',   # noqa: E501
    # Generated text: "50\nThe answer is 50.<|start_header_id|>assistant<|end_header_id|>\n\nThat's correct!\n\nYou added 50 ones together, and the result is indeed 50. Well done!\n\nWould you like to try another math problem?<|start_header_id|>assistant<|end_header_id|>\n\nI can generate a new problem for you. Here it is:\n\n2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 = ?\n\nCan you add up all the"   # noqa: E501
    # Prompt 3: 'The capital of France is',
    # Generated text: " a city of love, art, fashion, and cuisine. Paris, the City of Light, is a must-visit destination for anyone who appreciates beauty, history, and culture. From the iconic Eiffel Tower to the world-famous Louvre Museum, there's no shortage of things to see and do in this incredible city.\nHere are some of the top attractions and experiences to add to your Parisian itinerary:\n1. The Eiffel Tower: This iconic iron lattice tower is a symbol of Paris and one of the most recognizable landmarks in the world. Take the elevator to the top for breathtaking views of the city.\n2"   # noqa: E501

    outputs = llm.generate(inputs, sampling_params)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i}: {prompt!r}, Generated text: {generated_text!r}")
    # print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Number of iterations to run."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=(
            "path to save the pytorch profiler output. Can be visualized "
            "with ui.perfetto.dev or Tensorboard."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
