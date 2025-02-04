# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from utils import add_sampling_params_args, del_sampling_params_args

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    args = del_sampling_params_args(args)

    # Create an LLM.
    llm = LLM(**vars(args))
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser = add_sampling_params_args(parser)
    # Set example specific arguments
    parser.set_defaults(model="facebook/opt-125m")
    args = parser.parse_args()
    main(args)
