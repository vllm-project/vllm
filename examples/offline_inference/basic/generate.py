# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.pop("max_tokens"),
        temperature=args.pop("temperature"),
        top_p=args.pop("top_p"),
        top_k=args.pop("top_k"),
    )

    # Create an LLM.
    llm = LLM(**args)
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
    # Add engine args
    ea_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(ea_group)
    ea_group.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    sp = SamplingParams()
    sp_group = parser.add_argument_group("Sampling parameters")
    sp_group.add_argument("--max-tokens", type=int, default=sp.max_tokens)
    sp_group.add_argument("--temperature", type=float, default=sp.temperature)
    sp_group.add_argument("--top-p", type=float, default=sp.top_p)
    sp_group.add_argument("--top-k", type=int, default=sp.top_k)
    args: dict = vars(parser.parse_args())
    main(args)
