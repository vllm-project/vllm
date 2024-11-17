from dataclasses import asdict

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


def get_prompts(num_prompts: int):
    # The default sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    if num_prompts != len(prompts):
        prompts = (prompts * ((num_prompts // len(prompts)) + 1))[:num_prompts]

    return prompts


def main(args):
    # Create prompts
    prompts = get_prompts(args.num_prompts)

    # Create a sampling params object.
    sampling_params = SamplingParams(n=args.n,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     max_tokens=args.max_tokens)

    # Create an LLM.
    # The default model is 'facebook/opt-125m'
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**asdict(engine_args))

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    group = parser.add_argument_group("SamplingParams options")
    group.add_argument("--num-prompts",
                       type=int,
                       default=4,
                       help="Number of prompts used for inference")
    group.add_argument("--max-tokens",
                       type=int,
                       default=16,
                       help="Generated output length for sampling")
    group.add_argument('--n',
                       type=int,
                       default=1,
                       help='Number of generated sequences per prompt')
    group.add_argument('--temperature',
                       type=float,
                       default=0.8,
                       help='Temperature for text generation')
    group.add_argument('--top-p',
                       type=float,
                       default=0.95,
                       help='top_p for text generation')
    group.add_argument('--top-k',
                       type=int,
                       default=-1,
                       help='top_k for text generation')

    args = parser.parse_args()
    main(args)
