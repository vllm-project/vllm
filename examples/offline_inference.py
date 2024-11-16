from dataclasses import asdict

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from transformers import AutoTokenizer

def get_prompts(args):
    # The default sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # if user specifies input-len, generate fake fixed-length prompts
    # The code is copied from benchmark_throughput.py
    if args.input_len is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)

        for i in range(-10, 10):
            prompt = "hi " * (args.input_len + i)
            tokenized_prompt = tokenizer(prompt).input_ids
            if len(tokenized_prompt) == args.input_len:
                break
        else:
            raise ValueError(
                f"Failed to synthesize a prompt with {args.input_len} tokens.")

        prompts = [prompt for _ in range(args.batch_size)]

    if args.batch_size != len(prompts):
        prompts = (prompts * ((args.batch_size // len(prompts)) + 1))[:args.batch_size]

    return prompts

def main(args):
    # Create prompts
    prompts = get_prompts(args)

    # Create a sampling params object.
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len
    )

    # Create an LLM.
    # The default model is 'facebook/opt-125m', ensured by the default parameters of EngineArgs
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**asdict(engine_args))

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4,
                    help="Batch size for inference, default is lenght of sample prompts")
    parser.add_argument("--input-len", type=int, default=None,
                    help="Use fake fixed-length prompt as input if set")
    parser.add_argument("--output-len", type=int, default=16,
                    help="Output length for sampling")
    parser.add_argument('--n', type=int, default=1,
                    help='Number of generated sequences per prompt')
    parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for text generation')
    parser.add_argument('--top-p', type=float, default=0.95,
                    help='top_p for text generation')
    parser.add_argument('--top-k', type=int, default=-1,
                    help='top_k for text generation')

    EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)