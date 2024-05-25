# The CLI entrypoint to vLLM.
import argparse

from vllm.entrypoints.openai.api_server import run_server, complete
from vllm.entrypoints.openai.cli_args import make_arg_parser


def main():
    parser = argparse.ArgumentParser(description="vLLM CLI")
    make_arg_parser(parser)
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    # Override the `--model` optional argument, make it positional.
    serve_parser.add_argument(
        "model-tag",
        type=str,
        help="The model tag to serve")
    serve_parser.set_defaults(func=run_server)
    
    complete_parser = subparsers.add_parser(
        "complete",
        help="Generate text completions based on the given prompt",
        usage="vllm complete <complete_prompt> [options]")
    complete_parser.add_argument(
        "complete-prompt",
        type=str,
        help="The prompt to complete")
    complete_parser.set_defaults(func=complete)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
