# The CLI entrypoint to vLLM.
import argparse

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser


def main():
    parser = argparse.ArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers()

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    make_arg_parser(serve_parser)
    # Override the `--model` optional argument, make it positional.
    serve_parser.add_argument("model-tag", type=str, help="The model tag to serve")
    serve_parser.set_defaults(func=run_server)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
