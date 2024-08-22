# The CLI entrypoint to vLLM.
import argparse
import asyncio
import os
import signal
import sys
from typing import Dict, List, Optional, Union

import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def _merge_args_and_config(args: argparse.Namespace):
    """
        merge args from cli and config file supplied in the cli. 
        If an argument is present in cli and args then choose args value
        over config file.

        example:
        # config.yaml
        port: 1231

        # invoke server
        $ vllm serve --config ../config.yaml --port 3122

        # selected port = 3122
    """
    assert args.config, 'No config file specified.'

    # only expecting a flat dictionary of atomic types
    config: Dict[str, Union[int, str]] = {}

    try:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
    except Exception as ex:
        logger.error("Unable to read the config file at %s", args.config)
        logger.error(ex)

    for key, value in config.items():
        if hasattr(args, key):
            logger.info("Argument %s is specified via config and commandline.",
                        key)
            logger.info("Selecting the %s=%s from commandline.", key,
                        getattr(args, key))
            continue

        setattr(args, key, value)


def serve(args: argparse.Namespace) -> None:

    if args.config:
        _merge_args_and_config(args)

    # The default value of `--model`
    if args.model != EngineArgs.model:
        raise ValueError(
            "With `vllm serve`, you should provide the model as a "
            "positional argument instead of via the `--model` option.")

    # EngineArgs expects the model name to be passed as --model.
    args.model = args.model_tag

    asyncio.run(run_server(args))


def interactive_cli(args: argparse.Namespace) -> None:
    register_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    if args.command == "complete":
        complete(model_name, openai_client)
    elif args.command == "chat":
        chat(args.system_prompt, model_name, openai_client)


def complete(model_name: str, client: OpenAI) -> None:
    print("Please enter prompt to complete:")
    while True:
        input_prompt = input("> ")

        completion = client.completions.create(model=model_name,
                                               prompt=input_prompt)
        output = completion.choices[0].text
        print(output)


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation: List[ChatCompletionMessageParam] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        input_message = input("> ")
        conversation.append({"role": "user", "content": input_message})

        chat_completion = client.chat.completions.create(model=model_name,
                                                         messages=conversation)

        response_message = chat_completion.choices[0].message
        output = response_message.content

        conversation.append(response_message)  # type: ignore
        print(output)


def _add_query_options(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-Compatible RESTful API server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("The model name used in prompt completion, default to "
              "the first model in list models API call."))
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
        ))
    return parser


def main():
    parser = FlexibleArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    serve_parser.add_argument("model_tag",
                              type=str,
                              help="The model tag to serve")
    serve_parser.add_argument(
        "--config",
        type=str,
        required=False,
        default='',
        help="Read CLI options from a config file."
        "Must be a YAML with the following options:"
        "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server"
    )
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.set_defaults(dispatch_function=serve)

    complete_parser = subparsers.add_parser(
        "complete",
        help=("Generate text completions based on the given prompt "
              "via the running API server"),
        usage="vllm complete [options]")
    _add_query_options(complete_parser)
    complete_parser.set_defaults(dispatch_function=interactive_cli,
                                 command="complete")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="vllm chat [options]")
    _add_query_options(chat_parser)
    chat_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("The system prompt to be added to the chat template, "
              "used for models that support system prompts."))
    chat_parser.set_defaults(dispatch_function=interactive_cli, command="chat")

    args = parser.parse_args()
    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
