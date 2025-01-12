# The CLI entrypoint to vLLM.
import argparse
import os
import signal
import sys
from typing import List, Optional

import uvloop
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import vllm.version
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.disagg_connector import run_disagg_connector
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def serve(args: argparse.Namespace) -> None:
    # The default value of `--model`
    if args.model != EngineArgs.model:
        raise ValueError(
            "With `vllm serve`, you should provide the model as a "
            "positional argument instead of via the `--model` option.")

    # EngineArgs expects the model name to be passed as --model.
    args.model = args.model_tag

    uvloop.run(run_server(args))


def connect(args: argparse.Namespace) -> None:
    uvloop.run(run_disagg_connector(args))


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


def env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main():
    env_setup()

    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=vllm.version.__version__)

    subparsers = parser.add_subparsers(required=True, dest="subparser")

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
        default='',
        required=False,
        help="Read CLI options from a config file."
        "Must be a YAML with the following options:"
        "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference"
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

    connect_parser = subparsers.add_parser(
        "connect",
        help="Start the vLLM OpenAI Compatible API server And Connect to other"
        "servers disaggreate prefill and decode",
        usage="vllm connect <model_tag> [options]")
    connect_parser.add_argument("--port",
                                type=int,
                                default=8001,
                                help="The fastapi server port")
    connect_parser.add_argument("--prefill-addr",
                                type=str,
                                required=True,
                                help="The prefill address IP:PORT")
    connect_parser.add_argument("--decode-addr",
                                type=str,
                                required=True,
                                help="The decode address IP:PORT")
    connect_parser.set_defaults(dispatch_function=connect)

    args = parser.parse_args()
    if args.subparser == "serve":
        validate_parsed_serve_args(args)

    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
