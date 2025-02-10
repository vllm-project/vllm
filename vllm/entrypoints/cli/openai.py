# SPDX-License-Identifier: Apache-2.0
# Commands that act as an interactive OpenAI API client

import argparse
import os
import signal
import sys
from typing import List, Optional, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


def _register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def _interactive_cli(args: argparse.Namespace) -> Tuple[str, OpenAI]:
    _register_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    return model_name, openai_client


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation: List[ChatCompletionMessageParam] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        try:
            input_message = input("> ")
        except EOFError:
            return
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


class ChatCommand(CLISubcommand):
    """The `chat` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "chat"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        system_prompt = args.system_prompt
        conversation: List[ChatCompletionMessageParam] = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})

        print("Please enter a message for the chat model:")
        while True:
            try:
                input_message = input("> ")
            except EOFError:
                return
            conversation.append({"role": "user", "content": input_message})

            chat_completion = client.chat.completions.create(
                model=model_name, messages=conversation)

            response_message = chat_completion.choices[0].message
            output = response_message.content

            conversation.append(response_message)  # type: ignore
            print(output)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
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
        return chat_parser


class CompleteCommand(CLISubcommand):
    """The `complete` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "complete"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        print("Please enter prompt to complete:")
        while True:
            input_prompt = input("> ")
            completion = client.completions.create(model=model_name,
                                                   prompt=input_prompt)
            output = completion.choices[0].text
            print(output)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        complete_parser = subparsers.add_parser(
            "complete",
            help=("Generate text completions based on the given prompt "
                  "via the running API server"),
            usage="vllm complete [options]")
        _add_query_options(complete_parser)
        return complete_parser


def cmd_init() -> List[CLISubcommand]:
    return [ChatCommand(), CompleteCommand()]
