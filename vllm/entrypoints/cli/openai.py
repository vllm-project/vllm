# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import TYPE_CHECKING

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.entrypoints.cli.types import CLISubcommand

if TYPE_CHECKING:
    from vllm.utils import FlexibleArgumentParser


def _register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def _interactive_cli(args: argparse.Namespace) -> tuple[str, OpenAI]:
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


def _print_chat_stream(stream) -> str:
    output = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            output += delta.content
            print(delta.content, end="", flush=True)
    print()
    return output


def _print_completion_stream(stream) -> str:
    output = ""
    for chunk in stream:
        text = chunk.choices[0].text
        if text is not None:
            output += text
            print(text, end="", flush=True)
    print()
    return output


def chat(system_prompt: str | None, model_name: str, client: OpenAI) -> None:
    conversation: list[ChatCompletionMessageParam] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        try:
            input_message = input("> ")
        except EOFError:
            break
        conversation.append({"role": "user", "content": input_message})

        stream = client.chat.completions.create(model=model_name,
                                                messages=conversation,
                                                stream=True)
        output = _print_chat_stream(stream)
        conversation.append({"role": "assistant", "content": output})


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
    name = "chat"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        system_prompt = args.system_prompt
        conversation: list[ChatCompletionMessageParam] = []

        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})

        if args.quick:
            conversation.append({"role": "user", "content": args.quick})

            stream = client.chat.completions.create(model=model_name,
                                                    messages=conversation,
                                                    stream=True)
            output = _print_chat_stream(stream)
            conversation.append({"role": "assistant", "content": output})
            return

        print("Please enter a message for the chat model:")
        while True:
            try:
                input_message = input("> ")
            except EOFError:
                break
            conversation.append({"role": "user", "content": input_message})

            stream = client.chat.completions.create(model=model_name,
                                                    messages=conversation,
                                                    stream=True)
            output = _print_chat_stream(stream)
            conversation.append({"role": "assistant", "content": output})

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Add CLI arguments for the chat command."""
        _add_query_options(parser)
        parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            help=("The system prompt to be added to the chat template, "
                  "used for models that support system prompts."))
        parser.add_argument("-q",
                            "--quick",
                            type=str,
                            metavar="MESSAGE",
                            help=("Send a single prompt as MESSAGE "
                                  "and print the response, then exit."))
        return parser

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "chat",
            help="Generate chat completions via the running API server.",
            description="Generate chat completions via the running API server.",
            usage="vllm chat [options]")
        return ChatCommand.add_cli_args(parser)


class CompleteCommand(CLISubcommand):
    """The `complete` subcommand for the vLLM CLI. """
    name = 'complete'

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)

        if args.quick:
            stream = client.completions.create(model=model_name,
                                               prompt=args.quick,
                                               stream=True)
            _print_completion_stream(stream)
            return

        print("Please enter prompt to complete:")
        while True:
            try:
                input_prompt = input("> ")
            except EOFError:
                break
            stream = client.completions.create(model=model_name,
                                               prompt=input_prompt,
                                               stream=True)
            _print_completion_stream(stream)

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Add CLI arguments for the complete command."""
        _add_query_options(parser)
        parser.add_argument(
            "-q",
            "--quick",
            type=str,
            metavar="PROMPT",
            help=
            "Send a single prompt and print the completion output, then exit.")
        return parser

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "complete",
            help=("Generate text completions based on the given prompt "
                  "via the running API server."),
            description=("Generate text completions based on the given prompt "
                         "via the running API server."),
            usage="vllm complete [options]")
        return CompleteCommand.add_cli_args(parser)


def cmd_init() -> list[CLISubcommand]:
    return [ChatCommand(), CompleteCommand()]
