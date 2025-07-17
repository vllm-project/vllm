# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import readline
import signal
import sys
from typing import TYPE_CHECKING

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.entrypoints.cli.chat_commands import ChatContext, CommandRegistry
from vllm.entrypoints.cli.readline_utils import ReadlineEnhancer
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

            chat_completion = client.chat.completions.create(
                model=model_name, messages=conversation)
            print(chat_completion.choices[0].message.content)
            return

        # Initialize command system if enabled
        use_commands = getattr(args, 'enable_commands', True)
        command_registry = CommandRegistry() if use_commands else None

        # Create context for commands
        context = ChatContext(client=client,
                              model_name=model_name,
                              conversation=conversation,
                              system_prompt=system_prompt)
        readline_enhancer = ReadlineEnhancer(
            command_registry) if use_commands and readline else None

        # Initialize readline enhancements
        if use_commands and readline_enhancer:
            # Get available models for completion
            try:
                models = client.models.list()
                model_names = [m.id for m in models.data]
                readline_enhancer.set_available_models(model_names)
            except Exception:
                pass

        print("Type /help for available commands.")
        if readline:
            print("Use Tab for completion, Up/Down for history.")

        try:
            while True:
                try:
                    input_message = input("> ")
                except EOFError:
                    break

                # Skip empty input
                if not input_message.strip():
                    continue

                is_command_run = False
                if (use_commands and command_registry
                        and command_registry.is_command(input_message)):
                    is_command_run = True
                    cmd_name, cmd_args = command_registry.parse_command(
                        input_message)

                    if cmd_name:
                        command = command_registry.get(cmd_name)

                        if command:
                            result = command.execute(context, cmd_args)

                            if result is None:
                                # Command that prints nothing, just continue
                                continue

                            if result == "__EXIT__":
                                break

                            # Check for the special retry signal
                            if result.startswith("__RETRY__"):
                                # The command has already modified
                                # the conversation history.
                                # We just need to use the content
                                # provided in the signal
                                input_message = result[len("__RETRY__"):]
                                # Treat this as a normal chat message now
                                is_command_run = False
                            else:
                                print(result)
                        else:
                            print(f"Unknown command: /{cmd_name}")

                # If a command was run and it wasn't a retry,
                # start the next loop.
                if is_command_run:
                    continue

                # Chat
                # Always use the context as the source of truth.
                context.conversation.append({
                    "role": "user",
                    "content": input_message
                })

                chat_completion = client.chat.completions.create(
                    model=context.model_name, messages=context.conversation)

                response_message = chat_completion.choices[0].message
                context.conversation.append(response_message)
                print(response_message.content)

        finally:
            if readline_enhancer:
                readline_enhancer.save_history()

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        chat_parser = subparsers.add_parser(
            "chat",
            help="Generate chat completions via the running API server.",
            description="Generate chat completions via the running API server.",
            usage="vllm chat [options]")
        _add_query_options(chat_parser)
        chat_parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            help=("The system prompt to be added to the chat template, "
                  "used for models that support system prompts."))
        chat_parser.add_argument("-q",
                                 "--quick",
                                 type=str,
                                 metavar="MESSAGE",
                                 help=("Send a single prompt as MESSAGE "
                                       "and print the response, then exit."))
        chat_parser.add_argument(
            "--disable-commands",
            action="store_false",
            dest="enable_commands",
            default=True,
            help="Disable chat commands (like /help, /save, etc.)")
        return chat_parser


class CompleteCommand(CLISubcommand):
    """The `complete` subcommand for the vLLM CLI. """
    name = 'complete'

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)

        if args.quick:
            completion = client.completions.create(model=model_name,
                                                   prompt=args.quick)
            print(completion.choices[0].text)
            return

        print("Please enter prompt to complete:")
        while True:
            try:
                input_prompt = input("> ")
            except EOFError:
                break
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
                  "via the running API server."),
            description=("Generate text completions based on the given prompt "
                         "via the running API server."),
            usage="vllm complete [options]")
        _add_query_options(complete_parser)
        complete_parser.add_argument(
            "-q",
            "--quick",
            type=str,
            metavar="PROMPT",
            help=
            "Send a single prompt and print the completion output, then exit.")
        return complete_parser


def cmd_init() -> list[CLISubcommand]:
    return [ChatCommand(), CompleteCommand()]
