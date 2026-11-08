# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import signal
import sys
import time
import traceback
from typing import TYPE_CHECKING

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.entrypoints.cli.types import CLISubcommand

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def _register_signal_handlers():
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTSTP"):
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


def _print_chat_stream(stream, stats: bool = False) -> str:
    output: str = ""
    in_reasoning: bool = False
    start: float = time.perf_counter()
    ttft: float | None = None
    completion_tokens: int = 0
    for chunk in stream:
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if ttft is None and (reasoning or delta.content):
            ttft = time.perf_counter() - start
        if reasoning:
            if not in_reasoning:
                print("<think>", flush=True)
                in_reasoning = True
            print(reasoning, end="", flush=True)
        if delta.content:
            if in_reasoning:
                print("\n</think>", flush=True)
                in_reasoning = False
            output += delta.content
            print(delta.content, end="", flush=True)
    if in_reasoning:
        print("\n</think>", end="", flush=True)
    print()
    if stats:
        _print_metrics(start, ttft, completion_tokens)
    return output


def _print_metrics(start: float, ttft: float | None, completion_tokens: int) -> None:
    total_time = time.perf_counter() - start
    if ttft is None or total_time <= 0:
        return
    print(f"{'TTFT:':<5} {ttft * 1000:.2f} ms")
    print(
        f"{'TPS:':<5} {completion_tokens / total_time:.2f} tokens/s "
        f"({completion_tokens} tokens in {total_time:.2f}s)"
    )


def _print_completion_stream(stream, stats: bool = False) -> str:
    output = ""
    start = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0
    for chunk in stream:
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        text = chunk.choices[0].text
        if text:
            if ttft is None:
                ttft = time.perf_counter() - start
            output += text
            print(text, end="", flush=True)
    print()
    if stats:
        _print_metrics(start, ttft, completion_tokens)
    return output


def chat(
    system_prompt: str | None,
    model_name: str,
    client: OpenAI,
    *,
    stats: bool = False,
    interactive: bool = False,
) -> None:
    conversation: list[ChatCompletionMessageParam] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    initial_conversation = conversation.copy()
    read_input = input
    if interactive and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from prompt_toolkit import PromptSession
        except ImportError:
            print(
                "Line editing is unavailable. Install vllm[chat] to enable it.",
                file=sys.stderr,
            )
        else:
            read_input = PromptSession[str]().prompt

    print("Please enter a message for the chat model:")
    if interactive:
        print("Commands: /help, /clear, /exit. Prefix a message with // to send /.")
    while True:
        try:
            input_message = read_input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if interactive:
            command = input_message.strip()
            if command == "/exit":
                break
            if command == "/clear":
                conversation = initial_conversation.copy()
                print("Conversation cleared; system prompt preserved.")
                continue
            if command == "/help":
                print("Commands: /help, /clear, /exit. Use // to send a leading /.")
                continue
            if input_message.startswith("//"):
                input_message = input_message[1:]
            elif command.startswith("/"):
                print("Unknown command. Use /help, or // to send a leading /.")
                continue
            if not input_message.strip():
                continue

        messages = conversation + [{"role": "user", "content": input_message}]
        create_kwargs = {"model": model_name, "stream": True}
        if stats:
            create_kwargs["stream_options"] = {"include_usage": True}
        try:
            with client.chat.completions.create(
                messages=messages, **create_kwargs
            ) as stream:
                output = _print_chat_stream(stream, stats)
        except APIError:
            if not interactive:
                raise
            traceback.print_exc()
            print(
                "\nRequest failed or response was interrupted; this turn was not "
                "saved. Previous conversation is unchanged. "
                "Use /clear to start over or enter another message.",
                file=sys.stderr,
            )
            continue
        conversation = messages + [{"role": "assistant", "content": output}]


def _add_query_options(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-Compatible RESTful API server",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "The model name used in prompt completion, default to "
            "the first model in list models API call."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
            " It is important to note that this option only applies to the "
            "OpenAI-compatible API endpoints and NOT other endpoints that may "
            "be present in the server. See the security guide in the vLLM docs "
            "for more details."
        ),
    )
    return parser


class ChatCommand(CLISubcommand):
    """The `chat` subcommand for the vLLM CLI."""

    name = "chat"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        system_prompt = args.system_prompt
        stats = args.stats
        conversation: list[ChatCompletionMessageParam] = []

        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})

        create_kwargs = {"model": model_name, "stream": True}
        if stats:
            create_kwargs["stream_options"] = {"include_usage": True}

        if args.quick is not None:
            conversation.append({"role": "user", "content": args.quick})

            stream = client.chat.completions.create(
                messages=conversation, **create_kwargs
            )
            output = _print_chat_stream(stream, stats)
            conversation.append({"role": "assistant", "content": output})
            return

        chat(
            system_prompt,
            model_name,
            client,
            stats=stats,
            interactive=args.interactive,
        )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Add CLI arguments for the chat command."""
        _add_query_options(parser)
        parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            help=(
                "The system prompt to be added to the chat template, "
                "used for models that support system prompts."
            ),
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "-q",
            "--quick",
            type=str,
            metavar="MESSAGE",
            help=("Send a single prompt as MESSAGE and print the response, then exit."),
        )
        mode.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help=(
                "Enable /help, /clear, /exit and recovery from API errors. "
                "Install vllm[chat] for terminal line editing and session history. "
                "Full error tracebacks are preserved. Disabled by default. "
                "Cannot be combined with --quick."
            ),
        )
        parser.add_argument(
            "--stats",
            action="store_true",
            help="Print TTFT and TPS statistics after each response.",
        )
        return parser

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "chat",
            help="Generate chat completions via the running API server.",
            description="Generate chat completions via the running API server.",
            usage="vllm chat [options]",
        )
        return ChatCommand.add_cli_args(parser)


class CompleteCommand(CLISubcommand):
    """The `complete` subcommand for the vLLM CLI."""

    name = "complete"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        stats = args.stats

        kwargs = {
            "model": model_name,
            "stream": True,
        }
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
        if stats:
            kwargs["stream_options"] = {"include_usage": True}

        if args.quick is not None:
            stream = client.completions.create(prompt=args.quick, **kwargs)
            _print_completion_stream(stream, stats)
            return

        print("Please enter prompt to complete:")
        while True:
            try:
                input_prompt = input("> ")
            except EOFError:
                break
            stream = client.completions.create(prompt=input_prompt, **kwargs)
            _print_completion_stream(stream, stats)

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Add CLI arguments for the complete command."""
        _add_query_options(parser)
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="Maximum number of tokens to generate per output sequence.",
        )
        parser.add_argument(
            "-q",
            "--quick",
            type=str,
            metavar="PROMPT",
            help="Send a single prompt and print the completion output, then exit.",
        )
        parser.add_argument(
            "--stats",
            action="store_true",
            help="Print TTFT and TPS statistics after each response.",
        )
        return parser

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "complete",
            help=(
                "Generate text completions based on the given prompt "
                "via the running API server."
            ),
            description=(
                "Generate text completions based on the given prompt "
                "via the running API server."
            ),
            usage="vllm complete [options]",
        )
        return CompleteCommand.add_cli_args(parser)


def cmd_init() -> list[CLISubcommand]:
    return [ChatCommand(), CompleteCommand()]
