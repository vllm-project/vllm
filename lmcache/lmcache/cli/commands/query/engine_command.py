# SPDX-License-Identifier: Apache-2.0
"""``lmcache query engine`` — send one request to an OpenAI-compatible API."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.query._prompt import PromptBuilder
from lmcache.cli.commands.query._request import Request


class EngineCommand(BaseCommand):
    """Send one request to an OpenAI-compatible HTTP API."""

    def name(self) -> str:
        return "engine"

    def help(self) -> str:
        return "Send one request to an OpenAI-compatible HTTP API."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--url", required=True, help="Serving engine base URL.")
        parser.add_argument(
            "--prompt",
            required=True,
            help="Prompt text with optional {name} placeholders.",
        )
        parser.add_argument(
            "--model",
            default=None,
            metavar="ID",
            help="Model ID for the serving engine.",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum completion tokens (default: 128).",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="HTTP timeout in seconds (default: 30).",
        )
        parser.add_argument(
            "--documents",
            action="extend",
            nargs="+",
            default=[],
            metavar="NAME=PATH",
            help=(
                "Load file text for {NAME} in --prompt. "
                "Accepts one or more NAME=PATH values."
            ),
        )
        parser.add_argument(
            "--path",
            dest="documents",
            action="extend",
            nargs="+",
            metavar="NAME=PATH",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--completions",
            action="store_true",
            help="Use POST /v1/completions only.",
        )
        parser.add_argument(
            "--chat-first",
            action="store_true",
            help="Try /v1/chat/completions first, then fall back to /v1/completions.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            prompt_builder = PromptBuilder(args.prompt, args.documents)
            sender = Request(
                base=args.url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                completions_only=args.completions,
                chat_first=args.chat_first,
            )
            answer, engine_stats = sender.send_request(prompt_builder.complete_prompt)

            model_id = args.model or str(engine_stats["model"][1])
            metrics = self.create_metrics("Query Engine", args)
            metrics.add("model", "Model", model_id)
            metrics.add("answer", "Answer", answer)
            prompt_name, prompt_value = engine_stats["prompt_tokens"]
            metrics.add("prompt_tokens", prompt_name, int(prompt_value))
            output_name, output_value = engine_stats["output_tokens"]
            metrics.add("output_tokens", output_name, int(output_value))

            latency = metrics.add_section("latency", "Latency Metrics")
            for key, (name, value) in engine_stats.items():
                if key in ("model", "prompt_tokens", "output_tokens"):
                    continue
                latency.add(key, name, round(float(value), 2))

            metrics.emit()
        except (RuntimeError, ValueError) as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)
