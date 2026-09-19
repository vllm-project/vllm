# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


class AnalyzePrefixCacheSubcommand(CLISubcommand):
    """The `analyze-prefix-cache` subcommand for the vLLM CLI.

    Offline, no-GPU estimate of how prefix-cache-friendly a request
    dataset is, computed against the same full-block hash-chain vLLM's
    scheduler uses. See https://github.com/vllm-project/vllm/issues/47993.
    """

    name = "analyze-prefix-cache"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.entrypoints.prefix_cache_analysis import (
            analyze,
            load_plain_prompt_jsonl,
        )

        records = load_plain_prompt_jsonl(args.input)
        report = analyze(
            records,
            model=args.model,
            block_size=args.block_size,
            hash_algo=args.hash_algo,
            trust_remote_code=args.trust_remote_code,
            top_k_groups=args.top_k_groups,
        )

        if args.output_format == "json":
            import json

            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "analyze-prefix-cache",
            help=(
                "Offline, no-GPU analysis of prefix-cache friendliness "
                "for a request dataset."
            ),
            description=(
                "Tokenize a JSONL request dataset, compute the same "
                "full-block hash chains vLLM's scheduler uses for "
                "automatic prefix caching, and report cacheability "
                "estimates without running inference or requiring a GPU."
            ),
            usage=(
                "vllm analyze-prefix-cache --model <model> "
                "--input requests.jsonl [options]"
            ),
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model or tokenizer name/path used to tokenize prompts.",
        )
        parser.add_argument(
            "--input",
            type=str,
            required=True,
            help=(
                "Path to a JSONL file of requests. v1 supports plain-"
                "prompt JSONL: one {'prompt': ..., 'id': ...} object per "
                "line ('id' is optional)."
            ),
        )
        parser.add_argument(
            "--format",
            dest="input_format",
            type=str,
            default="plain",
            choices=["plain"],
            help=(
                "Input format. Only 'plain' prompt JSONL is supported in "
                "v1; OpenAI chat/batch JSONL is a planned follow-up."
            ),
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="KV-cache block size to hash against (default: 16).",
        )
        parser.add_argument(
            "--hash-algo",
            type=str,
            default="sha256",
            help="Block hash algorithm, matching --prefix-caching-hash-algo.",
        )
        parser.add_argument(
            "--output-format",
            type=str,
            default="text",
            choices=["text", "json"],
            help="Report format (default: text).",
        )
        parser.add_argument(
            "--top-k-groups",
            type=int,
            default=10,
            help="Number of top shared-prefix groups to report (default: 10).",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code when loading the tokenizer.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [AnalyzePrefixCacheSubcommand()]
