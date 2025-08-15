# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import sys
from argparse import SUPPRESS, HelpFormatter
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs/argparse"

sys.path.insert(0, str(ROOT_DIR))
sys.modules["aiohttp"] = MagicMock()
sys.modules["blake3"] = MagicMock()
sys.modules["vllm._C"] = MagicMock()

from vllm.benchmarks import latency  # noqa: E402
from vllm.benchmarks import serve  # noqa: E402
from vllm.benchmarks import throughput  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # noqa: E402
from vllm.entrypoints.cli.openai import ChatCommand  # noqa: E402
from vllm.entrypoints.cli.openai import CompleteCommand  # noqa: E402
from vllm.entrypoints.openai import cli_args  # noqa: E402
from vllm.entrypoints.openai import run_batch  # noqa: E402
from vllm.utils import FlexibleArgumentParser  # noqa: E402

logger = logging.getLogger("mkdocs")


class MarkdownFormatter(HelpFormatter):
    """Custom formatter that generates markdown for argument groups."""

    def __init__(self, prog, starting_heading_level=3):
        super().__init__(prog,
                         max_help_position=float('inf'),
                         width=float('inf'))
        self._section_heading_prefix = "#" * starting_heading_level
        self._argument_heading_prefix = "#" * (starting_heading_level + 1)
        self._markdown_output = []

    def start_section(self, heading):
        if heading not in {"positional arguments", "options"}:
            heading_md = f"\n{self._section_heading_prefix} {heading}\n\n"
            self._markdown_output.append(heading_md)

    def end_section(self):
        pass

    def add_text(self, text):
        if text:
            self._markdown_output.append(f"{text.strip()}\n\n")

    def add_usage(self, usage, actions, groups, prefix=None):
        pass

    def add_arguments(self, actions):
        for action in actions:
            if (len(action.option_strings) == 0
                    or "--help" in action.option_strings):
                continue

            option_strings = f'`{"`, `".join(action.option_strings)}`'
            heading_md = f"{self._argument_heading_prefix} {option_strings}\n\n"
            self._markdown_output.append(heading_md)

            if choices := action.choices:
                choices = f'`{"`, `".join(str(c) for c in choices)}`'
                self._markdown_output.append(
                    f"Possible choices: {choices}\n\n")
            elif ((metavar := action.metavar)
                  and isinstance(metavar, (list, tuple))):
                metavar = f'`{"`, `".join(str(m) for m in metavar)}`'
                self._markdown_output.append(
                    f"Possible choices: {metavar}\n\n")

            if action.help:
                self._markdown_output.append(f"{action.help}\n\n")

            if (default := action.default) != SUPPRESS:
                self._markdown_output.append(f"Default: `{default}`\n\n")

    def format_help(self):
        """Return the formatted help as markdown."""
        return "".join(self._markdown_output)


def create_parser(add_cli_args, **kwargs) -> FlexibleArgumentParser:
    """Create a parser for the given class with markdown formatting.
    
    Args:
        cls: The class to create a parser for
        **kwargs: Additional keyword arguments to pass to `cls.add_cli_args`.

    Returns:
        FlexibleArgumentParser: A parser with markdown formatting for the class.
    """
    parser = FlexibleArgumentParser(add_json_tip=False)
    parser.formatter_class = MarkdownFormatter
    with patch("vllm.config.DeviceConfig.__post_init__"):
        _parser = add_cli_args(parser, **kwargs)
    # add_cli_args might be in-place so return parser if _parser is None
    return _parser or parser


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    logger.info("Generating argparse documentation")
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Output directory: %s", ARGPARSE_DOC_DIR.resolve())

    # Create the ARGPARSE_DOC_DIR if it doesn't exist
    if not ARGPARSE_DOC_DIR.exists():
        ARGPARSE_DOC_DIR.mkdir(parents=True)

    # Create parsers to document
    parsers = {
        "engine_args":
        create_parser(EngineArgs.add_cli_args),
        "async_engine_args":
        create_parser(AsyncEngineArgs.add_cli_args, async_args_only=True),
        "serve":
        create_parser(cli_args.make_arg_parser),
        "chat":
        create_parser(ChatCommand.add_cli_args),
        "complete":
        create_parser(CompleteCommand.add_cli_args),
        "bench_latency":
        create_parser(latency.add_cli_args),
        "bench_throughput":
        create_parser(throughput.add_cli_args),
        "bench_serve":
        create_parser(serve.add_cli_args),
        "run-batch":
        create_parser(run_batch.make_arg_parser),
    }

    # Generate documentation for each parser
    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.md"
        with open(doc_path, "w") as f:
            f.write(parser.format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))
