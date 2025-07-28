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

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # noqa: E402
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
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

            self._markdown_output.append(f"{action.help}\n\n")

            if (default := action.default) != SUPPRESS:
                self._markdown_output.append(f"Default: `{default}`\n\n")

    def format_help(self):
        """Return the formatted help as markdown."""
        return "".join(self._markdown_output)


def create_parser(cls, **kwargs) -> FlexibleArgumentParser:
    """Create a parser for the given class with markdown formatting.
    
    Args:
        cls: The class to create a parser for
        **kwargs: Additional keyword arguments to pass to `cls.add_cli_args`.

    Returns:
        FlexibleArgumentParser: A parser with markdown formatting for the class.
    """
    parser = FlexibleArgumentParser()
    parser.formatter_class = MarkdownFormatter
    with patch("vllm.config.DeviceConfig.__post_init__"):
        return cls.add_cli_args(parser, **kwargs)


def create_serve_parser() -> FlexibleArgumentParser:
    """Create a parser for the serve command with markdown formatting."""
    parser = FlexibleArgumentParser()
    parser.formatter_class = lambda prog: MarkdownFormatter(
        prog, starting_heading_level=4)
    return make_arg_parser(parser)


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    logger.info("Generating argparse documentation")
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Output directory: %s", ARGPARSE_DOC_DIR.resolve())

    # Create the ARGPARSE_DOC_DIR if it doesn't exist
    if not ARGPARSE_DOC_DIR.exists():
        ARGPARSE_DOC_DIR.mkdir(parents=True)

    # Create parsers to document
    parsers = {
        "engine_args": create_parser(EngineArgs),
        "async_engine_args": create_parser(AsyncEngineArgs,
                                           async_args_only=True),
        "serve": create_serve_parser(),
    }

    # Generate documentation for each parser
    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.md"
        with open(doc_path, "w") as f:
            f.write(parser.format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))
