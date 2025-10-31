# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import logging
import sys
from argparse import SUPPRESS, HelpFormatter
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

from pydantic_core import core_schema

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs/argparse"

sys.path.insert(0, str(ROOT_DIR))
sys.modules["vllm._C"] = MagicMock()


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas."""

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)

    def __get_pydantic_core_schema__(self, source_type, handler):
        return core_schema.any_schema()


def auto_mock(module, attr, max_mocks=50):
    """Function that automatically mocks missing modules during imports."""
    logger.info("Importing %s from %s", attr, module)
    for _ in range(max_mocks):
        try:
            # First treat attr as an attr, then as a submodule
            with patch("importlib.metadata.version", return_value="0.0.0"):
                return getattr(
                    importlib.import_module(module),
                    attr,
                    importlib.import_module(f"{module}.{attr}"),
                )
        except importlib.metadata.PackageNotFoundError as e:
            raise e
        except ModuleNotFoundError as e:
            logger.info("Mocking %s for argparse doc generation", e.name)
            sys.modules[e.name] = PydanticMagicMock(name=e.name)
        except Exception as e:
            logger.warning("Failed to import %s.%s: %s", module, attr, e)

    raise ImportError(
        f"Failed to import {module}.{attr} after mocking {max_mocks} imports"
    )


bench_latency = auto_mock("vllm.benchmarks", "latency")
bench_serve = auto_mock("vllm.benchmarks", "serve")
bench_sweep_plot = auto_mock("vllm.benchmarks.sweep.plot", "SweepPlotArgs")
bench_sweep_serve = auto_mock("vllm.benchmarks.sweep.serve", "SweepServeArgs")
bench_sweep_serve_sla = auto_mock(
    "vllm.benchmarks.sweep.serve_sla", "SweepServeSLAArgs"
)
bench_throughput = auto_mock("vllm.benchmarks", "throughput")
AsyncEngineArgs = auto_mock("vllm.engine.arg_utils", "AsyncEngineArgs")
EngineArgs = auto_mock("vllm.engine.arg_utils", "EngineArgs")
ChatCommand = auto_mock("vllm.entrypoints.cli.openai", "ChatCommand")
CompleteCommand = auto_mock("vllm.entrypoints.cli.openai", "CompleteCommand")
openai_cli_args = auto_mock("vllm.entrypoints.openai", "cli_args")
openai_run_batch = auto_mock("vllm.entrypoints.openai", "run_batch")
FlexibleArgumentParser = auto_mock(
    "vllm.utils.argparse_utils", "FlexibleArgumentParser"
)


class MarkdownFormatter(HelpFormatter):
    """Custom formatter that generates markdown for argument groups."""

    def __init__(self, prog, starting_heading_level=3):
        super().__init__(prog, max_help_position=float("inf"), width=float("inf"))
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
            if len(action.option_strings) == 0 or "--help" in action.option_strings:
                continue

            option_strings = f"`{'`, `'.join(action.option_strings)}`"
            heading_md = f"{self._argument_heading_prefix} {option_strings}\n\n"
            self._markdown_output.append(heading_md)

            if choices := action.choices:
                choices = f"`{'`, `'.join(str(c) for c in choices)}`"
                self._markdown_output.append(f"Possible choices: {choices}\n\n")
            elif (metavar := action.metavar) and isinstance(metavar, (list, tuple)):
                metavar = f"`{'`, `'.join(str(m) for m in metavar)}`"
                self._markdown_output.append(f"Possible choices: {metavar}\n\n")

            if action.help:
                self._markdown_output.append(f"{action.help}\n\n")

            if (default := action.default) != SUPPRESS:
                # Make empty string defaults visible
                if default == "":
                    default = '""'
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
        # Engine args
        "engine_args": create_parser(EngineArgs.add_cli_args),
        "async_engine_args": create_parser(
            AsyncEngineArgs.add_cli_args, async_args_only=True
        ),
        # CLI
        "serve": create_parser(openai_cli_args.make_arg_parser),
        "chat": create_parser(ChatCommand.add_cli_args),
        "complete": create_parser(CompleteCommand.add_cli_args),
        "run-batch": create_parser(openai_run_batch.make_arg_parser),
        # Benchmark CLI
        "bench_latency": create_parser(bench_latency.add_cli_args),
        "bench_serve": create_parser(bench_serve.add_cli_args),
        "bench_sweep_plot": create_parser(bench_sweep_plot.add_cli_args),
        "bench_sweep_serve": create_parser(bench_sweep_serve.add_cli_args),
        "bench_sweep_serve_sla": create_parser(bench_sweep_serve_sla.add_cli_args),
        "bench_throughput": create_parser(bench_throughput.add_cli_args),
    }

    # Generate documentation for each parser
    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.md"
        # Specify encoding for building on Windows
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(super(type(parser), parser).format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))
