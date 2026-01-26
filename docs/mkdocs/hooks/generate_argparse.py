# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util
import logging
import sys
import traceback
from argparse import SUPPRESS, Action, HelpFormatter
from collections.abc import Iterable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

from pydantic_core import core_schema

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs/generated/argparse"

sys.path.insert(0, str(ROOT_DIR))


def mock_if_no_torch(mock_module: str, mock: MagicMock):
    if not importlib.util.find_spec("torch"):
        sys.modules[mock_module] = mock


# Mock custom op code
class MockCustomOp:
    @staticmethod
    def register(name):
        def decorator(cls):
            return cls

        return decorator


mock_if_no_torch("vllm._C", MagicMock())
mock_if_no_torch("vllm.model_executor.custom_op", MagicMock(CustomOp=MockCustomOp))
mock_if_no_torch(
    "vllm.utils.torch_utils", MagicMock(direct_register_custom_op=lambda *a, **k: None)
)


# Mock any version checks by reading from compiled CI requirements
with open(ROOT_DIR / "requirements/test.txt") as f:
    VERSIONS = dict(line.strip().split("==") for line in f if "==" in line)
importlib.metadata.version = lambda name: VERSIONS.get(name) or "0.0.0"


# Make torch.nn.Parameter safe to inherit from
mock_if_no_torch("torch.nn", MagicMock(Parameter=object))


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas."""

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.__spec__ = ModuleSpec(name, None)

    def __get_pydantic_core_schema__(self, source_type, handler):
        return core_schema.any_schema()


def auto_mock(module_name: str, attr: str, max_mocks: int = 100):
    """Function that automatically mocks missing modules during imports."""
    logger.info("Importing %s from %s", attr, module_name)

    for _ in range(max_mocks):
        try:
            module = importlib.import_module(module_name)

            # First treat attr as an attr, then as a submodule
            if hasattr(module, attr):
                return getattr(module, attr)

            return importlib.import_module(f"{module_name}.{attr}")
        except ModuleNotFoundError as e:
            assert e.name is not None
            logger.info("Mocking %s for argparse doc generation", e.name)
            sys.modules[e.name] = PydanticMagicMock(name=e.name)
        except Exception:
            logger.exception("Failed to import %s.%s: %s", module_name, attr)

    raise ImportError(
        f"Failed to import {module_name}.{attr} after mocking {max_mocks} imports"
    )


bench_latency = auto_mock("vllm.benchmarks", "latency")
bench_mm_processor = auto_mock("vllm.benchmarks", "mm_processor")
bench_serve = auto_mock("vllm.benchmarks", "serve")
bench_sweep_plot = auto_mock("vllm.benchmarks.sweep.plot", "SweepPlotArgs")
bench_sweep_plot_pareto = auto_mock(
    "vllm.benchmarks.sweep.plot_pareto", "SweepPlotParetoArgs"
)
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

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = auto_mock(
        "vllm.utils.argparse_utils", "FlexibleArgumentParser"
    )


class MarkdownFormatter(HelpFormatter):
    """Custom formatter that generates markdown for argument groups."""

    def __init__(self, prog: str, starting_heading_level: int = 3):
        super().__init__(prog, max_help_position=sys.maxsize, width=sys.maxsize)

        self._section_heading_prefix = "#" * starting_heading_level
        self._argument_heading_prefix = "#" * (starting_heading_level + 1)
        self._markdown_output = []

    def start_section(self, heading: str):
        if heading not in {"positional arguments", "options"}:
            heading_md = f"\n{self._section_heading_prefix} {heading}\n\n"
            self._markdown_output.append(heading_md)

    def end_section(self):
        pass

    def add_text(self, text: str):
        if text:
            self._markdown_output.append(f"{text.strip()}\n\n")

    def add_usage(self, usage, actions, groups, prefix=None):
        pass

    def add_arguments(self, actions: Iterable[Action]):
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
                # Skip showing "None" as default - it's not helpful to users
                # since None typically means "auto-detect" or "use runtime default"
                if default is None:
                    pass
                # Make empty string defaults visible
                elif default == "":
                    self._markdown_output.append("Default: `\"\"`\n\n")
                else:
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
    try:
        parser = FlexibleArgumentParser(add_json_tip=False)
        parser.formatter_class = MarkdownFormatter
        with patch("vllm.config.DeviceConfig.__post_init__"):
            _parser = add_cli_args(parser, **kwargs)
    except ModuleNotFoundError as e:
        # Auto-mock runtime imports
        if tb_list := traceback.extract_tb(e.__traceback__):
            path = Path(tb_list[-1].filename).relative_to(ROOT_DIR)
            auto_mock(module_name=".".join(path.parent.parts), attr=path.stem)
            return create_parser(add_cli_args, **kwargs)
        else:
            raise e
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
        "bench_mm_processor": create_parser(bench_mm_processor.add_cli_args),
        "bench_serve": create_parser(bench_serve.add_cli_args),
        "bench_sweep_plot": create_parser(bench_sweep_plot.add_cli_args),
        "bench_sweep_plot_pareto": create_parser(bench_sweep_plot_pareto.add_cli_args),
        "bench_sweep_serve": create_parser(bench_sweep_serve.add_cli_args),
        "bench_sweep_serve_sla": create_parser(bench_sweep_serve_sla.add_cli_args),
        "bench_throughput": create_parser(bench_throughput.add_cli_args),
    }

    # Generate documentation for each parser
    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.inc.md"
        # Specify encoding for building on Windows
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(super(type(parser), parser).format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    on_startup("build", False)
