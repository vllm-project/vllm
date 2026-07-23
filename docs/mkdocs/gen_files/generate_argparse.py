# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util
import logging
import sys
import textwrap
import traceback
from argparse import SUPPRESS, Action, HelpFormatter
from collections.abc import Callable, Iterable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import mkdocs_gen_files
from pydantic_core import core_schema

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from generated_content import append_to_page  # noqa: E402


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


class MockPluggableLayer:
    @staticmethod
    def register(name):
        def decorator(cls):
            return cls

        return decorator


mock_if_no_torch("vllm._C", MagicMock())
mock_if_no_torch("vllm._C_stable_libtorch", MagicMock())
mock_if_no_torch(
    "vllm.model_executor.custom_op",
    MagicMock(CustomOp=MockCustomOp, PluggableLayer=MockPluggableLayer),
)
mock_if_no_torch(
    "vllm.utils.torch_utils", MagicMock(direct_register_custom_op=lambda *a, **k: None)
)


# Mock any version checks by reading from compiled CI requirements
with open(ROOT_DIR / "requirements/test/cuda.txt") as f:
    VERSIONS = dict(line.strip().split("==") for line in f if "==" in line)
importlib.metadata.version = lambda name: VERSIONS.get(name) or "0.0.0"


# Make torch.nn.Parameter safe to inherit from
mock_if_no_torch("torch.nn", MagicMock(Parameter=object))


# Mock torch.library.infer_schema for vllm.ir.ops.IrOpInplaceOverload.__init__
# We need to return the corresponding number of inputs, as IR infra will assert it
def get_outputs(native_fn: Callable) -> str:
    """
    Extract output schema from function's return type annotation,
    e.g. 'Tensor' or 'Tensor, Tensor'.
    """
    import typing

    return_type = typing.get_type_hints(native_fn)["return"]
    origin = typing.get_origin(return_type)
    arg_name = lambda a: a.__name__ if hasattr(a, "__name__") else str(a)
    if origin is tuple:
        args = typing.get_args(return_type)
        return ", ".join(arg_name(arg) for arg in args)
    else:
        return f"{arg_name(return_type)}"


mock_if_no_torch(
    "torch.library",
    MagicMock(infer_schema=lambda fn, **k: f"(Tensor x) -> {get_outputs(fn)}"),
)


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas."""

    def __init__(self, *args, **kwargs):
        name = kwargs.get("name")
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
            logger.exception("Failed to import %s.%s", module_name, attr)
            raise

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
bench_sweep_serve_workload = auto_mock(
    "vllm.benchmarks.sweep.serve_workload", "SweepServeWorkloadArgs"
)
bench_throughput = auto_mock("vllm.benchmarks", "throughput")
AsyncEngineArgs = auto_mock("vllm.engine.arg_utils", "AsyncEngineArgs")
EngineArgs = auto_mock("vllm.engine.arg_utils", "EngineArgs")
ChatCommand = auto_mock("vllm.entrypoints.cli.openai", "ChatCommand")
CompleteCommand = auto_mock("vllm.entrypoints.cli.openai", "CompleteCommand")
RenderSubcommand = auto_mock("vllm.entrypoints.cli.launch", "RenderSubcommand")
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

            if action.choices or isinstance(action.metavar, list | tuple):
                choices_iterable = action.choices or action.metavar
                choices = f"`{'`, `'.join(str(c) for c in choices_iterable)}`"
                self._markdown_output.append(f":   Possible choices: {choices}\n\n")

            if action.help:
                help_dd = ":" + textwrap.indent(action.help, "    ")[1:]
                self._markdown_output.append(f"{help_dd}\n\n")

            # None usually means the default is determined at runtime
            if (default := action.default) != SUPPRESS and default is not None:
                # Make empty string defaults visible
                if default == "":
                    default = '""'
                self._markdown_output.append(f":   Default: `{default}`\n\n")

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


def format_help(parser: FlexibleArgumentParser) -> str:
    """Format a parser's help as markdown using `MarkdownFormatter`."""
    return super(type(parser), parser).format_help()


logger.info("Generating argparse documentation")
logger.debug("Root directory: %s", ROOT_DIR.resolve())

# Argument sections appended to handwritten pages
engine_args = create_parser(EngineArgs.add_cli_args)
async_engine_args = create_parser(AsyncEngineArgs.add_cli_args, async_args_only=True)
append_to_page(
    "configuration/engine_args.md",
    f"## `EngineArgs`\n\n{format_help(engine_args)}"
    f"## `AsyncEngineArgs`\n\n{format_help(async_engine_args)}",
)
append_to_page(
    "cli/bench/mm_processor.md",
    f"## Arguments\n\n{format_help(create_parser(bench_mm_processor.add_cli_args))}",
)

# CLI reference pages generated entirely from their parser: page -> (parser, JSON tip)
pages = {
    "cli/serve.md": (create_parser(openai_cli_args.make_arg_parser), True),
    "cli/chat.md": (create_parser(ChatCommand.add_cli_args), False),
    "cli/complete.md": (create_parser(CompleteCommand.add_cli_args), False),
    "cli/run-batch.md": (create_parser(openai_run_batch.make_arg_parser), True),
    "cli/launch/render.md": (create_parser(RenderSubcommand.add_cli_args), True),
    "cli/bench/latency.md": (create_parser(bench_latency.add_cli_args), True),
    "cli/bench/serve.md": (create_parser(bench_serve.add_cli_args), True),
    "cli/bench/throughput.md": (create_parser(bench_throughput.add_cli_args), True),
    "cli/bench/sweep/plot.md": (create_parser(bench_sweep_plot.add_cli_args), True),
    "cli/bench/sweep/plot_pareto.md": (
        create_parser(bench_sweep_plot_pareto.add_cli_args),
        True,
    ),
    "cli/bench/sweep/serve.md": (create_parser(bench_sweep_serve.add_cli_args), True),
    "cli/bench/sweep/serve_workload.md": (
        create_parser(bench_sweep_serve_workload.add_cli_args),
        True,
    ),
}

JSON_TIP = '## JSON CLI Arguments\n\n--8<-- "docs/cli/json_tip.inc.md"\n\n'

for doc_path, (parser, json_tip) in pages.items():
    title = "vllm " + Path(doc_path).relative_to("cli").with_suffix("").as_posix()
    content = f"# {title.replace('/', ' ')}\n\n"
    if parser.description:
        content += f"## Overview\n\n{parser.description}\n\n"
        # Rendered above instead of at the top of the Arguments section
        parser.description = None
    if json_tip:
        content += JSON_TIP
    content += f"## Arguments\n\n{format_help(parser)}"
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(content)
    logger.debug("CLI reference generated: %s", doc_path)

logger.info("Total argparse docs generated: %d", len(pages) + 2)
