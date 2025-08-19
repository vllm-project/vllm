# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import ArgumentError
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, Union

import pytest

from vllm.config import CompilationConfig, config
from vllm.engine.arg_utils import (EngineArgs, contains_type, get_kwargs,
                                   get_type, get_type_hints, is_not_builtin,
                                   is_type, literal_to_kwargs, optional_type,
                                   parse_type)
from vllm.utils import FlexibleArgumentParser


@pytest.mark.parametrize(("type", "value", "expected"), [
    (int, "42", 42),
    (float, "3.14", 3.14),
    (str, "Hello World!", "Hello World!"),
    (json.loads, '{"foo":1,"bar":2}', {
        "foo": 1,
        "bar": 2
    }),
])
def test_parse_type(type, value, expected):
    parse_type_func = parse_type(type)
    assert parse_type_func(value) == expected


def test_optional_type():
    optional_type_func = optional_type(int)
    assert optional_type_func("None") is None
    assert optional_type_func("42") == 42


@pytest.mark.parametrize(("type_hint", "type", "expected"), [
    (int, int, True),
    (int, float, False),
    (list[int], list, True),
    (list[int], tuple, False),
    (Literal[0, 1], Literal, True),
])
def test_is_type(type_hint, type, expected):
    assert is_type(type_hint, type) == expected


@pytest.mark.parametrize(("type_hints", "type", "expected"), [
    ({float, int}, int, True),
    ({int, tuple[int]}, int, True),
    ({int, tuple[int]}, float, False),
    ({str, Literal["x", "y"]}, Literal, True),
])
def test_contains_type(type_hints, type, expected):
    assert contains_type(type_hints, type) == expected


@pytest.mark.parametrize(("type_hints", "type", "expected"), [
    ({int, float}, int, int),
    ({int, float}, str, None),
    ({str, Literal["x", "y"]}, Literal, Literal["x", "y"]),
])
def test_get_type(type_hints, type, expected):
    assert get_type(type_hints, type) == expected


@pytest.mark.parametrize(("type_hints", "expected"), [
    ({Literal[1, 2]}, {
        "type": int,
        "choices": [1, 2]
    }),
    ({str, Literal["x", "y"]}, {
        "type": str,
        "metavar": ["x", "y"]
    }),
    ({Literal[1, "a"]}, Exception),
])
def test_literal_to_kwargs(type_hints, expected):
    context = nullcontext()
    if expected is Exception:
        context = pytest.raises(expected)
    with context:
        assert literal_to_kwargs(type_hints) == expected


@config
@dataclass
class NestedConfig:
    field: int = 1
    """field"""


@config
@dataclass
class DummyConfig:
    regular_bool: bool = True
    """Regular bool with default True"""
    optional_bool: Optional[bool] = None
    """Optional bool with default None"""
    optional_literal: Optional[Literal["x", "y"]] = None
    """Optional literal with default None"""
    tuple_n: tuple[int, ...] = field(default_factory=lambda: (1, 2, 3))
    """Tuple with variable length"""
    tuple_2: tuple[int, int] = field(default_factory=lambda: (1, 2))
    """Tuple with fixed length"""
    list_n: list[int] = field(default_factory=lambda: [1, 2, 3])
    """List with variable length"""
    list_literal: list[Literal[1, 2]] = field(default_factory=list)
    """List with literal choices"""
    list_union: list[Union[str, type[object]]] = field(default_factory=list)
    """List with union type"""
    literal_literal: Literal[Literal[1], Literal[2]] = 1
    """Literal of literals with default 1"""
    json_tip: dict = field(default_factory=dict)
    """Dict which will be JSON in CLI"""
    nested_config: NestedConfig = field(default_factory=NestedConfig)
    """Nested config"""


@pytest.mark.parametrize(("type_hint", "expected"), [
    (int, False),
    (DummyConfig, True),
])
def test_is_not_builtin(type_hint, expected):
    assert is_not_builtin(type_hint) == expected


@pytest.mark.parametrize(
    ("type_hint", "expected"), [
        (Annotated[int, "annotation"], {int}),
        (Optional[int], {int, type(None)}),
        (Annotated[Optional[int], "annotation"], {int, type(None)}),
        (Optional[Annotated[int, "annotation"]], {int, type(None)}),
    ],
    ids=["Annotated", "Optional", "Annotated_Optional", "Optional_Annotated"])
def test_get_type_hints(type_hint, expected):
    assert get_type_hints(type_hint) == expected


def test_get_kwargs():
    kwargs = get_kwargs(DummyConfig)
    print(kwargs)

    # bools should not have their type set
    assert kwargs["regular_bool"].get("type") is None
    assert kwargs["optional_bool"].get("type") is None
    # optional literals should have None as a choice
    assert kwargs["optional_literal"]["choices"] == ["x", "y", "None"]
    # tuples should have the correct nargs
    assert kwargs["tuple_n"]["nargs"] == "+"
    assert kwargs["tuple_2"]["nargs"] == 2
    # lists should work
    assert kwargs["list_n"]["type"] is int
    assert kwargs["list_n"]["nargs"] == "+"
    # lists with literals should have the correct choices
    assert kwargs["list_literal"]["type"] is int
    assert kwargs["list_literal"]["nargs"] == "+"
    assert kwargs["list_literal"]["choices"] == [1, 2]
    # lists with unions should become str type.
    # If not, we cannot know which type to use for parsing
    assert kwargs["list_union"]["type"] is str
    # literals of literals should have merged choices
    assert kwargs["literal_literal"]["choices"] == [1, 2]
    # dict should have json tip in help
    json_tip = "Should either be a valid JSON string or JSON keys"
    assert json_tip in kwargs["json_tip"]["help"]
    # nested config should should construct the nested config
    assert kwargs["nested_config"]["type"]('{"field": 2}') == NestedConfig(2)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (None, dict()),
        ('{"video": {"num_frames": 123} }', {
            "video": {
                "num_frames": 123
            }
        }),
        (
            '{"video": {"num_frames": 123, "fps": 1.0, "foo": "bar"}, "image": {"foo": "bar"} }',  # noqa
            {
                "video": {
                    "num_frames": 123,
                    "fps": 1.0,
                    "foo": "bar"
                },
                "image": {
                    "foo": "bar"
                }
            }),
    ])
def test_media_io_kwargs_parser(arg, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--media-io-kwargs", arg])

    assert args.media_io_kwargs == expected


def test_compilation_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # default value
    args = parser.parse_args([])
    assert args.compilation_config == CompilationConfig()

    # set to O3
    args = parser.parse_args(["-O0"])
    assert args.compilation_config.level == 0

    # set to O 3 (space)
    args = parser.parse_args(["-O", "1"])
    assert args.compilation_config.level == 1

    # set to O 3 (equals)
    args = parser.parse_args(["-O=2"])
    assert args.compilation_config.level == 2

    # set to O.level 3
    args = parser.parse_args(["-O.level", "3"])
    assert args.compilation_config.level == 3

    # set to string form of a dict
    args = parser.parse_args([
        "-O",
        '{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8], '
        '"use_inductor": false}',
    ])
    assert (args.compilation_config.level == 3 and
            args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
            and not args.compilation_config.use_inductor)

    # set to string form of a dict
    args = parser.parse_args([
        "--compilation-config="
        '{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8], '
        '"use_inductor": true}',
    ])
    assert (args.compilation_config.level == 3 and
            args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
            and args.compilation_config.use_inductor)


def test_prefix_cache_default():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])

    engine_args = EngineArgs.from_cli_args(args=args)
    assert (not engine_args.enable_prefix_caching
            ), "prefix caching defaults to off."

    # with flag to turn it on.
    args = parser.parse_args(["--enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_prefix_caching

    # with disable flag to turn it off.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert not engine_args.enable_prefix_caching


# yapf: disable
@pytest.mark.parametrize(("arg", "expected", "option"), [
    (None, None, "mm-processor-kwargs"),
    ("{}", {}, "mm-processor-kwargs"),
    (
        '{"num_crops": 4}',
        {
            "num_crops": 4
        },
        "mm-processor-kwargs"
    ),
    (
        '{"foo": {"bar": "baz"}}',
        {
            "foo":
            {
                "bar": "baz"
            }
        },
        "mm-processor-kwargs"
    ),
    (
        '{"cast_logits_dtype":"bfloat16","sequence_parallel_norm":true,"sequence_parallel_norm_threshold":2048}',
        {
            "cast_logits_dtype": "bfloat16",
            "sequence_parallel_norm": True,
            "sequence_parallel_norm_threshold": 2048,
        },
        "override-neuron-config"
    ),
])
# yapf: enable
def test_composite_arg_parser(arg, expected, option):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args([f"--{option}", arg])
    assert getattr(args, option.replace("-", "_")) == expected


def test_human_readable_model_len():
    # `exit_on_error` disabled to test invalid values below
    parser = EngineArgs.add_cli_args(
        FlexibleArgumentParser(exit_on_error=False))

    args = parser.parse_args([])
    assert args.max_model_len is None

    args = parser.parse_args(["--max-model-len", "1024"])
    assert args.max_model_len == 1024

    # Lower
    args = parser.parse_args(["--max-model-len", "1m"])
    assert args.max_model_len == 1_000_000
    args = parser.parse_args(["--max-model-len", "10k"])
    assert args.max_model_len == 10_000

    # Capital
    args = parser.parse_args(["--max-model-len", "3K"])
    assert args.max_model_len == 1024 * 3
    args = parser.parse_args(["--max-model-len", "10M"])
    assert args.max_model_len == 2**20 * 10

    # Decimal values
    args = parser.parse_args(["--max-model-len", "10.2k"])
    assert args.max_model_len == 10200
    # ..truncated to the nearest int
    args = parser.parse_args(["--max-model-len", "10.212345k"])
    assert args.max_model_len == 10212

    # Invalid (do not allow decimals with binary multipliers)
    for invalid in ["1a", "pwd", "10.24", "1.23M"]:
        with pytest.raises(ArgumentError):
            args = parser.parse_args(["--max-model-len", invalid])
