# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import ArgumentError
from contextlib import AbstractContextManager, nullcontext
from typing import Annotated, Literal

import pytest
from pydantic import Field

from vllm.config import AttentionConfig, CompilationConfig, config
from vllm.engine.arg_utils import (
    EngineArgs,
    contains_type,
    get_kwargs,
    get_type,
    get_type_hints,
    is_not_builtin,
    is_type,
    literal_to_kwargs,
    optional_type,
    parse_type,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize(
    ("type", "value", "expected"),
    [
        (int, "42", 42),
        (float, "3.14", 3.14),
        (str, "Hello World!", "Hello World!"),
        (json.loads, '{"foo":1,"bar":2}', {"foo": 1, "bar": 2}),
    ],
)
def test_parse_type(type, value, expected):
    parse_type_func = parse_type(type)
    assert parse_type_func(value) == expected


def test_optional_type():
    optional_type_func = optional_type(int)
    assert optional_type_func("None") is None
    assert optional_type_func("42") == 42


@pytest.mark.parametrize(
    ("type_hint", "type", "expected"),
    [
        (int, int, True),
        (int, float, False),
        (list[int], list, True),
        (list[int], tuple, False),
        (Literal[0, 1], Literal, True),
    ],
)
def test_is_type(type_hint, type, expected):
    assert is_type(type_hint, type) == expected


@pytest.mark.parametrize(
    ("type_hints", "type", "expected"),
    [
        ({float, int}, int, True),
        ({int, tuple}, int, True),
        ({int, tuple[int]}, int, True),
        ({int, tuple[int, ...]}, int, True),
        ({int, tuple[int]}, float, False),
        ({int, tuple[int, ...]}, float, False),
        ({str, Literal["x", "y"]}, Literal, True),
    ],
)
def test_contains_type(type_hints, type, expected):
    assert contains_type(type_hints, type) == expected


@pytest.mark.parametrize(
    ("type_hints", "type", "expected"),
    [
        ({int, float}, int, int),
        ({int, float}, str, None),
        ({str, Literal["x", "y"]}, Literal, Literal["x", "y"]),
    ],
)
def test_get_type(type_hints, type, expected):
    assert get_type(type_hints, type) == expected


@pytest.mark.parametrize(
    ("type_hints", "expected"),
    [
        ({Literal[1, 2]}, {"type": int, "choices": [1, 2]}),
        ({str, Literal["x", "y"]}, {"type": str, "metavar": ["x", "y"]}),
        ({Literal[1, "a"]}, Exception),
    ],
)
def test_literal_to_kwargs(type_hints, expected):
    context: AbstractContextManager[object] = nullcontext()
    if expected is Exception:
        context = pytest.raises(expected)
    with context:
        assert literal_to_kwargs(type_hints) == expected


@config
class NestedConfig:
    field: int = 1
    """field"""


@config
class DummyConfig:
    regular_bool: bool = True
    """Regular bool with default True"""
    optional_bool: bool | None = None
    """Optional bool with default None"""
    optional_literal: Literal["x", "y"] | None = None
    """Optional literal with default None"""
    tuple_n: tuple[int, ...] = Field(default_factory=lambda: (1, 2, 3))
    """Tuple with variable length"""
    tuple_2: tuple[int, int] = Field(default_factory=lambda: (1, 2))
    """Tuple with fixed length"""
    list_n: list[int] = Field(default_factory=lambda: [1, 2, 3])
    """List with variable length"""
    list_literal: list[Literal[1, 2]] = Field(default_factory=list)
    """List with literal choices"""
    list_union: list[str | type[object]] = Field(default_factory=list)
    """List with union type"""
    set_n: set[int] = Field(default_factory=lambda: {1, 2, 3})
    """Set with variable length"""
    literal_literal: Literal[Literal[1], Literal[2]] = 1
    """Literal of literals with default 1"""
    json_tip: dict = Field(default_factory=dict)
    """Dict which will be JSON in CLI"""
    nested_config: NestedConfig = Field(default_factory=NestedConfig)
    """Nested config"""


@pytest.mark.parametrize(
    ("type_hint", "expected"),
    [
        (int, False),
        (DummyConfig, True),
    ],
)
def test_is_not_builtin(type_hint, expected):
    assert is_not_builtin(type_hint) == expected


@pytest.mark.parametrize(
    ("type_hint", "expected"),
    [
        (Annotated[int, "annotation"], {int}),
        (int | None, {int, type(None)}),
        (Annotated[int | None, "annotation"], {int, type(None)}),
        (Annotated[int, "annotation"] | None, {int, type(None)}),
    ],
    ids=["Annotated", "or_None", "Annotated_or_None", "or_None_Annotated"],
)
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
    # sets should work like lists
    assert kwargs["set_n"]["type"] is int
    assert kwargs["set_n"]["nargs"] == "+"
    # literals of literals should have merged choices
    assert kwargs["literal_literal"]["choices"] == [1, 2]
    # dict should have json tip in help
    json_tip = "Should either be a valid JSON string or JSON keys"
    assert json_tip in kwargs["json_tip"]["help"]
    # nested config should construct the nested config
    assert kwargs["nested_config"]["type"]('{"field": 2}') == NestedConfig(2)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (None, dict()),
        ('{"video": {"num_frames": 123} }', {"video": {"num_frames": 123}}),
        (
            '{"video": {"num_frames": 123, "fps": 1.0, "foo": "bar"}, "image": {"foo": "bar"} }',  # noqa
            {
                "video": {"num_frames": 123, "fps": 1.0, "foo": "bar"},
                "image": {"foo": "bar"},
            },
        ),
    ],
)
def test_media_io_kwargs_parser(arg, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--media-io-kwargs", arg])

    assert args.media_io_kwargs == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-O", "1"], "1"),
        (["-O", "2"], "2"),
        (["-O", "3"], "3"),
        (["-O0"], "0"),
        (["-O1"], "1"),
        (["-O2"], "2"),
        (["-O3"], "3"),
    ],
)
def test_optimization_level(args, expected):
    """
    Test space-separated optimization levels (-O 1, -O 2, -O 3) map to
    optimization_level.
    """
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed_args = parser.parse_args(args)
    assert parsed_args.optimization_level == expected
    assert parsed_args.compilation_config.mode is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-cc.mode=0"], 0),
        (["-cc.mode=1"], 1),
        (["-cc.mode=2"], 2),
        (["-cc.mode=3"], 3),
    ],
)
def test_mode_parser(args, expected):
    """
    Test compilation config modes (-cc.mode=int) map to compilation_config.
    """
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed_args = parser.parse_args(args)
    assert parsed_args.compilation_config.mode == expected


def test_compilation_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # default value
    args = parser.parse_args([])
    assert args.compilation_config == CompilationConfig()

    # set to string form of a dict
    args = parser.parse_args(
        [
            "-cc",
            '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8], "backend": "eager"}',
        ]
    )
    assert (
        args.compilation_config.mode == 3
        and args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
        and args.compilation_config.backend == "eager"
    )

    # set to string form of a dict
    args = parser.parse_args(
        [
            "--compilation-config="
            '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8], '
            '"backend": "inductor"}',
        ]
    )
    assert (
        args.compilation_config.mode == 3
        and args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
        and args.compilation_config.backend == "inductor"
    )


def test_attention_config():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # default value
    args = parser.parse_args([])
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.attention_config == AttentionConfig()

    # set backend via dot notation
    args = parser.parse_args(["--attention-config.backend", "FLASH_ATTN"])
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.attention_config.backend is not None
    assert engine_args.attention_config.backend.name == "FLASH_ATTN"

    # set backend via --attention-backend shorthand
    args = parser.parse_args(["--attention-backend", "FLASHINFER"])
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.attention_backend is not None
    assert engine_args.attention_backend == "FLASHINFER"

    # set all fields via dot notation
    args = parser.parse_args(
        [
            "--attention-config.backend",
            "FLASH_ATTN",
            "--attention-config.flash_attn_version",
            "3",
            "--attention-config.use_prefill_decode_attention",
            "true",
            "--attention-config.flash_attn_max_num_splits_for_cuda_graph",
            "16",
            "--attention-config.use_cudnn_prefill",
            "true",
            "--attention-config.use_trtllm_ragged_deepseek_prefill",
            "true",
            "--attention-config.use_trtllm_attention",
            "true",
            "--attention-config.disable_flashinfer_prefill",
            "true",
            "--attention-config.disable_flashinfer_q_quantization",
            "true",
        ]
    )
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.attention_config.backend is not None
    assert engine_args.attention_config.backend.name == "FLASH_ATTN"
    assert engine_args.attention_config.flash_attn_version == 3
    assert engine_args.attention_config.use_prefill_decode_attention is True
    assert engine_args.attention_config.flash_attn_max_num_splits_for_cuda_graph == 16
    assert engine_args.attention_config.use_cudnn_prefill is True
    assert engine_args.attention_config.use_trtllm_ragged_deepseek_prefill is True
    assert engine_args.attention_config.use_trtllm_attention is True
    assert engine_args.attention_config.disable_flashinfer_prefill is True
    assert engine_args.attention_config.disable_flashinfer_q_quantization is True

    # set to string form of a dict with all fields
    args = parser.parse_args(
        [
            "--attention-config="
            '{"backend": "FLASHINFER", "flash_attn_version": 2, '
            '"use_prefill_decode_attention": false, '
            '"flash_attn_max_num_splits_for_cuda_graph": 8, '
            '"use_cudnn_prefill": false, '
            '"use_trtllm_ragged_deepseek_prefill": false, '
            '"use_trtllm_attention": false, '
            '"disable_flashinfer_prefill": false, '
            '"disable_flashinfer_q_quantization": false}',
        ]
    )
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.attention_config.backend is not None
    assert engine_args.attention_config.backend.name == "FLASHINFER"
    assert engine_args.attention_config.flash_attn_version == 2
    assert engine_args.attention_config.use_prefill_decode_attention is False
    assert engine_args.attention_config.flash_attn_max_num_splits_for_cuda_graph == 8
    assert engine_args.attention_config.use_cudnn_prefill is False
    assert engine_args.attention_config.use_trtllm_ragged_deepseek_prefill is False
    assert engine_args.attention_config.use_trtllm_attention is False
    assert engine_args.attention_config.disable_flashinfer_prefill is False
    assert engine_args.attention_config.disable_flashinfer_q_quantization is False

    # test --attention-backend flows into VllmConfig.attention_config
    args = parser.parse_args(
        [
            "--model",
            "facebook/opt-125m",
            "--attention-backend",
            "FLASH_ATTN",
        ]
    )
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.attention_config.backend == AttentionBackendEnum.FLASH_ATTN

    # test --attention-config.backend flows into VllmConfig.attention_config
    args = parser.parse_args(
        [
            "--model",
            "facebook/opt-125m",
            "--attention-config.backend",
            "FLASHINFER",
        ]
    )
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.attention_config.backend == AttentionBackendEnum.FLASHINFER

    # test --attention-backend and --attention-config.backend are mutually exclusive
    args = parser.parse_args(
        [
            "--model",
            "facebook/opt-125m",
            "--attention-backend",
            "FLASH_ATTN",
            "--attention-config.backend",
            "FLASHINFER",
        ]
    )
    assert args is not None
    engine_args = EngineArgs.from_cli_args(args)
    with pytest.raises(ValueError, match="mutually exclusive"):
        engine_args.create_engine_config()


def test_prefix_cache_default():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])

    # should be None by default (depends on model).
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_prefix_caching is None

    # with flag to turn it on.
    args = parser.parse_args(["--enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_prefix_caching

    # with disable flag to turn it off.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert not engine_args.enable_prefix_caching


@pytest.mark.parametrize(
    ("arg", "expected", "option"),
    [
        (None, None, "mm-processor-kwargs"),
        ("{}", {}, "mm-processor-kwargs"),
        ('{"num_crops": 4}', {"num_crops": 4}, "mm-processor-kwargs"),
        ('{"foo": {"bar": "baz"}}', {"foo": {"bar": "baz"}}, "mm-processor-kwargs"),
    ],
)
def test_composite_arg_parser(arg, expected, option):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args([f"--{option}", arg])
    assert getattr(args, option.replace("-", "_")) == expected


def test_human_readable_model_len():
    # `exit_on_error` disabled to test invalid values below
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(exit_on_error=False))

    args = parser.parse_args([])
    assert args.max_model_len is None

    args = parser.parse_args(["--max-model-len", "1024"])
    assert args.max_model_len == 1024

    # Lower
    args = parser.parse_args(["--max-model-len", "1m"])
    assert args.max_model_len == 1_000_000
    args = parser.parse_args(["--max-model-len", "10k"])
    assert args.max_model_len == 10_000
    args = parser.parse_args(["--max-model-len", "2g"])
    assert args.max_model_len == 2_000_000_000
    args = parser.parse_args(["--max-model-len", "2t"])
    assert args.max_model_len == 2_000_000_000_000

    # Capital
    args = parser.parse_args(["--max-model-len", "3K"])
    assert args.max_model_len == 2**10 * 3
    args = parser.parse_args(["--max-model-len", "10M"])
    assert args.max_model_len == 2**20 * 10
    args = parser.parse_args(["--max-model-len", "4G"])
    assert args.max_model_len == 2**30 * 4
    args = parser.parse_args(["--max-model-len", "4T"])
    assert args.max_model_len == 2**40 * 4

    # Decimal values
    args = parser.parse_args(["--max-model-len", "10.2k"])
    assert args.max_model_len == 10200
    # ..truncated to the nearest int
    args = parser.parse_args(["--max-model-len", "10.2123451234567k"])
    assert args.max_model_len == 10212
    args = parser.parse_args(["--max-model-len", "10.2123451234567m"])
    assert args.max_model_len == 10212345
    args = parser.parse_args(["--max-model-len", "10.2123451234567g"])
    assert args.max_model_len == 10212345123
    args = parser.parse_args(["--max-model-len", "10.2123451234567t"])
    assert args.max_model_len == 10212345123456

    # Special value -1 for auto-fit to GPU memory
    args = parser.parse_args(["--max-model-len", "-1"])
    assert args.max_model_len == -1

    # 'auto' is an alias for -1
    args = parser.parse_args(["--max-model-len", "auto"])
    assert args.max_model_len == -1
    args = parser.parse_args(["--max-model-len", "AUTO"])
    assert args.max_model_len == -1

    # Invalid (do not allow decimals with binary multipliers)
    for invalid in ["1a", "pwd", "10.24", "1.23M", "1.22T"]:
        with pytest.raises(ArgumentError):
            parser.parse_args(["--max-model-len", invalid])
