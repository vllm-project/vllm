# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentError, ArgumentTypeError

import pytest

from vllm.config import PoolerConfig
from vllm.engine.arg_utils import EngineArgs, nullable_kvs
from vllm.utils import FlexibleArgumentParser


@pytest.mark.parametrize(("arg", "expected"), [
    (None, dict()),
    ("image=16", {
        "image": 16
    }),
    ("image=16,video=2", {
        "image": 16,
        "video": 2
    }),
    ("Image=16, Video=2", {
        "image": 16,
        "video": 2
    }),
])
def test_limit_mm_per_prompt_parser(arg, expected):
    """This functionality is deprecated and will be removed in the future.
    This argument should be passed as JSON string instead.
    
    TODO: Remove with nullable_kvs."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--limit-mm-per-prompt", arg])

    assert args.limit_mm_per_prompt == expected


def test_compilation_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # default value
    args = parser.parse_args([])
    assert args.compilation_config is None

    # set to O3
    args = parser.parse_args(["-O3"])
    assert args.compilation_config.level == 3

    # set to O 3 (space)
    args = parser.parse_args(["-O", "3"])
    assert args.compilation_config.level == 3

    # set to O 3 (equals)
    args = parser.parse_args(["-O=3"])
    assert args.compilation_config.level == 3

    # set to string form of a dict
    args = parser.parse_args([
        "--compilation-config",
        "{'level': 3, 'cudagraph_capture_sizes': [1, 2, 4, 8]}",
    ])
    assert (args.compilation_config.level == 3 and
            args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8])

    # set to string form of a dict
    args = parser.parse_args([
        "--compilation-config="
        "{'level': 3, 'cudagraph_capture_sizes': [1, 2, 4, 8]}",
    ])
    assert (args.compilation_config.level == 3 and
            args.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8])


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


def test_valid_pooling_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([
        '--override-pooler-config',
        '{"pooling_type": "MEAN"}',
    ])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.override_pooler_config == PoolerConfig(
        pooling_type="MEAN", )


@pytest.mark.parametrize(
    ("arg"),
    [
        "image",  # Missing =
        "image=4,image=5",  # Conflicting values
        "image=video=4"  # Too many = in tokenized arg
    ])
def test_bad_nullable_kvs(arg):
    with pytest.raises(ArgumentTypeError):
        nullable_kvs(arg)


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
