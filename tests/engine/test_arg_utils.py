from argparse import ArgumentTypeError

import pytest

from vllm.engine.arg_utils import EngineArgs, nullable_kvs
from vllm.utils import FlexibleArgumentParser


@pytest.mark.parametrize(("arg", "expected"), [
    (None, None),
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
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--limit-mm-per-prompt", arg])

    assert args.limit_mm_per_prompt == expected


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
