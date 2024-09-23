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


@pytest.mark.parametrize(("arg", "expected"), [
    (None, None),
    ("{}", {}),
    ('{"num_crops": 4}', {
        "num_crops": 4
    }),
    ('{"foo": {"bar": "baz"}}', {
        "foo": {
            "bar": "baz"
        }
    }),
])
def test_mm_processor_kwargs_prompt_parser(arg, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    if arg is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(["--mm-processor-kwargs", arg])
    assert args.mm_processor_kwargs == expected
