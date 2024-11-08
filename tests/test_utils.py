import asyncio
import os
import socket
from functools import partial
from typing import AsyncIterator, Tuple

import pytest

from vllm.utils import (FlexibleArgumentParser, StoreBoolean, deprecate_kwargs,
                        get_open_port, merge_async_iterators, supports_kw)

from .utils import error_on_warning


@pytest.mark.asyncio
async def test_merge_async_iterators():

    async def mock_async_iterator(idx: int):
        try:
            while True:
                yield f"item from iterator {idx}"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"iterator {idx} cancelled")

    iterators = [mock_async_iterator(i) for i in range(3)]
    merged_iterator = merge_async_iterators(*iterators,
                                            is_cancelled=partial(asyncio.sleep,
                                                                 0,
                                                                 result=False))

    async def stream_output(generator: AsyncIterator[Tuple[int, str]]):
        async for idx, output in generator:
            print(f"idx: {idx}, output: {output}")

    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for iterator in iterators:
        try:
            # Can use anext() in python >= 3.10
            await asyncio.wait_for(iterator.__anext__(), 1)
        except StopAsyncIteration:
            # All iterators should be cancelled and print this message.
            print("Iterator was cancelled normally")
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e


def test_deprecate_kwargs_always():

    @deprecate_kwargs("old_arg", is_deprecated=True)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_never():

    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_additional_message():

    @deprecate_kwargs("old_arg", is_deprecated=True, additional_message="abcd")
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="abcd"):
        dummy(old_arg=1)


def test_get_open_port():
    os.environ["VLLM_PORT"] = "5678"
    # make sure we can get multiple ports, even if the env var is set
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
        s1.bind(("localhost", get_open_port()))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                s3.bind(("localhost", get_open_port()))
    os.environ.pop("VLLM_PORT")


# Tests for FlexibleArgumentParser
@pytest.fixture
def parser():
    parser = FlexibleArgumentParser()
    parser.add_argument('--image-input-type',
                        choices=['pixel_values', 'image_features'])
    parser.add_argument('--model-name')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--enable-feature', action='store_true')
    return parser


@pytest.fixture
def parser_with_config():
    parser = FlexibleArgumentParser()
    parser.add_argument('serve')
    parser.add_argument('model_tag')
    parser.add_argument('--served-model-name', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--tensor-parallel-size', type=int)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--multi-step-stream-outputs', action=StoreBoolean)
    return parser


def test_underscore_to_dash(parser):
    args = parser.parse_args(['--image_input_type', 'pixel_values'])
    assert args.image_input_type == 'pixel_values'


def test_mixed_usage(parser):
    args = parser.parse_args([
        '--image_input_type', 'image_features', '--model-name',
        'facebook/opt-125m'
    ])
    assert args.image_input_type == 'image_features'
    assert args.model_name == 'facebook/opt-125m'


def test_with_equals_sign(parser):
    args = parser.parse_args(
        ['--image_input_type=pixel_values', '--model-name=facebook/opt-125m'])
    assert args.image_input_type == 'pixel_values'
    assert args.model_name == 'facebook/opt-125m'


def test_with_int_value(parser):
    args = parser.parse_args(['--batch_size', '32'])
    assert args.batch_size == 32
    args = parser.parse_args(['--batch-size', '32'])
    assert args.batch_size == 32


def test_with_bool_flag(parser):
    args = parser.parse_args(['--enable_feature'])
    assert args.enable_feature is True
    args = parser.parse_args(['--enable-feature'])
    assert args.enable_feature is True


def test_invalid_choice(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(['--image_input_type', 'invalid_choice'])


def test_missing_required_argument(parser):
    parser.add_argument('--required-arg', required=True)
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_override_to_config(parser_with_config):
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--config', './data/test_config.yaml',
        '--tensor-parallel-size', '3'
    ])
    assert args.tensor_parallel_size == 3
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--tensor-parallel-size', '3', '--config',
        './data/test_config.yaml'
    ])
    assert args.tensor_parallel_size == 3
    assert args.port == 12312
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--tensor-parallel-size', '3', '--config',
        './data/test_config.yaml', '--port', '666'
    ])
    assert args.tensor_parallel_size == 3
    assert args.port == 666


def test_config_args(parser_with_config):
    args = parser_with_config.parse_args(
        ['serve', 'mymodel', '--config', './data/test_config.yaml'])
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code
    assert not args.multi_step_stream_outputs


def test_config_file(parser_with_config):
    with pytest.raises(FileNotFoundError):
        parser_with_config.parse_args(
            ['serve', 'mymodel', '--config', 'test_config.yml'])

    with pytest.raises(ValueError):
        parser_with_config.parse_args(
            ['serve', 'mymodel', '--config', './data/test_config.json'])

    with pytest.raises(ValueError):
        parser_with_config.parse_args([
            'serve', 'mymodel', '--tensor-parallel-size', '3', '--config',
            '--batch-size', '32'
        ])


def test_no_model_tag(parser_with_config):
    with pytest.raises(ValueError):
        parser_with_config.parse_args(
            ['serve', '--config', './data/test_config.yaml'])


# yapf: enable
@pytest.mark.parametrize(
    "callable,kw_name,requires_kw_only,allow_var_kwargs,is_supported",
    [
        # Tests for positional argument support
        (lambda foo: None, "foo", True, True, False),
        (lambda foo: None, "foo", False, True, True),
        # Tests for positional or keyword / keyword only
        (lambda foo=100: None, "foo", True, True, False),
        (lambda *, foo: None, "foo", False, True, True),
        # Tests to make sure the names of variadic params are NOT supported
        (lambda *args: None, "args", False, True, False),
        (lambda **kwargs: None, "kwargs", False, True, False),
        # Tests for if we allow var kwargs to add support
        (lambda foo: None, "something_else", False, True, False),
        (lambda foo, **kwargs: None, "something_else", False, True, True),
        (lambda foo, **kwargs: None, "kwargs", True, True, False),
        (lambda foo, **kwargs: None, "foo", True, True, False),
    ])
# yapf: disable
def test_supports_kw(callable,kw_name,requires_kw_only,
                     allow_var_kwargs,is_supported):
    assert supports_kw(
        callable=callable,
        kw_name=kw_name,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs
    ) == is_supported
