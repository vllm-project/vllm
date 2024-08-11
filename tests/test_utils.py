import asyncio
import os
import socket
import sys
from functools import partial
from typing import (TYPE_CHECKING, Any, AsyncIterator, Awaitable, Protocol,
                    Tuple, TypeVar)

import pytest

from vllm.utils import (FlexibleArgumentParser, deprecate_kwargs,
                        get_open_port, merge_async_iterators)

from .utils import error_on_warning

if sys.version_info < (3, 10):
    if TYPE_CHECKING:
        _AwaitableT = TypeVar("_AwaitableT", bound=Awaitable[Any])
        _AwaitableT_co = TypeVar("_AwaitableT_co",
                                 bound=Awaitable[Any],
                                 covariant=True)

        class _SupportsSynchronousAnext(Protocol[_AwaitableT_co]):

            def __anext__(self) -> _AwaitableT_co:
                ...

    def anext(i: "_SupportsSynchronousAnext[_AwaitableT]", /) -> "_AwaitableT":
        return i.__anext__()


@pytest.mark.asyncio
async def test_merge_async_iterators():

    async def mock_async_iterator(idx: int) -> AsyncIterator[str]:
        try:
            while True:
                yield f"item from iterator {idx}"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"iterator {idx} cancelled")

    iterators = [mock_async_iterator(i) for i in range(3)]
    merged_iterator: AsyncIterator[Tuple[int, str]] = merge_async_iterators(
        *iterators, is_cancelled=partial(asyncio.sleep, 0, result=False))

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
            await asyncio.wait_for(anext(iterator), 1)
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

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_never():

    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
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
