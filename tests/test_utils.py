# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import socket
from typing import AsyncIterator, Tuple
from unittest.mock import patch

import pytest
import torch
from vllm_test_utils import monitor

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.utils import (FlexibleArgumentParser, MemorySnapshot,
                        PlaceholderModule, StoreBoolean, bind_kv_cache,
                        deprecate_kwargs, get_open_port, memory_profiling,
                        merge_async_iterators, supports_kw)

from .utils import error_on_warning, fork_new_process_for_each_test


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
    merged_iterator = merge_async_iterators(*iterators)

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


@fork_new_process_for_each_test
def test_memory_profiling():
    # Fake out some model loading + inference memory usage to test profiling
    # Memory used by other processes will show up as cuda usage outside of torch
    from vllm.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary)
    lib = CudaRTLibrary()
    # 512 MiB allocation outside of this instance
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)

    baseline_snapshot = MemorySnapshot()

    # load weights

    weights = torch.randn(128, 1024, 1024, device='cuda', dtype=torch.float32)

    weights_memory = 128 * 1024 * 1024 * 4 # 512 MiB

    def measure_current_non_torch():
        free, total = torch.cuda.mem_get_info()
        current_used = total - free
        current_torch = torch.cuda.memory_reserved()
        current_non_torch = current_used - current_torch
        return current_non_torch

    with memory_profiling(baseline_snapshot=baseline_snapshot,
    weights_memory=weights_memory) as result, \
        monitor(measure_current_non_torch) as monitored_values:
        # make a memory spike, 1 GiB
        spike = torch.randn(256, 1024, 1024, device='cuda', dtype=torch.float32)
        del spike

        # Add some extra non-torch memory 256 MiB (simulate NCCL)
        handle2 = lib.cudaMalloc(256 * 1024 * 1024)

    # this is an analytic value, it is exact,
    # we only have 256 MiB non-torch memory increase
    measured_diff = monitored_values.values[-1] - monitored_values.values[0]
    assert measured_diff == 256 * 1024 * 1024

    # Check that the memory usage is within 5% of the expected values
    # 5% tolerance is caused by cuda runtime.
    # we cannot control cuda runtime in the granularity of bytes,
    # which causes a small error (<10 MiB in practice)
    non_torch_ratio = result.non_torch_increase / (256 * 1024 * 1024) # noqa
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024
    del weights
    lib.cudaFree(handle1)
    lib.cudaFree(handle2)


def test_bind_kv_cache():
    from vllm.attention import Attention

    ctx = {
        'layers.0.self_attn': Attention(32, 128, 0.1),
        'layers.1.self_attn': Attention(32, 128, 0.1),
        'layers.2.self_attn': Attention(32, 128, 0.1),
        'layers.3.self_attn': Attention(32, 128, 0.1),
    }
    kv_cache = [
        torch.zeros((1, )),
        torch.zeros((1, )),
        torch.zeros((1, )),
        torch.zeros((1, )),
    ]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0]
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[2]
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[3]

def test_bind_kv_cache_non_attention():
    from vllm.attention import Attention

    # example from Jamba PP=2
    ctx = {
        'model.layers.20.attn': Attention(32, 128, 0.1),
        'model.layers.28.attn': Attention(32, 128, 0.1),
    }
    kv_cache = [
        torch.zeros((1, )),
        torch.zeros((1, )),
    ]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx['model.layers.20.attn'].kv_cache[0] is kv_cache[0]
    assert ctx['model.layers.28.attn'].kv_cache[0] is kv_cache[1]


def test_bind_kv_cache_encoder_decoder():
    from vllm.attention import Attention, AttentionType

    # example from bart
    ctx = {
        'encoder.layers.0.self_attn.attn':
            Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER),
        'decoder.layers.0.encoder_attn.attn':
            Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER_DECODER),
        'decoder.layers.0.self_attn.attn':
            Attention(32, 128, 0.1, attn_type=AttentionType.DECODER),
    }

    kv_cache = [
        torch.zeros((1, )),
    ]
    encoder_kv_cache = ctx['encoder.layers.0.self_attn.attn'].kv_cache

    bind_kv_cache(ctx, [kv_cache])
    assert ctx['encoder.layers.0.self_attn.attn'].kv_cache is encoder_kv_cache
    assert ctx['decoder.layers.0.encoder_attn.attn'].kv_cache[0] is kv_cache[0]
    assert ctx['decoder.layers.0.self_attn.attn'].kv_cache[0] is kv_cache[0]


def test_bind_kv_cache_pp():
    with patch("vllm.utils.cuda_device_count_stateless", lambda: 2):
        # this test runs with 1 GPU, but we simulate 2 GPUs
        cfg = VllmConfig(
            parallel_config=ParallelConfig(pipeline_parallel_size=2))
    with set_current_vllm_config(cfg):
        from vllm.attention import Attention

        ctx = {
            'layers.0.self_attn': Attention(32, 128, 0.1),
        }
        kv_cache = [
            [torch.zeros((1, ))],
            [torch.zeros((1, ))]
        ]
        bind_kv_cache(ctx, kv_cache)
        assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0][0]
        assert ctx['layers.0.self_attn'].kv_cache[1] is kv_cache[1][0]


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    def build_ctx():
        return pytest.raises(ModuleNotFoundError,
                             match="No module named")

    with build_ctx():
        int(placeholder)

    with build_ctx():
        placeholder()

    with build_ctx():
        _ = placeholder.some_attr

    with build_ctx():
        # Test conflict with internal __name attribute
        _ = placeholder.name

    # OK to print the placeholder or use it in a f-string
    _ = repr(placeholder)
    _ = str(placeholder)

    # No error yet; only error when it is used downstream
    placeholder_attr = placeholder.placeholder_attr("attr")

    with build_ctx():
        int(placeholder_attr)

    with build_ctx():
        placeholder_attr()

    with build_ctx():
        _ = placeholder_attr.some_attr

    with build_ctx():
        # Test conflict with internal __module attribute
        _ = placeholder_attr.module
