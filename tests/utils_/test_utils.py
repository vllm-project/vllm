# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import asyncio
import hashlib
import json
import pickle
import socket
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
import torch
import zmq
from transformers import AutoTokenizer
from vllm_test_utils.monitor import monitor

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.transformers_utils.detokenizer_utils import (
    convert_ids_list_to_tokens)
from vllm.utils import (CacheInfo, FlexibleArgumentParser, LRUCache,
                        MemorySnapshot, PlaceholderModule, StoreBoolean,
                        bind_kv_cache, common_broadcastable_dtype,
                        current_stream, deprecate_kwargs, get_open_port,
                        get_tcp_uri, is_lossless_cast, join_host_port,
                        make_zmq_path, make_zmq_socket, memory_profiling,
                        merge_async_iterators, sha256, split_host_port,
                        split_zmq_path, supports_kw, swap_dict_values)

from ..utils import create_new_process_for_each_test, error_on_warning


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

    async def stream_output(generator: AsyncIterator[tuple[int, str]]):
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


def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        # make sure we can get multiple ports, even if the env var is set
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("localhost", get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(("localhost", get_open_port()))


# Tests for FlexibleArgumentParser
@pytest.fixture
def parser():
    parser = FlexibleArgumentParser()
    parser.add_argument('--image-input-type',
                        choices=['pixel_values', 'image_features'])
    parser.add_argument('--model-name')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--enable-feature', action='store_true')
    parser.add_argument('--hf-overrides', type=json.loads)
    parser.add_argument('-O', '--compilation-config', type=json.loads)
    return parser


@pytest.fixture
def parser_with_config():
    parser = FlexibleArgumentParser()
    parser.add_argument('serve')
    parser.add_argument('model_tag', nargs='?')
    parser.add_argument('--model', type=str)
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


def test_cli_override_to_config(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--config', cli_config_file,
        '--tensor-parallel-size', '3'
    ])
    assert args.tensor_parallel_size == 3
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--tensor-parallel-size', '3', '--config',
        cli_config_file
    ])
    assert args.tensor_parallel_size == 3
    assert args.port == 12312
    args = parser_with_config.parse_args([
        'serve', 'mymodel', '--tensor-parallel-size', '3', '--config',
        cli_config_file, '--port', '666'
    ])
    assert args.tensor_parallel_size == 3
    assert args.port == 666


def test_config_args(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args(
        ['serve', 'mymodel', '--config', cli_config_file])
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


def test_no_model_tag(parser_with_config, cli_config_file):
    with pytest.raises(ValueError):
        parser_with_config.parse_args(['serve', '--config', cli_config_file])


def test_dict_args(parser):
    args = [
        "--model-name=something.something",
        "--hf-overrides.key1",
        "val1",
        # Test nesting
        "--hf-overrides.key2.key3",
        "val2",
        "--hf-overrides.key2.key4",
        "val3",
        # Test compile config and compilation level
        "-O.use_inductor=true",
        "-O.backend",
        "custom",
        "-O1",
        # Test = sign
        "--hf-overrides.key5=val4",
        # Test underscore to dash conversion
        "--hf_overrides.key_6",
        "val5",
        "--hf_overrides.key-7.key_8",
        "val6",
        # Test data type detection
        "--hf_overrides.key9",
        "100",
        "--hf_overrides.key10",
        "100.0",
        "--hf_overrides.key11",
        "true",
        "--hf_overrides.key12.key13",
        "null",
        # Test '-' and '.' in value
        "--hf_overrides.key14.key15",
        "-minus.and.dot",
        # Test array values
        "-O.custom_ops+",
        "-quant_fp8",
        "-O.custom_ops+=+silu_mul,-rms_norm",
    ]
    parsed_args = parser.parse_args(args)
    assert parsed_args.model_name == "something.something"
    assert parsed_args.hf_overrides == {
        "key1": "val1",
        "key2": {
            "key3": "val2",
            "key4": "val3",
        },
        "key5": "val4",
        "key_6": "val5",
        "key-7": {
            "key_8": "val6",
        },
        "key9": 100,
        "key10": 100.0,
        "key11": True,
        "key12": {
            "key13": None,
        },
        "key14": {
            "key15": "-minus.and.dot",
        }
    }
    assert parsed_args.compilation_config == {
        "level": 1,
        "use_inductor": True,
        "backend": "custom",
        "custom_ops": ["-quant_fp8", "+silu_mul", "-rms_norm"],
    }


def test_duplicate_dict_args(caplog_vllm, parser):
    args = [
        "--model-name=something.something",
        "--hf-overrides.key1",
        "val1",
        "--hf-overrides.key1",
        "val2",
        "-O1",
        "-O.level",
        "2",
        "-O3",
    ]

    parsed_args = parser.parse_args(args)
    # Should be the last value
    assert parsed_args.hf_overrides == {"key1": "val2"}
    assert parsed_args.compilation_config == {"level": 3}

    assert len(caplog_vllm.records) == 1
    assert "duplicate" in caplog_vllm.text
    assert "--hf-overrides.key1" in caplog_vllm.text
    assert "-O.level" in caplog_vllm.text


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


@create_new_process_for_each_test()
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

def test_bind_kv_cache_kv_sharing():
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
    shared_kv_cache_layers = {
        'layers.2.self_attn': 'layers.1.self_attn',
        'layers.3.self_attn': 'layers.0.self_attn'
    }
    bind_kv_cache(ctx, [kv_cache], shared_kv_cache_layers)
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0]
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[0]

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


def test_bind_kv_cache_encoder_decoder(monkeypatch: pytest.MonkeyPatch):
    # V1 TESTS: ENCODER_DECODER is not supported on V1 yet.
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")

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


class TestLRUCache(LRUCache):

    def _on_remove(self, key, value):
        if not hasattr(self, "_remove_counter"):
            self._remove_counter = 0
        self._remove_counter += 1


def test_lru_cache():
    cache = TestLRUCache(3)
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)

    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(2, 2)
    assert len(cache) == 2

    cache.put(3, 3)
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}

    cache.put(4, 4)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1

    assert cache.get(2) == 2
    assert cache.stat() == CacheInfo(hits=1, total=1)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)

    assert cache[2] == 2
    assert cache.stat() == CacheInfo(hits=2, total=2)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)

    cache.put(5, 5)
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2

    assert cache.pop(5) == 5
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    assert cache.get(-1) is None
    assert cache.stat() == CacheInfo(hits=2, total=3)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=1)

    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.get(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.put(6, 6)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache

    cache.remove_oldest()
    assert len(cache) == 2
    assert set(cache.cache) == {2, 6}
    assert cache._remove_counter == 4

    cache.clear()
    assert len(cache) == 0
    assert cache._remove_counter == 6
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)

    cache._remove_counter = 0

    cache[1] = 1
    assert len(cache) == 1

    cache[1] = 1
    assert len(cache) == 1

    cache[2] = 2
    assert len(cache) == 2

    cache[3] = 3
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}

    cache[4] = 4
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache[2] == 2

    cache[5] = 5
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2

    del cache[5]
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache[6] = 6
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache


# yapf: disable
@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
# yapf: enable
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


# yapf: disable
@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
# yapf: enable
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    def build_ctx():
        return pytest.raises(ModuleNotFoundError, match="No module named")

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


# yapf: disable
@pytest.mark.parametrize(
    "obj,key1,key2",
    [
        # Tests for both keys exist
        ({1: "a", 2: "b"}, 1, 2),
        # Tests for one key does not exist
        ({1: "a", 2: "b"}, 1, 3),
        # Tests for both keys do not exist
        ({1: "a", 2: "b"}, 3, 4),
    ])
# yapf: enable
def test_swap_dict_values(obj, key1, key2):
    original_obj = obj.copy()
    swap_dict_values(obj, key1, key2)
    if key1 in original_obj:
        assert obj[key2] == original_obj[key1]
    else:
        assert key2 not in obj
    if key2 in original_obj:
        assert obj[key1] == original_obj[key2]
    else:
        assert key1 not in obj


def test_model_specification(parser_with_config, cli_config_file,
                             cli_config_file_with_model):
    # Test model in CLI takes precedence over config
    args = parser_with_config.parse_args(
        ['serve', 'cli-model', '--config', cli_config_file_with_model])
    assert args.model_tag == 'cli-model'
    assert args.served_model_name == 'mymodel'

    # Test model from config file works
    args = parser_with_config.parse_args([
        'serve',
        '--config',
        cli_config_file_with_model,
    ])
    assert args.model == 'config-model'
    assert args.served_model_name == 'mymodel'

    # Test no model specified anywhere raises error
    with pytest.raises(ValueError, match="No model specified!"):
        parser_with_config.parse_args(['serve', '--config', cli_config_file])

    # Test using --model option raises error
    with pytest.raises(
            ValueError,
            match=
        ("With `vllm serve`, you should provide the model as a positional "
         "argument or in a config file instead of via the `--model` option."),
    ):
        parser_with_config.parse_args(['serve', '--model', 'my-model'])

    # Test other config values are preserved
    args = parser_with_config.parse_args([
        'serve',
        'cli-model',
        '--config',
        cli_config_file_with_model,
    ])
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code is True
    assert args.multi_step_stream_outputs is False
    assert args.port == 12312


@pytest.mark.parametrize("input", [(), ("abc", ), (None, ),
                                   (None, bool, [1, 2, 3])])
@pytest.mark.parametrize("output", [0, 1, 2])
def test_sha256(input: tuple, output: int):
    hash = sha256(input)
    assert hash is not None
    assert isinstance(hash, int)
    assert hash != 0

    bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    assert hash == int.from_bytes(hashlib.sha256(bytes).digest(),
                                  byteorder="big")

    # hashing again, returns the same value
    assert hash == sha256(input)

    # hashing different input, returns different value
    assert hash != sha256(input + (1, ))


@pytest.mark.parametrize(
    "path,expected",
    [
        ("ipc://some_path", ("ipc", "some_path", "")),
        ("tcp://127.0.0.1:5555", ("tcp", "127.0.0.1", "5555")),
        ("tcp://[::1]:5555", ("tcp", "::1", "5555")),  # IPv6 address
        ("inproc://some_identifier", ("inproc", "some_identifier", "")),
    ])
def test_split_zmq_path(path, expected):
    assert split_zmq_path(path) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        "invalid_path",  # Missing scheme
        "tcp://127.0.0.1",  # Missing port
        "tcp://[::1]",  # Missing port for IPv6
        "tcp://:5555",  # Missing host
    ])
def test_split_zmq_path_invalid(invalid_path):
    with pytest.raises(ValueError):
        split_zmq_path(invalid_path)


def test_make_zmq_socket_ipv6():
    # Check if IPv6 is supported by trying to create an IPv6 socket
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
    except socket.error:
        pytest.skip("IPv6 is not supported on this system")

    ctx = zmq.Context()
    ipv6_path = "tcp://[::]:5555"  # IPv6 loopback address
    socket_type = zmq.REP  # Example socket type

    # Create the socket
    zsock: zmq.Socket = make_zmq_socket(ctx, ipv6_path, socket_type)

    # Verify that the IPV6 option is set
    assert zsock.getsockopt(
        zmq.IPV6) == 1, "IPV6 option should be enabled for IPv6 addresses"

    # Clean up
    zsock.close()
    ctx.term()


def test_make_zmq_path():
    assert make_zmq_path("tcp", "127.0.0.1", "5555") == "tcp://127.0.0.1:5555"
    assert make_zmq_path("tcp", "::1", "5555") == "tcp://[::1]:5555"


def test_get_tcp_uri():
    assert get_tcp_uri("127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert get_tcp_uri("::1", 5555) == "tcp://[::1]:5555"


def test_split_host_port():
    # valid ipv4
    assert split_host_port("127.0.0.1:5555") == ("127.0.0.1", 5555)
    # invalid ipv4
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("127.0.0.1::5555")
    with pytest.raises(ValueError):
        # tailing colon
        assert split_host_port("127.0.0.1:5555:")
    with pytest.raises(ValueError):
        # no colon
        assert split_host_port("127.0.0.15555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("127.0.0.1:5555a")

    # valid ipv6
    assert split_host_port("[::1]:5555") == ("::1", 5555)
    # invalid ipv6
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("[::1]::5555")
    with pytest.raises(IndexError):
        # no colon
        assert split_host_port("[::1]5555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("[::1]:5555a")


def test_join_host_port():
    assert join_host_port("127.0.0.1", 5555) == "127.0.0.1:5555"
    assert join_host_port("::1", 5555) == "[::1]:5555"


def test_convert_ids_list_to_tokens():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    token_ids = tokenizer.encode("Hello, world!")
    # token_ids = [9707, 11, 1879, 0]
    assert tokenizer.convert_ids_to_tokens(token_ids) == [
        'Hello', ',', 'Ġworld', '!'
    ]
    tokens = convert_ids_list_to_tokens(tokenizer, token_ids)
    assert tokens == ['Hello', ',', ' world', '!']


def test_current_stream_multithread():
    import threading
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_default_stream = torch.cuda.current_stream()
    child_stream = torch.cuda.Stream()

    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(
            timeout=5), "Child thread failed to enter stream context in time"

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, "Main thread's current_stream was contaminated by child thread"
        assert main_current_stream == main_default_stream, "Main thread's current_stream is not the default stream"

        # Notify child thread it can exit
        thread_can_exit.set()

    finally:
        # Ensure child thread exits properly
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")
