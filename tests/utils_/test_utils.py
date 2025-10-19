# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from transformers import AutoTokenizer
from vllm_test_utils.monitor import monitor

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.transformers_utils.detokenizer_utils import convert_ids_list_to_tokens

from vllm.utils import (
    FlexibleArgumentParser,
    bind_kv_cache,
    unique_filepath,
)
from vllm.utils.hashing import sha256
from vllm.utils.torch_utils import (
    common_broadcastable_dtype,
    current_stream,
    is_lossless_cast,
)
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from ..utils import create_new_process_for_each_test, flat_product


# Tests for FlexibleArgumentParser
@pytest.fixture
def parser():
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--image-input-type", choices=["pixel_values", "image_features"]
    )
    parser.add_argument("--model-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--enable-feature", action="store_true")
    parser.add_argument("--hf-overrides", type=json.loads)
    parser.add_argument("-O", "--compilation-config", type=json.loads)
    return parser


@pytest.fixture
def parser_with_config():
    parser = FlexibleArgumentParser()
    parser.add_argument("serve")
    parser.add_argument("model_tag", nargs="?")
    parser.add_argument("--model", type=str)
    parser.add_argument("--served-model-name", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def test_underscore_to_dash(parser):
    args = parser.parse_args(["--image_input_type", "pixel_values"])
    assert args.image_input_type == "pixel_values"


def test_mixed_usage(parser):
    args = parser.parse_args(
        ["--image_input_type", "image_features", "--model-name", "facebook/opt-125m"]
    )
    assert args.image_input_type == "image_features"
    assert args.model_name == "facebook/opt-125m"


def test_with_equals_sign(parser):
    args = parser.parse_args(
        ["--image_input_type=pixel_values", "--model-name=facebook/opt-125m"]
    )
    assert args.image_input_type == "pixel_values"
    assert args.model_name == "facebook/opt-125m"


def test_with_int_value(parser):
    args = parser.parse_args(["--batch_size", "32"])
    assert args.batch_size == 32
    args = parser.parse_args(["--batch-size", "32"])
    assert args.batch_size == 32


def test_with_bool_flag(parser):
    args = parser.parse_args(["--enable_feature"])
    assert args.enable_feature is True
    args = parser.parse_args(["--enable-feature"])
    assert args.enable_feature is True


def test_invalid_choice(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--image_input_type", "invalid_choice"])


def test_missing_required_argument(parser):
    parser.add_argument("--required-arg", required=True)
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_override_to_config(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args(
        ["serve", "mymodel", "--config", cli_config_file, "--tensor-parallel-size", "3"]
    )
    assert args.tensor_parallel_size == 3
    args = parser_with_config.parse_args(
        ["serve", "mymodel", "--tensor-parallel-size", "3", "--config", cli_config_file]
    )
    assert args.tensor_parallel_size == 3
    assert args.port == 12312
    args = parser_with_config.parse_args(
        [
            "serve",
            "mymodel",
            "--tensor-parallel-size",
            "3",
            "--config",
            cli_config_file,
            "--port",
            "666",
        ]
    )
    assert args.tensor_parallel_size == 3
    assert args.port == 666


def test_config_args(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args(
        ["serve", "mymodel", "--config", cli_config_file]
    )
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code


def test_config_file(parser_with_config):
    with pytest.raises(FileNotFoundError):
        parser_with_config.parse_args(
            ["serve", "mymodel", "--config", "test_config.yml"]
        )

    with pytest.raises(ValueError):
        parser_with_config.parse_args(
            ["serve", "mymodel", "--config", "./data/test_config.json"]
        )

    with pytest.raises(ValueError):
        parser_with_config.parse_args(
            [
                "serve",
                "mymodel",
                "--tensor-parallel-size",
                "3",
                "--config",
                "--batch-size",
                "32",
            ]
        )


def test_no_model_tag(parser_with_config, cli_config_file):
    with pytest.raises(ValueError):
        parser_with_config.parse_args(["serve", "--config", cli_config_file])


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
        # Test compile config and compilation mode
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
        },
    }
    assert parsed_args.compilation_config == {
        "mode": 1,
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
        "-O.mode",
        "2",
        "-O3",
    ]

    parsed_args = parser.parse_args(args)
    # Should be the last value
    assert parsed_args.hf_overrides == {"key1": "val2"}
    assert parsed_args.compilation_config == {"mode": 3}

    assert len(caplog_vllm.records) == 1
    assert "duplicate" in caplog_vllm.text
    assert "--hf-overrides.key1" in caplog_vllm.text
    assert "-O.mode" in caplog_vllm.text


@create_new_process_for_each_test()
def test_memory_profiling():
    # Fake out some model loading + inference memory usage to test profiling
    # Memory used by other processes will show up as cuda usage outside of torch
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    # 512 MiB allocation outside of this instance
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)

    baseline_snapshot = MemorySnapshot()

    # load weights

    weights = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)

    weights_memory = 128 * 1024 * 1024 * 4  # 512 MiB

    def measure_current_non_torch():
        free, total = torch.cuda.mem_get_info()
        current_used = total - free
        current_torch = torch.cuda.memory_reserved()
        current_non_torch = current_used - current_torch
        return current_non_torch

    with (
        memory_profiling(
            baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
        ) as result,
        monitor(measure_current_non_torch) as monitored_values,
    ):
        # make a memory spike, 1 GiB
        spike = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float32)
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
    non_torch_ratio = result.non_torch_increase / (256 * 1024 * 1024)  # noqa
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024
    del weights
    lib.cudaFree(handle1)
    lib.cudaFree(handle2)


def test_bind_kv_cache():
    from vllm.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1),
        "layers.1.self_attn": Attention(32, 128, 0.1),
        "layers.2.self_attn": Attention(32, 128, 0.1),
        "layers.3.self_attn": Attention(32, 128, 0.1),
    }
    kv_cache = [
        torch.zeros((1,)),
        torch.zeros((1,)),
        torch.zeros((1,)),
        torch.zeros((1,)),
    ]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx["layers.0.self_attn"].kv_cache[0] is kv_cache[0]
    assert ctx["layers.1.self_attn"].kv_cache[0] is kv_cache[1]
    assert ctx["layers.2.self_attn"].kv_cache[0] is kv_cache[2]
    assert ctx["layers.3.self_attn"].kv_cache[0] is kv_cache[3]


def test_bind_kv_cache_kv_sharing():
    from vllm.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1),
        "layers.1.self_attn": Attention(32, 128, 0.1),
        "layers.2.self_attn": Attention(32, 128, 0.1),
        "layers.3.self_attn": Attention(32, 128, 0.1),
    }
    kv_cache = [
        torch.zeros((1,)),
        torch.zeros((1,)),
        torch.zeros((1,)),
        torch.zeros((1,)),
    ]
    shared_kv_cache_layers = {
        "layers.2.self_attn": "layers.1.self_attn",
        "layers.3.self_attn": "layers.0.self_attn",
    }
    bind_kv_cache(ctx, [kv_cache], shared_kv_cache_layers)
    assert ctx["layers.0.self_attn"].kv_cache[0] is kv_cache[0]
    assert ctx["layers.1.self_attn"].kv_cache[0] is kv_cache[1]
    assert ctx["layers.2.self_attn"].kv_cache[0] is kv_cache[1]
    assert ctx["layers.3.self_attn"].kv_cache[0] is kv_cache[0]


def test_bind_kv_cache_non_attention():
    from vllm.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1),
        "model.layers.28.attn": Attention(32, 128, 0.1),
    }
    kv_cache = [
        torch.zeros((1,)),
        torch.zeros((1,)),
    ]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx["model.layers.20.attn"].kv_cache[0] is kv_cache[0]
    assert ctx["model.layers.28.attn"].kv_cache[0] is kv_cache[1]


def test_bind_kv_cache_pp():
    with patch("vllm.utils.torch_utils.cuda_device_count_stateless", lambda: 2):
        # this test runs with 1 GPU, but we simulate 2 GPUs
        cfg = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=2))
    with set_current_vllm_config(cfg):
        from vllm.attention import Attention

        ctx = {
            "layers.0.self_attn": Attention(32, 128, 0.1),
        }
        kv_cache = [[torch.zeros((1,))], [torch.zeros((1,))]]
        bind_kv_cache(ctx, kv_cache)
        assert ctx["layers.0.self_attn"].kv_cache[0] is kv_cache[0][0]
        assert ctx["layers.0.self_attn"].kv_cache[1] is kv_cache[1][0]


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
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def test_model_specification(
    parser_with_config, cli_config_file, cli_config_file_with_model
):
    # Test model in CLI takes precedence over config
    args = parser_with_config.parse_args(
        ["serve", "cli-model", "--config", cli_config_file_with_model]
    )
    assert args.model_tag == "cli-model"
    assert args.served_model_name == "mymodel"

    # Test model from config file works
    args = parser_with_config.parse_args(
        [
            "serve",
            "--config",
            cli_config_file_with_model,
        ]
    )
    assert args.model == "config-model"
    assert args.served_model_name == "mymodel"

    # Test no model specified anywhere raises error
    with pytest.raises(ValueError, match="No model specified!"):
        parser_with_config.parse_args(["serve", "--config", cli_config_file])

    # Test using --model option raises error
    # with pytest.raises(
    #         ValueError,
    #         match=
    #     ("With `vllm serve`, you should provide the model as a positional "
    #      "argument or in a config file instead of via the `--model` option."),
    # ):
    #     parser_with_config.parse_args(['serve', '--model', 'my-model'])

    # Test using --model option back-compatibility
    # (when back-compatibility ends, the above test should be uncommented
    # and the below test should be removed)
    args = parser_with_config.parse_args(
        [
            "serve",
            "--tensor-parallel-size",
            "2",
            "--model",
            "my-model",
            "--trust-remote-code",
            "--port",
            "8001",
        ]
    )
    assert args.model is None
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code is True
    assert args.port == 8001

    args = parser_with_config.parse_args(
        [
            "serve",
            "--tensor-parallel-size=2",
            "--model=my-model",
            "--trust-remote-code",
            "--port=8001",
        ]
    )
    assert args.model is None
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code is True
    assert args.port == 8001

    # Test other config values are preserved
    args = parser_with_config.parse_args(
        [
            "serve",
            "cli-model",
            "--config",
            cli_config_file_with_model,
        ]
    )
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code is True
    assert args.port == 12312


@pytest.mark.parametrize("input", [(), ("abc",), (None,), (None, bool, [1, 2, 3])])
def test_sha256(input: tuple):
    digest = sha256(input)
    assert digest is not None
    assert isinstance(digest, bytes)
    assert digest != b""

    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    assert digest == hashlib.sha256(input_bytes).digest()

    # hashing again, returns the same value
    assert digest == sha256(input)

    # hashing different input, returns different value
    assert digest != sha256(input + (1,))


def test_convert_ids_list_to_tokens():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    token_ids = tokenizer.encode("Hello, world!")
    # token_ids = [9707, 11, 1879, 0]
    assert tokenizer.convert_ids_to_tokens(token_ids) == ["Hello", ",", "Ġworld", "!"]
    tokens = convert_ids_list_to_tokens(tokenizer, token_ids)
    assert tokens == ["Hello", ",", " world", "!"]


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
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_default_stream, (
            "Main thread's current_stream is not the default stream"
        )

        # Notify child thread it can exit
        thread_can_exit.set()

    finally:
        # Ensure child thread exits properly
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_load_config_file(tmp_path):
    # Define the configuration data
    config_data = {
        "enable-logging": True,
        "list-arg": ["item1", "item2"],
        "port": 12323,
        "tensor-parallel-size": 4,
    }

    # Write the configuration data to a temporary YAML file
    config_file_path = tmp_path / "config.yaml"
    with open(config_file_path, "w") as config_file:
        yaml.dump(config_data, config_file)

    # Initialize the parser
    parser = FlexibleArgumentParser()

    # Call the function with the temporary file path
    processed_args = parser.load_config_file(str(config_file_path))

    # Expected output
    expected_args = [
        "--enable-logging",
        "--list-arg",
        "item1",
        "item2",
        "--port",
        "12323",
        "--tensor-parallel-size",
        "4",
    ]

    # Assert that the processed arguments match the expected output
    assert processed_args == expected_args
    os.remove(str(config_file_path))


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_flat_product():
    # Check regular itertools.product behavior
    result1 = list(flat_product([1, 2, 3], ["a", "b"]))
    assert result1 == [
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
        (3, "a"),
        (3, "b"),
    ]

    # check that the tuples get flattened
    result2 = list(flat_product([(1, 2), (3, 4)], ["a", "b"], [(5, 6)]))
    assert result2 == [
        (1, 2, "a", 5, 6),
        (1, 2, "b", 5, 6),
        (3, 4, "a", 5, 6),
        (3, 4, "b", 5, 6),
    ]
