# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import json
import os

import pytest
import yaml
from transformers import AutoTokenizer
from pydantic import ValidationError

from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens

from vllm.utils.argparse_utils import FlexibleArgumentParser
from ..utils import flat_product


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
    parser.add_argument("-cc", "--compilation-config", type=json.loads)
    parser.add_argument("--optimization-level", type=int)
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
        "-cc.use_inductor_graph_partition=true",
        "-cc.backend",
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
        "-cc.custom_ops+",
        "-quant_fp8",
        "-cc.custom_ops+=+silu_mul,-rms_norm",
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
    assert parsed_args.optimization_level == 1
    assert parsed_args.compilation_config == {
        "use_inductor_graph_partition": True,
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
        "-cc.mode",
        "2",
        "-O3",
    ]

    parsed_args = parser.parse_args(args)
    # Should be the last value
    assert parsed_args.hf_overrides == {"key1": "val2"}
    assert parsed_args.optimization_level == 3
    assert parsed_args.compilation_config == {"mode": 2}

    assert len(caplog_vllm.records) == 1
    assert "duplicate" in caplog_vllm.text
    assert "--hf-overrides.key1" in caplog_vllm.text
    assert "--optimization-level" in caplog_vllm.text


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


def test_convert_ids_list_to_tokens():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    token_ids = tokenizer.encode("Hello, world!")
    # token_ids = [9707, 11, 1879, 0]
    assert tokenizer.convert_ids_to_tokens(token_ids) == ["Hello", ",", "Ġworld", "!"]
    tokens = convert_ids_list_to_tokens(tokenizer, token_ids)
    assert tokens == ["Hello", ",", " world", "!"]


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


def test_load_config_file_nested(tmp_path):
    """Test that nested dicts in YAML config are converted to JSON strings."""
    config_data = {
        "port": 8000,
        "compilation-config": {
            "pass_config": {"fuse_allreduce_rms": True},
        },
    }
    config_file_path = tmp_path / "nested_config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(config_data, f)

    parser = FlexibleArgumentParser()
    processed_args = parser.load_config_file(str(config_file_path))

    assert processed_args[processed_args.index("--port") + 1] == "8000"
    cc_value = json.loads(
        processed_args[processed_args.index("--compilation-config") + 1]
    )
    assert cc_value == {"pass_config": {"fuse_allreduce_rms": True}}


def test_nested_config_end_to_end(tmp_path):
    """Test end-to-end parsing of nested configs in YAML files."""
    config_data = {
        "compilation-config": {
            "mode": 3,
            "pass_config": {"fuse_allreduce_rms": True},
        },
    }
    config_file_path = tmp_path / "nested_config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(config_data, f)

    parser = FlexibleArgumentParser()
    parser.add_argument("-cc", "--compilation-config", type=json.loads)
    args = parser.parse_args(["--config", str(config_file_path)])

    assert args.compilation_config == {
        "mode": 3,
        "pass_config": {"fuse_allreduce_rms": True},
    }


def test_compilation_mode_string_values(parser):
    """Test that -cc.mode accepts both integer and string mode values."""
    args = parser.parse_args(["-cc.mode", "0"])
    assert args.compilation_config == {"mode": 0}

    args = parser.parse_args(["-O3"])
    assert args.optimization_level == 3

    args = parser.parse_args(["-cc.mode=NONE"])
    assert args.compilation_config == {"mode": "NONE"}

    args = parser.parse_args(["-cc.mode", "STOCK_TORCH_COMPILE"])
    assert args.compilation_config == {"mode": "STOCK_TORCH_COMPILE"}

    args = parser.parse_args(["-cc.mode=DYNAMO_TRACE_ONCE"])
    assert args.compilation_config == {"mode": "DYNAMO_TRACE_ONCE"}

    args = parser.parse_args(["-cc.mode", "VLLM_COMPILE"])
    assert args.compilation_config == {"mode": "VLLM_COMPILE"}

    args = parser.parse_args(["-cc.mode=none"])
    assert args.compilation_config == {"mode": "none"}

    args = parser.parse_args(["-cc.mode=vllm_compile"])
    assert args.compilation_config == {"mode": "vllm_compile"}


def test_compilation_config_mode_validator():
    """Test that CompilationConfig.mode field validator converts strings to integers."""
    from vllm.config.compilation import CompilationConfig, CompilationMode

    config = CompilationConfig(mode=0)
    assert config.mode == CompilationMode.NONE

    config = CompilationConfig(mode=3)
    assert config.mode == CompilationMode.VLLM_COMPILE

    config = CompilationConfig(mode="NONE")
    assert config.mode == CompilationMode.NONE

    config = CompilationConfig(mode="STOCK_TORCH_COMPILE")
    assert config.mode == CompilationMode.STOCK_TORCH_COMPILE

    config = CompilationConfig(mode="DYNAMO_TRACE_ONCE")
    assert config.mode == CompilationMode.DYNAMO_TRACE_ONCE

    config = CompilationConfig(mode="VLLM_COMPILE")
    assert config.mode == CompilationMode.VLLM_COMPILE

    config = CompilationConfig(mode="none")
    assert config.mode == CompilationMode.NONE

    config = CompilationConfig(mode="vllm_compile")
    assert config.mode == CompilationMode.VLLM_COMPILE

    with pytest.raises(ValidationError, match="Invalid compilation mode"):
        CompilationConfig(mode="INVALID_MODE")


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
