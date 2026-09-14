# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json

import pytest

from vllm.entrypoints.openai.responses.store import (
    ResponsesStoreConfig,
    add_responses_store_cli_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def parser():
    return add_responses_store_cli_args(FlexibleArgumentParser())


def test_defaults(parser):
    config = ResponsesStoreConfig.from_cli_args(parser.parse_args([]))
    assert not config.enabled
    assert config.disk_enabled
    assert config.memory_capacity_bytes == 512 * 1024 * 1024
    assert config.disk_capacity_bytes == 4096 * 1024 * 1024
    assert config.memory_ttl_seconds == 300
    assert config.disk_ttl_seconds == 3600


@pytest.mark.parametrize(
    "cli_args",
    [
        [
            "--responses-store-config",
            (
                '{"enabled": true, "disk_enabled": false, '
                '"memory_capacity_mb": 2, "memory_ttl_seconds": 0, '
                '"cleanup_max_bytes_mb": 1, "cleanup_interval_seconds": 1}'
            ),
        ],
        [
            "--responses-store-config.enabled",
            "true",
            "--responses-store-config.disk_enabled",
            "false",
            "--responses-store-config.memory_capacity_mb",
            "2",
            "--responses-store-config.memory_ttl_seconds",
            "0",
            "--responses-store-config.cleanup_max_bytes_mb",
            "1",
            "--responses-store-config.cleanup_interval_seconds",
            "1",
        ],
    ],
)
def test_json_and_dotted_options(parser, cli_args):
    config = ResponsesStoreConfig.from_cli_args(parser.parse_args(cli_args))
    assert config.enabled
    assert not config.disk_enabled
    assert config.memory_capacity_bytes == 2 * 1024 * 1024
    assert config.memory_low_watermark_bytes == int(2 * 1024 * 1024 * 0.6)
    assert config.memory_high_watermark_bytes == int(2 * 1024 * 1024 * 0.8)
    assert config.memory_ttl_seconds is None
    assert config.cleanup_max_bytes == 1024 * 1024
    assert config.cleanup_interval_seconds == 1


def test_yaml_options(parser, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "responses-store-config:\n"
        "  enabled: true\n"
        "  disk_enabled: false\n"
        "  num_shards: 8\n"
    )
    args = parser.parse_args(["--config", str(path)])
    config = ResponsesStoreConfig.from_cli_args(args)
    assert config.enabled
    assert not config.disk_enabled
    assert config.num_shards == 8


@pytest.mark.parametrize(
    "settings, message",
    [
        ([], "JSON object"),
        (False, "JSON object"),
        ("", "JSON object"),
        ({"typo": 1}, "unknown"),
        ({"disk_path": ":memory:"}, "unknown"),
        ({"key_file": "key"}, "unknown"),
        ({"enabled": "false"}, "must be bool"),
        ({"disk_enabled": 1}, "must be bool"),
        ({"num_shards": True}, "must be int"),
        ({"memory_capacity_mb": 1.5}, "must be int"),
        ({"memory_ttl_seconds": None}, "must be int"),
        ({"cleanup_interval_seconds": "1"}, "must be float"),
        ({"cleanup_interval_seconds": float("inf")}, "finite"),
        ({"memory_high_watermark": float("nan")}, "finite"),
        ({"memory_capacity_mb": 0}, "capacities"),
        ({"disk_capacity_mb": -1}, "capacities"),
        ({"memory_low_watermark": 0.8}, "watermarks"),
        ({"disk_high_watermark": 1.1}, "watermarks"),
        ({"memory_ttl_seconds": -1}, "TTL"),
        ({"disk_ttl_seconds": -1}, "TTL"),
        ({"cleanup_interval_seconds": 0}, "cleanup interval"),
        ({"cleanup_max_candidates": 0}, "cleanup max candidates"),
        ({"cleanup_max_bytes_mb": 0}, "cleanup max bytes"),
        ({"num_shards": 0}, "num shards"),
        ({"disk_write_interval_seconds": 0}, "disk write interval"),
    ],
)
def test_invalid_config(parser, settings, message):
    args = parser.parse_args(["--responses-store-config", json.dumps(settings)])
    with pytest.raises(ValueError, match=message):
        ResponsesStoreConfig.from_cli_args(args)


@pytest.mark.parametrize(
    "settings, disk_path, message",
    [
        ({"enabled": True, "disk_enabled": False}, "store.db", "disk tier"),
        ({"enabled": True}, None, "explicit disk path"),
        ({"enabled": True}, ":memory:", "persistent disk path"),
    ],
)
def test_key_file_constraints(settings, disk_path, message):
    args = argparse.Namespace(
        responses_store_config=settings,
        responses_store_disk_path=disk_path,
        responses_store_key_file="store.key",
    )
    with pytest.raises(ValueError, match=message):
        ResponsesStoreConfig.from_cli_args(args)


def test_separate_path_options(parser):
    args = parser.parse_args(
        [
            "--responses-store-config",
            '{"enabled": true}',
            "--responses-store-disk-path",
            "responses.sqlite3",
            "--responses-store-key-file",
            "responses.key",
        ]
    )
    config = ResponsesStoreConfig.from_cli_args(args)
    assert config.enabled
    assert config.disk_path == "responses.sqlite3"
    assert config.key_file == "responses.key"


def test_invalid_json(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--responses-store-config", "{invalid}"])


def test_old_flags_are_not_registered(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--enable-responses-store"])
