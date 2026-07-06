# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from tools.artifact_transfer.benchmark_server_config import (
    BenchmarkServerConfig,
    artifact_transfer_config,
    benchmark_client_hint,
    build_launch_plan,
    render_plan,
    vllm_serve_argv,
)


def test_none_mode_omits_artifact_config():
    config = BenchmarkServerConfig(
        mode="none",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18010,
        run_id="bench",
        policy_version="p0",
    )
    plan = build_launch_plan(config)
    assert plan["artifact_transfer_config"] is None
    assert "--artifact-transfer-config" not in plan["argv"]
    assert plan["benchmark_client_hint"]["mode"] == "none"


def test_traditional_api_mode_omits_server_artifact_config():
    config = BenchmarkServerConfig(
        mode="traditional_api",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18011,
        run_id="bench",
        policy_version="p0",
        transfer_queue_config_path="/services/tq/client_config.pkl",
    )
    assert artifact_transfer_config(config) is None
    hint = benchmark_client_hint(config)
    assert hint["mode"] == "traditional_api"
    assert hint["service_config_path"] == "/services/tq/client_config.pkl"


def test_direct_worker_mode_builds_transfer_queue_connector_config():
    config = BenchmarkServerConfig(
        mode="direct_worker",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18012,
        run_id="bench",
        policy_version="p1",
        transfer_queue_config_path="/services/tq/client_config.pkl",
        gpu_id=2,
    )
    artifact_config = artifact_transfer_config(config)
    assert artifact_config is not None
    assert artifact_config["artifact_connector"] == "TransferQueueArtifactConnector"
    assert artifact_config["transfer_mode"] == "final"
    assert artifact_config["export_fields"] == [
        "prompt_token_ids",
        "response_token_ids",
        "response_logprobs",
    ]
    assert artifact_config["artifact_connector_extra_config"] == {
        "transfer_queue_config_path": "/services/tq/client_config.pkl",
        "run_id": "bench",
        "policy_version": "p1",
        "model_id": "tiny",
        "publish_mode": "sync",
        "publish_queue_maxsize": 4096,
        "publish_batch_size": 8,
        "publish_flush_interval_ms": 2.0,
        "publish_drain_on_shutdown": True,
    }

    plan = build_launch_plan(config)
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "--artifact-transfer-config" in plan["argv"]
    argv_json = plan["argv"][plan["argv"].index("--artifact-transfer-config") + 1]
    assert json.loads(argv_json) == artifact_config


def test_direct_worker_requires_transfer_queue_config_path():
    config = BenchmarkServerConfig(
        mode="direct_worker",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18012,
        run_id="bench",
        policy_version="p1",
    )
    with pytest.raises(ValueError, match="transfer-queue-config-path"):
        build_launch_plan(config)


def test_shell_render_includes_exports_and_quoted_json():
    config = BenchmarkServerConfig(
        mode="none",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18010,
        run_id="bench",
        policy_version="p0",
        extra_vllm_args=["--disable-log-requests"],
    )
    shell = render_plan(build_launch_plan(config), "shell")
    assert "export VLLM_SERVER_DEV_MODE=1" in shell
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in shell
    assert "vllm serve /models/tiny" in shell
    assert "--disable-log-requests" in shell


def test_device_argument_is_optional():
    config = BenchmarkServerConfig(
        mode="none",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18010,
        run_id="bench",
        policy_version="p0",
    )
    assert "--device" not in vllm_serve_argv(config)
    assert "--device" in vllm_serve_argv(
        BenchmarkServerConfig(**{**config.__dict__, "device": "cuda"})
    )


def test_direct_worker_can_enable_debug_metrics_path():
    config = BenchmarkServerConfig(
        mode="direct_worker",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18012,
        run_id="bench",
        policy_version="p1",
        transfer_queue_config_path="/services/tq/client_config.pkl",
        artifact_metrics_path="/tmp/artifact_metrics.jsonl",
    )

    artifact_config = artifact_transfer_config(config)

    assert artifact_config is not None
    assert (
        artifact_config["artifact_connector_extra_config"]["artifact_metrics_path"]
        == "/tmp/artifact_metrics.jsonl"
    )


def test_direct_worker_can_enable_async_publish_mode():
    config = BenchmarkServerConfig(
        mode="direct_worker",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18012,
        run_id="bench",
        policy_version="p1",
        transfer_queue_config_path="/services/tq/client_config.pkl",
        publish_mode="async",
        publish_queue_maxsize=123,
        publish_drain_on_shutdown=False,
    )

    artifact_config = artifact_transfer_config(config)

    assert artifact_config is not None
    assert artifact_config["artifact_connector_extra_config"]["publish_mode"] == "async"
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_queue_maxsize"]
        == 123
    )
    assert artifact_config["artifact_connector_extra_config"]["publish_batch_size"] == 8
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_flush_interval_ms"]
        == 2.0
    )
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_drain_on_shutdown"]
        is False
    )


def test_direct_worker_can_enable_async_batch_publish_mode():
    config = BenchmarkServerConfig(
        mode="direct_worker",
        model_path="/models/tiny",
        served_model_name="tiny",
        port=18012,
        run_id="bench",
        policy_version="p1",
        transfer_queue_config_path="/services/tq/client_config.pkl",
        publish_mode="async_batch",
        publish_batch_size=16,
        publish_flush_interval_ms=5.5,
        publish_drain_on_shutdown=False,
    )

    artifact_config = artifact_transfer_config(config)

    assert artifact_config is not None
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_mode"]
        == "async_batch"
    )
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_batch_size"] == 16
    )
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_flush_interval_ms"]
        == 5.5
    )
    assert (
        artifact_config["artifact_connector_extra_config"]["publish_drain_on_shutdown"]
        is False
    )
