# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build reproducible vLLM launch plans for artifact-transfer benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_MODES = ("none", "traditional_api", "direct_worker")


@dataclass(frozen=True)
class BenchmarkServerConfig:
    mode: str
    model_path: str
    served_model_name: str
    port: int
    run_id: str
    policy_version: str
    transfer_queue_config_path: str | None = None
    host: str = "0.0.0.0"
    device: str | None = None
    gpu_id: int | None = None
    artifact_metrics_path: str | None = None
    publish_mode: str = "sync"
    publish_queue_maxsize: int = 4096
    publish_batch_size: int = 8
    publish_flush_interval_ms: float = 2.0
    publish_drain_on_shutdown: bool = True
    max_model_len: int = 128
    gpu_memory_utilization: float = 0.1
    enforce_eager: bool = True
    venv_path: str = "/mnt/data1/yibo/vllm-workspace/.venv"
    use_flashinfer_sampler: bool = False
    server_dev_mode: bool = True
    enable_weight_transfer: bool = False
    extra_vllm_args: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.mode not in BENCHMARK_MODES:
            raise ValueError(f"unsupported benchmark server mode: {self.mode}")
        if self.port <= 0:
            raise ValueError("--port must be positive")
        if self.max_model_len <= 0:
            raise ValueError("--max-model-len must be positive")
        if not (0 < self.gpu_memory_utilization <= 1):
            raise ValueError("--gpu-memory-utilization must be in (0, 1]")
        if self.publish_mode not in ("sync", "async"):
            if self.publish_mode != "async_batch":
                raise ValueError("--publish-mode must be sync, async, or async_batch")
        if self.publish_queue_maxsize <= 0:
            raise ValueError("--publish-queue-maxsize must be positive")
        if self.publish_batch_size <= 0:
            raise ValueError("--publish-batch-size must be positive")
        if self.publish_flush_interval_ms < 0:
            raise ValueError("--publish-flush-interval-ms must be non-negative")
        if self.mode == "direct_worker" and not self.transfer_queue_config_path:
            raise ValueError("direct_worker mode requires --transfer-queue-config-path")


def artifact_transfer_config(config: BenchmarkServerConfig) -> dict[str, Any] | None:
    if config.mode != "direct_worker":
        return None
    assert config.transfer_queue_config_path is not None
    artifact_config = {
        "artifact_connector": "TransferQueueArtifactConnector",
        "artifact_connector_module_path": (
            "vllm.distributed.artifact_transfer.artifact_connector.v1."
            "transfer_queue_connector"
        ),
        "artifact_role": "artifact_producer",
        "transfer_mode": "final",
        "export_fields": [
            "prompt_token_ids",
            "response_token_ids",
            "response_logprobs",
        ],
        "failure_policy": "fail_request",
        "artifact_connector_extra_config": {
            "transfer_queue_config_path": config.transfer_queue_config_path,
            "run_id": config.run_id,
            "policy_version": config.policy_version,
            "model_id": config.served_model_name,
            "publish_mode": config.publish_mode,
            "publish_queue_maxsize": config.publish_queue_maxsize,
            "publish_batch_size": config.publish_batch_size,
            "publish_flush_interval_ms": config.publish_flush_interval_ms,
            "publish_drain_on_shutdown": config.publish_drain_on_shutdown,
        },
    }
    if config.artifact_metrics_path is not None:
        artifact_config["artifact_connector_extra_config"]["artifact_metrics_path"] = (
            config.artifact_metrics_path
        )
    return artifact_config


def environment(config: BenchmarkServerConfig) -> dict[str, str]:
    env = {
        "PATH": f"{config.venv_path}/bin:$PATH",
        "VLLM_SERVER_DEV_MODE": "1" if config.server_dev_mode else "0",
        "VLLM_USE_FLASHINFER_SAMPLER": ("1" if config.use_flashinfer_sampler else "0"),
    }
    if config.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    return env


def vllm_serve_argv(config: BenchmarkServerConfig) -> list[str]:
    argv = [
        "vllm",
        "serve",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--served-model-name",
        config.served_model_name,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
    ]
    if config.device is not None:
        argv.extend(["--device", config.device])
    if config.enforce_eager:
        argv.append("--enforce-eager")
    if config.enable_weight_transfer:
        argv.extend(["--weight-transfer-config", json.dumps({"backend": "nccl"})])
    artifact_config = artifact_transfer_config(config)
    if artifact_config is not None:
        argv.extend(["--artifact-transfer-config", json.dumps(artifact_config)])
    argv.extend(config.extra_vllm_args)
    return argv


def shell_command(config: BenchmarkServerConfig) -> str:
    exports = [f"export {name}={value}" for name, value in environment(config).items()]
    return "\n".join(exports + [shlex.join(vllm_serve_argv(config))])


def benchmark_client_hint(config: BenchmarkServerConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "endpoint": f"http://172.16.1.247:{config.port}/v1/completions",
        "model": config.served_model_name,
        "run_id": config.run_id,
        "policy_version": config.policy_version,
        "service_config_path": (
            config.transfer_queue_config_path
            if config.mode == "traditional_api"
            else None
        ),
    }


def build_launch_plan(config: BenchmarkServerConfig) -> dict[str, Any]:
    config.validate()
    return {
        "config": asdict(config),
        "environment": environment(config),
        "artifact_transfer_config": artifact_transfer_config(config),
        "argv": vllm_serve_argv(config),
        "shell_command": shell_command(config),
        "benchmark_client_hint": benchmark_client_hint(config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=BENCHMARK_MODES)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--transfer-queue-config-path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--artifact-metrics-path")
    parser.add_argument(
        "--publish-mode",
        choices=("sync", "async", "async_batch"),
        default="sync",
    )
    parser.add_argument("--publish-queue-maxsize", type=int, default=4096)
    parser.add_argument("--publish-batch-size", type=int, default=8)
    parser.add_argument(
        "--publish-flush-interval-ms",
        type=float,
        default=2.0,
    )
    parser.add_argument("--no-publish-drain-on-shutdown", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument(
        "--venv-path",
        default="/mnt/data1/yibo/vllm-workspace/.venv",
    )
    parser.add_argument("--use-flashinfer-sampler", action="store_true")
    parser.add_argument("--no-server-dev-mode", action="store_true")
    parser.add_argument("--enable-weight-transfer", action="store_true")
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    parser.add_argument(
        "--format",
        choices=("json", "shell"),
        default="json",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BenchmarkServerConfig:
    return BenchmarkServerConfig(
        mode=args.mode,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        port=args.port,
        run_id=args.run_id,
        policy_version=args.policy_version,
        transfer_queue_config_path=args.transfer_queue_config_path,
        host=args.host,
        device=args.device,
        gpu_id=args.gpu_id,
        artifact_metrics_path=args.artifact_metrics_path,
        publish_mode=args.publish_mode,
        publish_queue_maxsize=args.publish_queue_maxsize,
        publish_batch_size=args.publish_batch_size,
        publish_flush_interval_ms=args.publish_flush_interval_ms,
        publish_drain_on_shutdown=not args.no_publish_drain_on_shutdown,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        venv_path=args.venv_path,
        use_flashinfer_sampler=args.use_flashinfer_sampler,
        server_dev_mode=not args.no_server_dev_mode,
        enable_weight_transfer=args.enable_weight_transfer,
        extra_vllm_args=list(args.extra_vllm_arg),
    )


def render_plan(plan: dict[str, Any], output_format: str) -> str:
    if output_format == "shell":
        return str(plan["shell_command"]) + "\n"
    return json.dumps(plan, indent=2, sort_keys=True) + "\n"


def main() -> None:
    args = parse_args()
    plan = build_launch_plan(config_from_args(args))
    rendered = render_plan(plan, args.format)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
