# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in snapshot preparation while the ordinary startup engine is private."""

import os
import platform
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllm.logger import init_logger
from vllm.snapshot.engine import SnapshotSession
from vllm.snapshot.file_control import FileSnapshotControl

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class StartupSnapshotConfig(BaseModel):
    """Launcher-only startup extension to the proposed ``--snapshot-config``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["startup"]
    control_dir: str
    timeout_s: float = Field(default=900, gt=0, allow_inf_nan=False)

    @field_validator("control_dir")
    @classmethod
    def absolute_control_dir(cls, value: str) -> str:
        if not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("control_dir must be an absolute path without '..'")
        return value


def validate_startup_snapshot_args(args: Namespace) -> StartupSnapshotConfig:
    """Reject unsupported dispatch paths before a launcher starts an engine."""
    from vllm import envs

    config = StartupSnapshotConfig.model_validate(args.snapshot_config)
    if (
        getattr(args, "subparser", "serve") != "serve"
        or getattr(args, "grpc", False)
        or getattr(args, "headless", False)
        or envs.VLLM_USE_RUST_FRONTEND
        or getattr(args, "api_server_count", None) not in (None, 1)
        or getattr(args, "uds", None)
    ):
        raise ValueError("startup snapshots require one Python HTTP serve frontend")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "nnodes",
    ):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"startup snapshots require {name}=1")
    if (
        getattr(args, "data_parallel_size_local", None) not in (None, 1)
        or getattr(args, "node_rank", 0) != 0
        or getattr(args, "data_parallel_backend", "mp") != "mp"
    ):
        raise ValueError("startup snapshots require local single-rank execution")
    if any(
        getattr(args, name, None)
        for name in (
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_multi_port_external_lb",
            "enable_expert_parallel",
            "enable_elastic_ep",
        )
    ) or any(
        getattr(args, name, None) is not None
        for name in ("data_parallel_rank", "data_parallel_start_rank")
    ):
        raise ValueError("startup snapshots require local single-rank execution")
    if getattr(args, "distributed_executor_backend", None) not in (None, "mp", "uni"):
        raise ValueError("startup snapshots require a local executor")
    if getattr(args, "logprobs_mode", None) in {"raw_logits", "processed_logits"}:
        raise ValueError("startup snapshots require log probabilities for validation")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("startup snapshots require Linux x86_64")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") != "spawn":
        raise ValueError("startup snapshots require spawned workers")
    return config


def validate_startup_snapshot_config(config: "VllmConfig") -> None:
    """Validate the resolved compact profile before constructing workers."""
    from vllm.platforms import current_platform

    model = config.model_config
    parallel = config.parallel_config
    if not current_platform.is_cuda():
        raise ValueError("startup snapshots require CUDA")
    if (
        model.runner_type != "generate"
        or model.is_multimodal_model
        or model.is_moe
        or model.is_hybrid
        or model.quantization is not None
    ):
        raise ValueError(
            "startup snapshots require a dense unquantized generation model"
        )
    if (
        parallel.world_size != 1
        or parallel.data_parallel_size != 1
        or parallel.data_parallel_size_local != 1
        or parallel.data_parallel_rank != 0
        or parallel.nnodes != 1
        or parallel.node_rank != 0
        or parallel.data_parallel_backend != "mp"
        or parallel._api_process_count != 1
        or parallel.distributed_executor_backend not in ("uni", "mp")
    ):
        raise ValueError("startup snapshots require local single-rank execution")
    if any(
        getattr(config, name) is not None
        for name in (
            "kv_transfer_config",
            "ec_transfer_config",
            "speculative_config",
            "lora_config",
        )
    ):
        raise ValueError(
            "startup snapshots do not support transfer, speculation or LoRA"
        )
    if (
        config.load_config.load_format not in ("auto", "safetensors")
        or config.offload_config.uva.cpu_offload_gb
        or config.offload_config.prefetch.offload_group_size
        or not model.enable_sleep_mode
    ):
        raise ValueError("startup snapshots require the compact sleep/reload policy")


async def run_startup_snapshot_server(args: Namespace, **uvicorn_kwargs) -> None:
    """Capture before public startup, then use the normal serving application."""
    from vllm.entrypoints.launchers.api_server.entry import (
        build_and_serve,
        build_async_engine_client,
    )
    from vllm.entrypoints.launchers.launcher import (
        bind_server_socket,
        prepare_server_args,
    )

    config = validate_startup_snapshot_args(args)
    prepare_server_args(args)
    control = FileSnapshotControl(Path(config.control_dir), config.timeout_s)
    args.enable_sleep_mode = True
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Match the TP1 capturer's policy before NCCL initializes in a worker.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    session = None
    sock = None
    phase = "initialize engine"
    try:
        async with build_async_engine_client(args, snapshot_startup=True) as engine:
            session = SnapshotSession(engine, timeout_s=config.timeout_s)
            try:
                phase = "prepare"
                oracle = await session.prepare()
                phase = "capture barrier"
                control.publish_ready(oracle)
                activation = await control.wait_for_activation()
                args.host, args.port = activation.host, activation.port
                phase = "recover"
                control.write_status("recovering")
                await session.recover()
                control.write_status("validated")
                phase = "serve"
                listen_address, sock = bind_server_socket(args, reuse_port=False)
                shutdown_task = await build_and_serve(
                    engine, listen_address, sock, args, **uvicorn_kwargs
                )
            except BaseException as error:
                control.write_error(
                    session.phase if session.state == "failed" else phase, error
                )
                # The builder terminates owned workers without draining when
                # this startup context exits with an error, including cancelled RPCs.
                raise
        await shutdown_task
    except BaseException as error:
        if session is None:
            control.write_error(phase, error)
        raise
    finally:
        if sock is not None:
            sock.close()
