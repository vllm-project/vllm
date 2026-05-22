# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization entry point for the unified communication layer.

Called from vllm-hust startup code to register backends and prepare
the communication groups.
"""

from __future__ import annotations

import os

import torch

from vllm.distributed.unified_comm.backend import register_backend
from vllm.distributed.unified_comm.strategy import (
    CommStrategy,
    DefaultStrategy,
)
from vllm.distributed.unified_comm.transfer_plane import (
    ECTransferPlane,
    KVTransferPlane,
    TransferPlaneRegistry,
    TransferProtocol,
    WeightTransferPlane,
)


def initialize_unified_comm(
    device_type: str | None = None,
    strategy: CommStrategy | None = None,
) -> None:
    """
    初始化统一通信层。

    自动检测硬件环境，注册对应后端，初始化传输平面。

    Args:
        device_type: 强制指定设备类型。None 则自动检测。
        strategy: 自定义策略。None 则使用 DefaultStrategy。

    Usage:
        # 自动检测环境
        initialize_unified_comm()

        # 强制使用 HCCL
        initialize_unified_comm(device_type="npu")

        # 使用自定义策略
        initialize_unified_comm(strategy=ConfigDrivenStrategy(my_config))
    """
    # 1. 自动检测设备类型
    if device_type is None:
        device_type = _detect_device_type()

    # 2. 注册后端
    _register_backends(device_type)

    # 3. 注册传输平面
    _register_transfer_planes()

    # 4. 设置默认策略
    global _global_strategy
    _global_strategy = strategy or DefaultStrategy()


def _detect_device_type() -> str:
    """自动检测当前可用的设备类型"""
    # 优先检测 NPU
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass

    # 检测 CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Fallback: CPU
    return "cpu"


def _register_backends(device_type: str) -> None:
    """根据设备类型注册对应后端"""
    if device_type == "npu":
        try:
            from vllm.distributed.unified_comm.backends.hccl_backend import (
                HCCLBackend,
            )

            # 通过环境变量 UNIFIED_COMM_USE_DIRECT_HCCL=1 启用 Mode B
            # (直接调 libhccl.so C API, 低延迟; 默认 Mode A 走 torch.distributed)
            use_direct = os.environ.get(
                "UNIFIED_COMM_USE_DIRECT_HCCL", "0"
            ).lower() in ("1", "true", "yes")
            register_backend(HCCLBackend(use_direct_hccl=use_direct))
        except ImportError:
            pass

    elif device_type == "cuda":
        from vllm.distributed.unified_comm.backends.nccl_backend import (
            NCCLBackend,
        )

        use_direct = _should_use_direct_nccl()
        register_backend(NCCLBackend(use_direct_nccl=use_direct))

    # 始终注册 Gloo 作为 CPU 通信的 fallback（如需要可在此扩展）


def _should_use_direct_nccl() -> bool:
    """
    判断是否应该使用直接 NCCL C API 模式。

    通过环境变量 UNIFIED_COMM_USE_DIRECT_NCCL 控制：
      - "1" / "true": 启用直接模式 (Mode B)
      - "0" / "false" / 未设置: 使用 torch.distributed 模式 (Mode A)
    """
    val = os.environ.get("UNIFIED_COMM_USE_DIRECT_NCCL", "0").lower()
    return val in ("1", "true", "yes")


def _register_transfer_planes() -> None:
    """注册默认传输平面"""
    registry = TransferPlaneRegistry()
    registry.register(KVTransferPlane(preferred_protocol=TransferProtocol.P2P))
    registry.register(
        WeightTransferPlane(preferred_protocol=TransferProtocol.COLLECTIVE)
    )
    registry.register(ECTransferPlane(preferred_protocol=TransferProtocol.COLLECTIVE))


# 全局策略实例
_global_strategy: CommStrategy = DefaultStrategy()


def get_strategy() -> CommStrategy:
    """获取全局通信策略"""
    return _global_strategy


def set_strategy(strategy: CommStrategy) -> None:
    """设置全局通信策略"""
    global _global_strategy
    _global_strategy = strategy
