# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer 1: ``CommBackend`` - device-level communication backend.

Hides the differences between underlying communication libraries
(NCCL / HCCL / MetaX CCL) and exposes a uniform set of low-level
collective primitives.

Design principles:
  - Interfaces are declared as ABCs.
  - Backends are pluggable via a registry.
  - A backend implementation only needs to know *how* to communicate;
    *when* to communicate is decided by the layers above.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch
import torch.distributed as dist

# ============================================================
# 基础数据类型定义
# ============================================================


class ReduceOp(Enum):
    """归约操作类型，设备无关"""

    SUM = auto()
    PRODUCT = auto()
    MAX = auto()
    MIN = auto()
    AVG = auto()

    def to_torch(self) -> dist.ReduceOp:
        """转为 torch 原生 ReduceOp"""
        # OPT #3: 模块级 mapping, 避免每次调用重建 dict
        try:
            return _REDUCE_OP_TO_TORCH[self]
        except KeyError:
            if self == ReduceOp.AVG:
                raise ValueError(
                    "AVG not directly supported by torch ReduceOp, use SUM + divide"
                ) from None
            raise


# OPT #3: 模块级 mapping 表 (放在 ReduceOp 之后, 因为依赖类成员)
_REDUCE_OP_TO_TORCH: dict[ReduceOp, dist.ReduceOp] = {
    ReduceOp.SUM: dist.ReduceOp.SUM,
    ReduceOp.PRODUCT: dist.ReduceOp.PRODUCT,
    ReduceOp.MAX: dist.ReduceOp.MAX,
    ReduceOp.MIN: dist.ReduceOp.MIN,
}


@dataclass
class CommConfig:
    """通信后端配置"""

    library_path: str | None = None  # 通信库路径（如 libhccl.so）
    enable_high_priority_stream: bool = False  # 是否使用高优先级流
    timeout_seconds: float = 300.0  # 超时时间
    extra: dict[str, Any] = field(default_factory=dict)  # 后端特有配置


@dataclass
class CommGroupInfo:
    """通信组描述"""

    ranks: list[int]  # 组内全局 rank 列表
    rank_in_group: int  # 当前进程在组内的 rank
    world_size: int  # 组大小
    device: torch.device  # 绑定设备
    backend_name: str = ""  # 使用的后端名称


# ============================================================
# CommBackend 抽象基类
# ============================================================


class CommBackend(ABC):
    """
    设备通信后端抽象基类。

    每种硬件（NVIDIA GPU / Ascend NPU / MetaX GPU）实现一个具体子类。
    只需实现底层通信原语，上层的组合逻辑由 CollectiveOps 处理。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称标识，如 'nccl', 'hccl', 'metax_ccl'"""
        ...

    @property
    @abstractmethod
    def device_type(self) -> str:
        """支持的设备类型，如 'cuda', 'npu', 'metax'"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查当前环境是否支持此后端"""
        ...

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    @abstractmethod
    def init_comm_group(self, group_info: CommGroupInfo, config: CommConfig) -> Any:
        """
        初始化通信组，返回后端内部的通信句柄（communicator handle）。

        Args:
            group_info: 通信组描述信息
            config: 后端配置

        Returns:
            comm_handle: 后端内部通信句柄，后续操作需传入
        """
        ...

    @abstractmethod
    def destroy_comm_group(self, comm_handle: Any) -> None:
        """销毁通信组，释放资源"""
        ...

    # ----------------------------------------------------------
    # 集合通信原语
    # ----------------------------------------------------------

    @abstractmethod
    def all_reduce(
        self,
        comm_handle: Any,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        AllReduce：所有进程对 tensor 执行归约，结果写回所有进程。

        Returns:
            归约后的 tensor（可以是 in-place 或新 tensor）
        """
        ...

    @abstractmethod
    def all_gather(
        self,
        comm_handle: Any,
        input_tensor: torch.Tensor,
        world_size: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        AllGather：收集所有进程的 tensor，拼接后返回。

        Returns:
            shape = [world_size * input_size, ...] 的拼接 tensor
        """
        ...

    @abstractmethod
    def reduce_scatter(
        self,
        comm_handle: Any,
        input_tensor: torch.Tensor,
        world_size: int,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        ReduceScatter：先归约再分散到各进程。

        Returns:
            当前进程分到的那份 tensor
        """
        ...

    @abstractmethod
    def broadcast(
        self,
        comm_handle: Any,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        Broadcast：从 src 进程广播 tensor 到所有进程。
        """
        ...

    @abstractmethod
    def all_to_all(
        self,
        comm_handle: Any,
        input_tensor: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        All-to-All：每个进程向每个进程发送不同的数据块。
        """
        ...

    # ----------------------------------------------------------
    # 点对点通信原语
    # ----------------------------------------------------------

    @abstractmethod
    def send(
        self,
        comm_handle: Any,
        tensor: torch.Tensor,
        dst: int,
        stream: Any = None,
    ) -> None:
        """点对点发送"""
        ...

    @abstractmethod
    def recv(
        self,
        comm_handle: Any,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """点对点接收"""
        ...

    # ----------------------------------------------------------
    # 同步
    # ----------------------------------------------------------

    @abstractmethod
    def barrier(self, comm_handle: Any) -> None:
        """组内同步屏障"""
        ...

    @abstractmethod
    def synchronize(self, stream: Any = None) -> None:
        """同步流上的所有操作"""
        ...


# ============================================================
# CommBackend 注册表
# ============================================================


class CommBackendRegistry:
    """
    后端注册表：单例，管理所有已注册的通信后端。

    使用方式:
        # 注册
        registry = CommBackendRegistry()
        registry.register(NCCLBackend())
        registry.register(HCCLBackend())

        # 获取
        backend = registry.get("nccl")

        # 自动发现可用后端
        backend = registry.get_available_backend("cuda")
    """

    _instance: CommBackendRegistry | None = None
    _backends: dict[str, CommBackend]

    def __new__(cls) -> CommBackendRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._backends = {}
        return cls._instance

    def register(self, backend: CommBackend) -> None:
        """注册一个通信后端"""
        self._backends[backend.name] = backend

    def get(self, name: str) -> CommBackend:
        """按名称获取后端"""
        if name not in self._backends:
            available = list(self._backends.keys())
            raise KeyError(f"Backend '{name}' not registered. Available: {available}")
        return self._backends[name]

    def get_available_backend(self, device_type: str) -> CommBackend:
        """
        根据设备类型自动选择可用后端。

        Args:
            device_type: 'cuda', 'npu', 'metax' 等

        Returns:
            第一个匹配且可用的后端
        """
        for backend in self._backends.values():
            if backend.device_type == device_type and backend.is_available():
                return backend
        raise RuntimeError(
            f"No available backend for device type '{device_type}'. "
            f"Registered backends: {list(self._backends.keys())}"
        )

    def list_backends(self) -> list[str]:
        """列出所有已注册后端名称"""
        return list(self._backends.keys())


# ============================================================
# 便捷函数
# ============================================================

_registry = CommBackendRegistry()


def register_backend(backend: CommBackend) -> None:
    """注册通信后端的快捷函数"""
    _registry.register(backend)


def get_backend(name: str) -> CommBackend:
    """获取通信后端的快捷函数"""
    return _registry.get(name)


def get_available_backend(device_type: str) -> CommBackend:
    """根据设备类型获取可用后端的快捷函数"""
    return _registry.get_available_backend(device_type)
