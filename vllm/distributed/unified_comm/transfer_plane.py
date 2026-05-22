# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer 3: ``TransferPlane`` - data-transfer abstraction.

Sits above the collective layer and provides a unified transfer
abstraction for the different kinds of data movement that vLLM
performs (KV cache, EC, weight transfer).

Design principles:
  - One ``TransferPlane`` implementation per transfer type.
  - The plane hides whether the transfer is backed by RDMA, shared
    memory, or a collective operation.
  - Planes are managed by a registry and may be registered at runtime.
  - The strategy layer is consulted before each transfer to choose
    the protocol and tuning knobs.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.unified_comm.collective import CollectiveGroup
from vllm.distributed.unified_comm.types import TransferProtocol, TransferType

logger = logging.getLogger(__name__)


@dataclass
class TransferMetadata:
    """传输元数据，描述一次传输的上下文"""

    transfer_type: TransferType
    src_rank: int
    dst_ranks: list[int]
    tensor_shape: tuple[int, ...]
    tensor_dtype: torch.dtype
    layer_id: int | None = None  # 用于 KV Cache：第几层
    sequence_id: int | None = None  # 用于 KV Cache：哪个序列
    expert_id: int | None = None  # 用于 EC：哪个专家
    tags: dict[str, Any] = field(default_factory=dict)  # 扩展字段


@dataclass
class TransferResult:
    """传输结果"""

    success: bool
    tensor: torch.Tensor | None = None  # 接收到的 tensor（接收方有效）
    error: str | None = None
    latency_ms: float | None = None  # 传输延迟（用于性能分析）


# ============================================================
# TransferPlane 抽象基类
# ============================================================


class TransferPlane(ABC):
    """
    传输平面抽象基类。

    每种数据传输模式实现一个 TransferPlane，隐藏底层传输协议细节。

    Usage:
        # KV Cache 传输示例
        kv_plane = KVTransferPlane(group, protocol=TransferProtocol.P2P)

        # 发送方（prefill worker）
        kv_plane.send(kv_tensor, metadata)

        # 接收方（decode worker）
        result = kv_plane.recv(metadata)
        decoded_kv = result.tensor
    """

    @property
    @abstractmethod
    def transfer_type(self) -> TransferType:
        """此平面处理的传输类型"""
        ...

    @property
    @abstractmethod
    def protocol(self) -> TransferProtocol:
        """底层使用的传输协议"""
        ...

    @abstractmethod
    def initialize(self, group: CollectiveGroup, **kwargs) -> None:
        """
        初始化传输平面。

        Args:
            group: 底层通信组
            **kwargs: 平面特有的初始化参数
        """
        ...

    @abstractmethod
    def send(
        self,
        tensor: torch.Tensor,
        metadata: TransferMetadata,
    ) -> TransferResult:
        """
        发送 tensor。

        Args:
            tensor: 要发送的数据
            metadata: 传输元数据（描述发给谁、是什么数据等）

        Returns:
            TransferResult（发送方通常只关心 success）
        """
        ...

    @abstractmethod
    def recv(
        self,
        metadata: TransferMetadata,
    ) -> TransferResult:
        """
        接收 tensor。

        Args:
            metadata: 传输元数据（描述从哪里接收、期望什么形状等）

        Returns:
            TransferResult（包含接收到的 tensor）
        """
        ...

    @abstractmethod
    def send_async(
        self,
        tensor: torch.Tensor,
        metadata: TransferMetadata,
    ) -> TransferHandle:
        """异步发送，返回句柄用于后续等待"""
        ...

    @abstractmethod
    def recv_async(
        self,
        metadata: TransferMetadata,
    ) -> TransferHandle:
        """异步接收，返回句柄用于后续等待"""
        ...

    def shutdown(self) -> None:
        """关闭传输平面，释放资源 (default: no-op)."""
        return None


# ============================================================
# TransferHandle - 异步操作句柄
# ============================================================


class TransferHandle:
    """
    异步传输操作的句柄。

    用于等待异步传输完成并获取结果。
    """

    def __init__(self, future: Any = None):
        self._future = future
        self._result: TransferResult | None = None

    def wait(self, timeout: float | None = None) -> TransferResult:
        """等待操作完成"""
        if self._result is not None:
            return self._result
        raise NotImplementedError("Subclass must implement wait()")

    def is_done(self) -> bool:
        """检查是否已完成"""
        return self._result is not None


# ============================================================
# 具体传输平面实现
# ============================================================


class KVTransferPlane(TransferPlane):
    """
    KV Cache 传输平面。

    用于 prefill-decode 分离部署场景下的 KV Cache 迁移。
    支持 P2P 和 RDMA 两种协议。
    """

    def __init__(self, preferred_protocol: TransferProtocol = TransferProtocol.P2P):
        self._preferred_protocol = preferred_protocol
        self._group: CollectiveGroup | None = None
        self._initialized = False

    @property
    def transfer_type(self) -> TransferType:
        return TransferType.KV_CACHE

    @property
    def protocol(self) -> TransferProtocol:
        return self._preferred_protocol

    def initialize(self, group: CollectiveGroup, **kwargs) -> None:
        self._group = group
        self._initialized = True

    def send(
        self,
        tensor: torch.Tensor,
        metadata: TransferMetadata,
    ) -> TransferResult:
        """发送 KV Cache tensor 到目标 decode worker"""
        assert self._initialized, "KVTransferPlane not initialized"
        assert self._group is not None

        start_time = time.perf_counter()

        try:
            for dst in metadata.dst_ranks:
                self._group.send(tensor.contiguous(), dst)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return TransferResult(success=True, latency_ms=latency_ms)
        except Exception as e:
            return TransferResult(success=False, error=str(e))

    def recv(
        self,
        metadata: TransferMetadata,
    ) -> TransferResult:
        """从 prefill worker 接收 KV Cache tensor"""
        assert self._initialized, "KVTransferPlane not initialized"
        assert self._group is not None

        start_time = time.perf_counter()

        recv_buf = torch.empty(
            metadata.tensor_shape,
            dtype=metadata.tensor_dtype,
            device=self._group.device,
        )

        try:
            result_tensor = self._group.recv(recv_buf, metadata.src_rank)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TransferResult(
                success=True, tensor=result_tensor, latency_ms=latency_ms
            )
        except Exception as e:
            return TransferResult(success=False, error=str(e))

    def send_async(
        self, tensor: torch.Tensor, metadata: TransferMetadata
    ) -> TransferHandle:
        result = self.send(tensor, metadata)
        handle = TransferHandle()
        handle._result = result
        return handle

    def recv_async(self, metadata: TransferMetadata) -> TransferHandle:
        result = self.recv(metadata)
        handle = TransferHandle()
        handle._result = result
        return handle


class WeightTransferPlane(TransferPlane):
    """
    模型权重传输平面。

    用于模型加载时的权重分发、热更新时的权重同步。
    """

    def __init__(
        self,
        preferred_protocol: TransferProtocol = TransferProtocol.COLLECTIVE,
    ):
        self._preferred_protocol = preferred_protocol
        self._group: CollectiveGroup | None = None
        self._initialized = False

    @property
    def transfer_type(self) -> TransferType:
        return TransferType.WEIGHT

    @property
    def protocol(self) -> TransferProtocol:
        return self._preferred_protocol

    def initialize(self, group: CollectiveGroup, **kwargs) -> None:
        self._group = group
        self._initialized = True

    def send(self, tensor: torch.Tensor, metadata: TransferMetadata) -> TransferResult:
        """广播权重（大权重自动分桶）"""
        assert self._initialized and self._group is not None

        start_time = time.perf_counter()

        try:
            result_tensor = self._group.broadcast(tensor, src=metadata.src_rank)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TransferResult(
                success=True, tensor=result_tensor, latency_ms=latency_ms
            )
        except Exception as e:
            return TransferResult(success=False, error=str(e))

    def recv(self, metadata: TransferMetadata) -> TransferResult:
        """接收广播的权重（broadcast 模式下 send 和 recv 合一）"""
        assert self._group is not None
        return self.send(
            torch.empty(
                metadata.tensor_shape,
                dtype=metadata.tensor_dtype,
                device=self._group.device,
            ),
            metadata,
        )

    def send_async(
        self, tensor: torch.Tensor, metadata: TransferMetadata
    ) -> TransferHandle:
        result = self.send(tensor, metadata)
        handle = TransferHandle()
        handle._result = result
        return handle

    def recv_async(self, metadata: TransferMetadata) -> TransferHandle:
        result = self.recv(metadata)
        handle = TransferHandle()
        handle._result = result
        return handle


class ECTransferPlane(TransferPlane):
    """
    Expert-Centric 传输平面。

    用于 MoE 模型中 token 到 expert 的路由和回传。
    底层通常基于 All-to-All。
    """

    def __init__(
        self,
        preferred_protocol: TransferProtocol = TransferProtocol.COLLECTIVE,
    ):
        self._preferred_protocol = preferred_protocol
        self._group: CollectiveGroup | None = None
        self._initialized = False

    @property
    def transfer_type(self) -> TransferType:
        return TransferType.EC

    @property
    def protocol(self) -> TransferProtocol:
        return self._preferred_protocol

    def initialize(self, group: CollectiveGroup, **kwargs) -> None:
        self._group = group
        self._initialized = True

    def send(self, tensor: torch.Tensor, metadata: TransferMetadata) -> TransferResult:
        """All-to-All 发送（MoE dispatch）"""
        assert self._initialized and self._group is not None

        start_time = time.perf_counter()

        try:
            scatter_sizes = metadata.tags.get("scatter_sizes")
            gather_sizes = metadata.tags.get("gather_sizes")
            result_tensor = self._group.all_to_all(
                tensor,
                scatter_dim=metadata.tags.get("scatter_dim", 0),
                gather_dim=metadata.tags.get("gather_dim", 0),
                scatter_sizes=scatter_sizes,
                gather_sizes=gather_sizes,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TransferResult(
                success=True, tensor=result_tensor, latency_ms=latency_ms
            )
        except Exception as e:
            return TransferResult(success=False, error=str(e))

    def recv(self, metadata: TransferMetadata) -> TransferResult:
        """EC 模式下 send 和 recv 通过 All-to-All 同时完成"""
        raise NotImplementedError(
            "ECTransferPlane uses all_to_all which is symmetric. Use send() instead."
        )

    def send_async(
        self, tensor: torch.Tensor, metadata: TransferMetadata
    ) -> TransferHandle:
        result = self.send(tensor, metadata)
        handle = TransferHandle()
        handle._result = result
        return handle

    def recv_async(self, metadata: TransferMetadata) -> TransferHandle:
        raise NotImplementedError("Use send_async for EC (all-to-all is symmetric)")


# ============================================================
# TransferPlane 注册表
# ============================================================


class TransferPlaneRegistry:
    """
    传输平面注册表：管理所有已注册的传输平面实例。

    Usage:
        registry = TransferPlaneRegistry()
        registry.register(KVTransferPlane())
        registry.register(WeightTransferPlane())

        kv_plane = registry.get(TransferType.KV_CACHE)
    """

    _instance: TransferPlaneRegistry | None = None
    _planes: dict[TransferType, TransferPlane]

    def __new__(cls) -> TransferPlaneRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._planes = {}
        return cls._instance

    def register(self, plane: TransferPlane) -> None:
        """注册传输平面"""
        self._planes[plane.transfer_type] = plane

    def get(self, transfer_type: TransferType) -> TransferPlane:
        """获取传输平面"""
        if transfer_type not in self._planes:
            available = [t.name for t in self._planes]
            raise KeyError(
                f"TransferPlane for {transfer_type.name} not registered. "
                f"Available: {available}"
            )
        return self._planes[transfer_type]

    def list_planes(self) -> list[TransferType]:
        """列出已注册的传输类型"""
        return list(self._planes.keys())

    def shutdown_all(self) -> None:
        """关闭所有传输平面"""
        for plane in self._planes.values():
            plane.shutdown()
        self._planes.clear()
