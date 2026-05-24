# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer 4: ``CommStrategy`` - communication strategy layer.

Picks the optimal algorithm and transport link based on runtime
context (hardware topology, message size, communication pattern).

Design principles:
  - Strategy is decoupled from execution: it only makes decisions and
    never performs communication itself.
  - Both rule-driven and config-driven modes are supported.
  - Extensible: callers may register custom strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from vllm.distributed.unified_comm.types import TransferProtocol

# ============================================================
# 策略上下文（输入）
# ============================================================


class CommPattern(Enum):
    """通信模式"""

    ALL_REDUCE = auto()
    ALL_GATHER = auto()
    REDUCE_SCATTER = auto()
    BROADCAST = auto()
    ALL_TO_ALL = auto()
    P2P_SEND_RECV = auto()
    KV_TRANSFER = auto()
    WEIGHT_SYNC = auto()
    EC_DISPATCH = auto()


@dataclass
class TopologyInfo:
    """硬件拓扑信息"""

    num_nodes: int = 1  # 节点数
    gpus_per_node: int = 8  # 每节点加速器数
    intra_node_bandwidth_gbps: float = 600.0  # 节点内带宽 (GB/s)
    inter_node_bandwidth_gbps: float = 100.0  # 节点间带宽 (GB/s)
    has_nvswitch: bool = False  # 是否有 NVSwitch / HCCS
    has_rdma: bool = False  # 是否支持 RDMA
    device_type: str = "cuda"  # 设备类型


@dataclass
class CommContext:
    """
    通信策略的决策上下文。

    策略层根据这些信息做出算法和链路选择。
    """

    pattern: CommPattern  # 通信模式
    tensor_size_bytes: int  # 数据量（字节）
    world_size: int  # 参与通信的进程数
    topology: TopologyInfo  # 硬件拓扑
    is_intra_node: bool = True  # 是否节点内通信
    dtype: torch.dtype = torch.float16  # 数据类型
    urgency: str = "normal"  # 优先级：high / normal / low
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 策略决策（输出）
# ============================================================


class CommAlgorithm(Enum):
    """通信算法"""

    # AllReduce 算法
    RING = auto()  # Ring AllReduce
    TREE = auto()  # Tree AllReduce
    RECURSIVE_HALVING = auto()  # 递归减半
    DIRECT = auto()  # 直接通信（小数据量）
    BUCKET = auto()  # 分桶通信

    # 传输协议选择
    COLLECTIVE_BASED = auto()  # 基于集合通信
    P2P_BASED = auto()  # 基于点对点
    RDMA_BASED = auto()  # 基于 RDMA
    SHM_BASED = auto()  # 基于共享内存

    # 特殊
    AUTO = auto()  # 让后端自动选择


@dataclass
class CommDecision:
    """
    策略层的决策结果。

    上层按此决策执行通信。
    """

    algorithm: CommAlgorithm  # 选定的算法
    protocol: TransferProtocol  # 选定的传输协议
    use_high_priority_stream: bool = False  # 是否使用高优先级流
    num_chunks: int = 1  # 分块数（用于分桶/流水线）
    overlap_compute: bool = False  # 是否与计算重叠
    compression: str | None = None  # 压缩方式（如 "fp16", "int8"）
    backend_hint: str | None = None  # 推荐的后端名称
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CommDecision(algo={self.algorithm.name}, "
            f"proto={self.protocol.name}, "
            f"chunks={self.num_chunks}, "
            f"overlap={self.overlap_compute})"
        )


# ============================================================
# CommStrategy 抽象基类
# ============================================================


class CommStrategy(ABC):
    """
    通信策略抽象基类。

    不同策略实现可以基于规则、配置文件、或运行时 profiling。

    Usage:
        strategy = DefaultStrategy()
        ctx = CommContext(
            pattern=CommPattern.ALL_REDUCE,
            tensor_size_bytes=1024*1024,
            world_size=8,
            topology=TopologyInfo(gpus_per_node=8),
        )
        decision = strategy.decide(ctx)
    """

    @abstractmethod
    def decide(self, context: CommContext) -> CommDecision:
        """
        根据上下文做出通信策略决策。

        Args:
            context: 通信上下文信息

        Returns:
            CommDecision: 策略决策结果
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        ...


# ============================================================
# 默认策略实现
# ============================================================


class DefaultStrategy(CommStrategy):
    """
    默认通信策略：基于简单规则的启发式选择。

    规则：
    - 小数据量（< 256KB）：直接通信
    - 中等数据量：Ring 算法
    - 大数据量 + NVSwitch：Tree 算法
    - 跨节点：分桶 + 流水线
    - KV Transfer：优先 P2P，有 RDMA 则用 RDMA
    - EC Dispatch：All-to-All（底层 backend 自选算法）
    """

    # 阈值配置（可外部覆盖）
    SMALL_MSG_THRESHOLD = 256 * 1024  # 256 KB
    LARGE_MSG_THRESHOLD = 64 * 1024 * 1024  # 64 MB
    BUCKET_SIZE = 25 * 1024 * 1024  # 25 MB per bucket

    def name(self) -> str:
        return "default"

    def decide(self, context: CommContext) -> CommDecision:
        pattern = context.pattern

        # ---- KV Transfer ----
        if pattern == CommPattern.KV_TRANSFER:
            return self._decide_kv_transfer(context)

        # ---- EC Dispatch (MoE All-to-All) ----
        if pattern == CommPattern.EC_DISPATCH:
            return self._decide_ec_dispatch(context)

        # ---- Weight Sync ----
        if pattern == CommPattern.WEIGHT_SYNC:
            return self._decide_weight_sync(context)

        # ---- 集合通信模式 ----
        return self._decide_collective(context)

    def _decide_collective(self, context: CommContext) -> CommDecision:
        """集合通信策略选择"""
        size = context.tensor_size_bytes
        topo = context.topology

        # 小数据量：直接通信
        if size < self.SMALL_MSG_THRESHOLD:
            return CommDecision(
                algorithm=CommAlgorithm.DIRECT,
                protocol=TransferProtocol.COLLECTIVE,
            )

        # 有 NVSwitch/HCCS 且节点内：Tree 算法
        if context.is_intra_node and topo.has_nvswitch:
            return CommDecision(
                algorithm=CommAlgorithm.TREE,
                protocol=TransferProtocol.COLLECTIVE,
            )

        # 大数据量跨节点：分桶 + 流水线
        if size > self.LARGE_MSG_THRESHOLD and not context.is_intra_node:
            num_chunks = max(1, size // self.BUCKET_SIZE)
            return CommDecision(
                algorithm=CommAlgorithm.BUCKET,
                protocol=TransferProtocol.COLLECTIVE,
                num_chunks=num_chunks,
                overlap_compute=True,
            )

        # 默认：Ring 算法
        return CommDecision(
            algorithm=CommAlgorithm.RING,
            protocol=TransferProtocol.COLLECTIVE,
        )

    def _decide_kv_transfer(self, context: CommContext) -> CommDecision:
        """KV Cache 传输策略"""
        topo = context.topology

        # 有 RDMA：优先 RDMA
        if topo.has_rdma and not context.is_intra_node:
            return CommDecision(
                algorithm=CommAlgorithm.RDMA_BASED,
                protocol=TransferProtocol.RDMA,
                use_high_priority_stream=True,
            )

        # 节点内：共享内存
        if context.is_intra_node:
            return CommDecision(
                algorithm=CommAlgorithm.SHM_BASED,
                protocol=TransferProtocol.SHM,
            )

        # 默认：P2P
        return CommDecision(
            algorithm=CommAlgorithm.P2P_BASED,
            protocol=TransferProtocol.P2P,
            use_high_priority_stream=True,
        )

    def _decide_ec_dispatch(self, context: CommContext) -> CommDecision:
        """MoE Expert-Centric dispatch 策略"""
        return CommDecision(
            algorithm=CommAlgorithm.AUTO,
            protocol=TransferProtocol.COLLECTIVE,
            overlap_compute=context.urgency == "high",
        )

    def _decide_weight_sync(self, context: CommContext) -> CommDecision:
        """权重同步策略"""
        size = context.tensor_size_bytes

        # 大模型权重：分桶广播
        if size > self.LARGE_MSG_THRESHOLD:
            num_chunks = max(1, size // self.BUCKET_SIZE)
            return CommDecision(
                algorithm=CommAlgorithm.BUCKET,
                protocol=TransferProtocol.COLLECTIVE,
                num_chunks=num_chunks,
            )

        # 小权重：直接广播
        return CommDecision(
            algorithm=CommAlgorithm.DIRECT,
            protocol=TransferProtocol.COLLECTIVE,
        )


# ============================================================
# 配置驱动策略
# ============================================================


class ConfigDrivenStrategy(CommStrategy):
    """
    配置驱动策略：从配置文件/字典读取策略规则。

    适用于用户需要精细控制通信行为的场景。

    Config 格式示例:
        {
            "all_reduce": {
                "small_threshold_bytes": 262144,
                "algorithm": "ring",
                "overlap_compute": false
            },
            "kv_transfer": {
                "protocol": "rdma",
                "high_priority": true
            }
        }
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def name(self) -> str:
        return "config_driven"

    def decide(self, context: CommContext) -> CommDecision:
        """根据配置做决策"""
        pattern_key = context.pattern.name.lower()
        pattern_config = self._config.get(pattern_key, {})

        algorithm_str = pattern_config.get("algorithm", "auto")
        protocol_str = pattern_config.get("protocol", "collective")

        algorithm = self._parse_algorithm(algorithm_str)
        protocol = self._parse_protocol(protocol_str)

        return CommDecision(
            algorithm=algorithm,
            protocol=protocol,
            use_high_priority_stream=pattern_config.get("high_priority", False),
            num_chunks=pattern_config.get("num_chunks", 1),
            overlap_compute=pattern_config.get("overlap_compute", False),
            compression=pattern_config.get("compression"),
            backend_hint=pattern_config.get("backend"),
        )

    @staticmethod
    def _parse_algorithm(s: str) -> CommAlgorithm:
        mapping = {
            "ring": CommAlgorithm.RING,
            "tree": CommAlgorithm.TREE,
            "recursive_halving": CommAlgorithm.RECURSIVE_HALVING,
            "direct": CommAlgorithm.DIRECT,
            "bucket": CommAlgorithm.BUCKET,
            "auto": CommAlgorithm.AUTO,
            "p2p": CommAlgorithm.P2P_BASED,
            "rdma": CommAlgorithm.RDMA_BASED,
            "shm": CommAlgorithm.SHM_BASED,
        }
        return mapping.get(s.lower(), CommAlgorithm.AUTO)

    @staticmethod
    def _parse_protocol(s: str) -> TransferProtocol:
        mapping = {
            "collective": TransferProtocol.COLLECTIVE,
            "p2p": TransferProtocol.P2P,
            "rdma": TransferProtocol.RDMA,
            "shm": TransferProtocol.SHM,
            "store": TransferProtocol.STORE,
        }
        return mapping.get(s.lower(), TransferProtocol.COLLECTIVE)
