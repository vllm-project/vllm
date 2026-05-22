# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer 2: ``CollectiveOps`` - device-agnostic collective layer.

Builds on top of :class:`CommBackend` to provide device-independent
collective operations and to manage communication-group lifecycles.

Design principles:
  - ``CollectiveGroup`` owns the state and operations of one group.
  - ``CollectiveOps`` provides stateless composite primitives.
  - Both synchronous and asynchronous modes are supported.
  - Before issuing a collective the strategy layer is consulted to
    pick the actual execution path.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

from vllm.distributed.unified_comm.backend import (
    CommBackend,
    CommConfig,
    CommGroupInfo,
    ReduceOp,
    get_available_backend,
    get_backend,
)
from vllm.distributed.unified_comm.strategy import (
    CommAlgorithm,
    CommContext,
    CommDecision,
    CommPattern,
    CommStrategy,
    TopologyInfo,
)

logger = logging.getLogger(__name__)


# ============================================================
# CollectiveGroup - 通信组封装
# ============================================================


class CollectiveGroup:
    """
    统一通信组：封装了一个通信组的完整生命周期和操作。

    相当于现有 GroupCoordinator 的设备无关替代品。

    Usage:
        group = CollectiveGroup.create(
            ranks=[0, 1, 2, 3],
            local_rank=0,
            device=torch.device("cuda:0"),
            backend_name="nccl",  # 或 "hccl", "metax_ccl"
        )

        result = group.all_reduce(tensor)
        group.barrier()
        group.destroy()
    """

    def __init__(
        self,
        backend: CommBackend,
        group_info: CommGroupInfo,
        comm_handle: Any,
        config: CommConfig,
        strategy: CommStrategy | None = None,
        topology: TopologyInfo | None = None,
    ):
        self._backend = backend
        self._group_info = group_info
        self._comm_handle = comm_handle
        self._config = config
        self._destroyed = False
        # 策略层接入
        self._strategy = strategy
        self._topology = topology or self._detect_topology()
        self._strategy_enabled = strategy is not None
        # OPT #1: 决策缓存 — key=(pattern, size_bucket, dtype) → CommDecision
        self._decision_cache: dict[tuple, CommDecision] = {}
        # OPT #5: 缓存 _is_intra_node() (它只依赖 ranks/topology, 不会变)
        self._cached_is_intra_node: bool | None = None
        # OPT #2: 小 tensor fast-path 阈值 (字节)
        self._small_msg_threshold: int = 256 * 1024

    # ----------------------------------------------------------
    # 工厂方法
    # ----------------------------------------------------------

    @classmethod
    def create(
        cls,
        ranks: list[int],
        local_rank: int,
        device: torch.device,
        backend_name: str | None = None,
        config: CommConfig | None = None,
        strategy: CommStrategy | None = None,
        topology: TopologyInfo | None = None,
        existing_device_group: Any = None,
        existing_cpu_group: Any = None,
    ) -> CollectiveGroup:
        """
        创建通信组。

        Args:
            ranks: 参与通信的全局 rank 列表
            local_rank: 当前进程的全局 rank
            device: 绑定设备
            backend_name: 后端名称。如果不指定，根据 device.type 自动选择
            config: 后端配置
            strategy: 通信策略。传入则在每次通信前咨询策略层。
            topology: 硬件拓扑信息。None 则自动探测。
            existing_device_group: 已有的 device ProcessGroup（避免重复创建）
            existing_cpu_group: 已有的 CPU ProcessGroup（避免重复创建）

        Returns:
            初始化完毕的 CollectiveGroup
        """
        config = config or CommConfig()

        # 将已有 PG 传递给后端，避免重复创建
        if existing_device_group is not None:
            config.extra["existing_device_group"] = existing_device_group
        if existing_cpu_group is not None:
            config.extra["existing_cpu_group"] = existing_cpu_group

        # 自动选择后端
        if backend_name:
            backend = get_backend(backend_name)
        else:
            backend = get_available_backend(device.type)

        # 构造组信息
        rank_in_group = ranks.index(local_rank)
        group_info = CommGroupInfo(
            ranks=ranks,
            rank_in_group=rank_in_group,
            world_size=len(ranks),
            device=device,
            backend_name=backend.name,
        )

        # 初始化通信组
        comm_handle = backend.init_comm_group(group_info, config)

        # 若未传入策略，尝试获取全局策略
        if strategy is None:
            try:
                from vllm.distributed.unified_comm.initialize import get_strategy

                strategy = get_strategy()
            except Exception:
                pass

        return cls(
            backend=backend,
            group_info=group_info,
            comm_handle=comm_handle,
            config=config,
            strategy=strategy,
            topology=topology,
        )

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def rank(self) -> int:
        """当前进程在组内的 rank"""
        return self._group_info.rank_in_group

    @property
    def world_size(self) -> int:
        """组内进程数"""
        return self._group_info.world_size

    @property
    def device(self) -> torch.device:
        """绑定设备"""
        return self._group_info.device

    @property
    def backend_name(self) -> str:
        """使用的后端名称"""
        return self._backend.name

    @property
    def ranks(self) -> list[int]:
        """组内全局 rank 列表"""
        return self._group_info.ranks

    # ----------------------------------------------------------
    # 集合通信操作（策略驱动）
    # ----------------------------------------------------------

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        AllReduce — 策略驱动版本。

        根据策略决策选择：
        - DIRECT / RING / TREE: 直接单次 AllReduce（由后端库选择算法）
        - BUCKET: 分桶 AllReduce，每桶独立通信（大数据量优化）

        OPT #2: 小 tensor 跳过策略层, 直接走 backend.
        """
        self._check_alive()

        # OPT #2: small-tensor fast-path
        if tensor.nelement() * tensor.element_size() < self._small_msg_threshold:
            return self._backend.all_reduce(self._comm_handle, tensor, op, stream)

        # 咨询策略层
        decision = self._consult_strategy(CommPattern.ALL_REDUCE, tensor)

        if (
            decision
            and decision.algorithm == CommAlgorithm.BUCKET
            and decision.num_chunks > 1
        ):
            return self._all_reduce_bucketed(tensor, op, decision, stream)

        # 默认路径：单次 AllReduce
        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.all_reduce(self._comm_handle, tensor, op, effective_stream)

    def all_gather(
        self,
        tensor: torch.Tensor,
        stream: Any = None,
    ) -> torch.Tensor:
        """AllGather：收集所有进程的 tensor（策略驱动）"""
        self._check_alive()
        # OPT #2: small-tensor fast-path
        if tensor.nelement() * tensor.element_size() < self._small_msg_threshold:
            return self._backend.all_gather(
                self._comm_handle, tensor, self.world_size, stream
            )
        decision = self._consult_strategy(CommPattern.ALL_GATHER, tensor)
        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.all_gather(
            self._comm_handle, tensor, self.world_size, effective_stream
        )

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """ReduceScatter：归约后分散（策略驱动）"""
        self._check_alive()
        decision = self._consult_strategy(CommPattern.REDUCE_SCATTER, tensor)
        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.reduce_scatter(
            self._comm_handle, tensor, self.world_size, op, effective_stream
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """Broadcast：从 src 广播（策略驱动）"""
        self._check_alive()
        # OPT #2: small-tensor fast-path
        if tensor.nelement() * tensor.element_size() < self._small_msg_threshold:
            return self._backend.broadcast(self._comm_handle, tensor, src, stream)
        decision = self._consult_strategy(CommPattern.BROADCAST, tensor)

        # 大权重广播：分桶
        if (
            decision
            and decision.algorithm == CommAlgorithm.BUCKET
            and decision.num_chunks > 1
        ):
            return self._broadcast_bucketed(tensor, src, decision, stream)

        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.broadcast(self._comm_handle, tensor, src, effective_stream)

    def all_to_all(
        self,
        input_tensor: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
        stream: Any = None,
    ) -> torch.Tensor:
        """All-to-All（策略驱动）"""
        self._check_alive()
        decision = self._consult_strategy(CommPattern.ALL_TO_ALL, input_tensor)
        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.all_to_all(
            self._comm_handle,
            input_tensor,
            scatter_dim,
            gather_dim,
            scatter_sizes,
            gather_sizes,
            effective_stream,
        )

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        stream: Any = None,
    ) -> None:
        """点对点发送（策略驱动）"""
        self._check_alive()
        decision = self._consult_strategy(CommPattern.P2P_SEND_RECV, tensor)
        effective_stream = self._resolve_stream(stream, decision)
        self._backend.send(self._comm_handle, tensor, dst, effective_stream)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """点对点接收（策略驱动）"""
        self._check_alive()
        decision = self._consult_strategy(CommPattern.P2P_SEND_RECV, tensor)
        effective_stream = self._resolve_stream(stream, decision)
        return self._backend.recv(self._comm_handle, tensor, src, effective_stream)

    def barrier(self) -> None:
        """组内同步"""
        self._check_alive()
        self._backend.barrier(self._comm_handle)

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def destroy(self) -> None:
        """销毁通信组"""
        if not self._destroyed:
            self._backend.destroy_comm_group(self._comm_handle)
            self._destroyed = True

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("This CollectiveGroup has been destroyed")

    def __del__(self):
        self.destroy()

    def __repr__(self) -> str:
        return (
            f"CollectiveGroup(backend={self.backend_name}, "
            f"rank={self.rank}/{self.world_size}, "
            f"device={self.device}, "
            f"strategy={self._strategy.name() if self._strategy else 'none'})"
        )

    # ----------------------------------------------------------
    # 策略层接入
    # ----------------------------------------------------------

    def _consult_strategy(
        self,
        pattern: CommPattern,
        tensor: torch.Tensor,
    ) -> CommDecision | None:
        """
        咨询策略层获取决策。

        如果没有配置策略或策略出错，返回 None（降级为默认行为）。

        OPT #1: 加 LRU-style 缓存。key 由 pattern + size-bucket + dtype 决定;
                hit 时跳过整个 CommContext 构造和 strategy.decide() 调用。
        OPT #4: 把 logger.debug 的 f-string 改为 lazy / isEnabledFor 守卫。
        """
        if not self._strategy_enabled or self._strategy is None:
            return None

        size_bytes = tensor.nelement() * tensor.element_size()
        # size 分桶: 0=tiny(<256K), 1=small(<2M), 2=med(<64M), 3=large
        if size_bytes < 256 * 1024:
            size_bucket = 0
        elif size_bytes < 2 * 1024 * 1024:
            size_bucket = 1
        elif size_bytes < 64 * 1024 * 1024:
            size_bucket = 2
        else:
            size_bucket = 3
        cache_key = (pattern, size_bucket, tensor.dtype)

        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            ctx = CommContext(
                pattern=pattern,
                tensor_size_bytes=size_bytes,
                world_size=self.world_size,
                topology=self._topology,
                is_intra_node=self._is_intra_node(),
                dtype=tensor.dtype,
            )
            decision = self._strategy.decide(ctx)
            self._decision_cache[cache_key] = decision
            # OPT #4: 仅在 debug 开启时构造 f-string
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Strategy] pattern=%s, size=%s, decision=%s",
                    pattern.name,
                    ctx.tensor_size_bytes,
                    decision,
                )
            return decision
        except Exception as e:
            logger.warning(
                "Strategy decision failed, falling back to default: %s",
                e,
            )
            return None

    def _resolve_stream(self, user_stream: Any, decision: CommDecision | None) -> Any:
        """
        根据策略决策解析应使用的 stream。

        优先级：用户显式传入 > 策略要求高优先级 > None（使用默认 stream）
        """
        if user_stream is not None:
            return user_stream
        if decision and decision.use_high_priority_stream:
            return self._get_high_priority_stream()
        return None

    def _get_high_priority_stream(self) -> Any:
        """获取高优先级 stream（延迟创建）"""
        if not hasattr(self, "_hp_stream") or self._hp_stream is None:
            device = self._group_info.device
            if device.type == "cuda":
                self._hp_stream = torch.cuda.Stream(device=device, priority=-1)
            elif device.type == "npu":
                try:
                    import torch_npu  # noqa: F401

                    self._hp_stream = torch.npu.Stream(device=device, priority=-1)
                except Exception:
                    self._hp_stream = None
            else:
                self._hp_stream = None
        return self._hp_stream

    # ----------------------------------------------------------
    # 策略执行器
    # ----------------------------------------------------------

    def _all_reduce_bucketed(
        self,
        tensor: torch.Tensor,
        op: ReduceOp,
        decision: CommDecision,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        分桶 AllReduce：将大 tensor 切分为多块，逐块通信。
        """
        num_chunks = decision.num_chunks
        flat = tensor.contiguous().view(-1)
        chunk_size = (flat.numel() + num_chunks - 1) // num_chunks
        effective_stream = self._resolve_stream(stream, decision)

        logger.debug(
            "[BucketedAllReduce] total_elements=%s, chunks=%s, chunk_size=%s",
            flat.numel(),
            num_chunks,
            chunk_size,
        )

        # 分桶通信（chunk 是 flat 的 view，in-place 修改已生效）
        chunks = list(flat.split(chunk_size))
        for chunk in chunks:
            self._backend.all_reduce(self._comm_handle, chunk, op, effective_stream)

        return flat.view(tensor.shape)

    def _broadcast_bucketed(
        self,
        tensor: torch.Tensor,
        src: int,
        decision: CommDecision,
        stream: Any = None,
    ) -> torch.Tensor:
        """分桶 Broadcast：将大权重切分为多块逐块广播。"""
        num_chunks = decision.num_chunks
        flat = tensor.contiguous().view(-1)
        chunk_size = (flat.numel() + num_chunks - 1) // num_chunks
        effective_stream = self._resolve_stream(stream, decision)

        logger.debug(
            "[BucketedBroadcast] total_elements=%s, chunks=%s, src=%s",
            flat.numel(),
            num_chunks,
            src,
        )

        chunks = list(flat.split(chunk_size))
        for chunk in chunks:
            self._backend.broadcast(self._comm_handle, chunk, src, effective_stream)

        return flat.view(tensor.shape)

    # ----------------------------------------------------------
    # 拓扑探测
    # ----------------------------------------------------------

    def _detect_topology(self) -> TopologyInfo:
        """自动探测硬件拓扑信息。"""
        import os

        device_type = self._group_info.device.type
        gpus_per_node = 8  # 保守默认值

        if device_type == "cuda":
            with contextlib.suppress(Exception):
                gpus_per_node = torch.cuda.device_count()
        elif device_type == "npu":
            try:
                import torch_npu  # noqa: F401

                gpus_per_node = torch.npu.device_count()
            except Exception:
                pass

        # 检测节点数
        num_nodes = max(1, self.world_size // gpus_per_node)

        # NVSwitch 探测
        has_nvswitch = os.environ.get("UNIFIED_COMM_HAS_NVSWITCH", "").lower() in (
            "1",
            "true",
        )

        if device_type == "cuda" and not has_nvswitch:
            has_nvswitch = self._probe_nvswitch()

        # RDMA 探测
        has_rdma = os.environ.get("UNIFIED_COMM_HAS_RDMA", "").lower() in ("1", "true")
        if not has_rdma:
            nccl_ib = os.environ.get("NCCL_IB_DISABLE", "0")
            if nccl_ib == "0":
                has_rdma = self._probe_rdma()

        # 带宽估计
        intra_bw = 600.0 if has_nvswitch else 300.0

        inter_bw = 100.0 if has_rdma else 25.0

        return TopologyInfo(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            intra_node_bandwidth_gbps=intra_bw,
            inter_node_bandwidth_gbps=inter_bw,
            has_nvswitch=has_nvswitch,
            has_rdma=has_rdma,
            device_type=device_type,
        )

    def _is_intra_node(self) -> bool:
        """判断当前通信组是否全在同一个节点内 (OPT #5: 缓存结果)"""
        if self._cached_is_intra_node is not None:
            return self._cached_is_intra_node
        gpus_per_node = self._topology.gpus_per_node
        if self.world_size <= gpus_per_node:
            ranks = self._group_info.ranks
            node_ids = set(r // gpus_per_node for r in ranks)
            result = len(node_ids) == 1
        else:
            result = False
        self._cached_is_intra_node = result
        return result

    @staticmethod
    def _probe_nvswitch() -> bool:
        """尝试探测 NVSwitch（CUDA 环境）。"""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                output = result.stdout
                if "NV12" in output or "NV18" in output:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _probe_rdma() -> bool:
        """尝试探测 RDMA 可用性。"""
        try:
            import os

            ib_path = "/sys/class/infiniband"
            if os.path.exists(ib_path):
                devices = os.listdir(ib_path)
                return len(devices) > 0
        except Exception:
            pass
        return False


# ============================================================
# CollectiveOps - 组合通信操作（无状态工具类）
# ============================================================


class CollectiveOps:
    """
    高级集合通信操作：在 CollectiveGroup 之上提供常用组合模式。

    这些是无状态函数，方便上层（如 TransferPlane、模型层）直接调用。
    """

    @staticmethod
    def all_reduce_coalesced(
        group: CollectiveGroup,
        tensors: list[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
    ) -> list[torch.Tensor]:
        """
        合并多个小 tensor 的 AllReduce（减少通信次数）。

        将多个 tensor 打平合并为一个大 tensor 做 AllReduce，然后拆分回去。
        """
        if not tensors:
            return []

        shapes = [t.shape for t in tensors]
        flat_tensors = [t.contiguous().view(-1) for t in tensors]
        coalesced = torch.cat(flat_tensors, dim=0)

        result = group.all_reduce(coalesced, op)

        outputs = []
        offset = 0
        for shape in shapes:
            numel = 1
            for s in shape:
                numel *= s
            outputs.append(result[offset : offset + numel].view(shape))
            offset += numel

        return outputs

    @staticmethod
    def ring_all_reduce(
        group: CollectiveGroup,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        """
        Ring AllReduce 实现（用于后端不支持 AllReduce 时的 fallback）。

        分为 reduce-scatter 和 all-gather 两步。
        """
        scattered = group.reduce_scatter(tensor, op)
        result = group.all_gather(scattered)
        return result

    @staticmethod
    def broadcast_from_rank0(
        group: CollectiveGroup,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """从 rank 0 广播 tensor 到所有进程"""
        return group.broadcast(tensor, src=0)

    @staticmethod
    def scatter(
        group: CollectiveGroup,
        tensor: torch.Tensor,
        src: int = 0,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Scatter：将 tensor 从 src 均分到所有进程。

        基于 Send/Recv 实现。
        """
        if group.rank == src:
            chunks = tensor.chunk(group.world_size, dim=dim)
            for dst_rank in range(group.world_size):
                if dst_rank != src:
                    group.send(chunks[dst_rank].contiguous(), dst_rank)
            return chunks[src].contiguous()
        else:
            chunk_size = list(tensor.shape)
            chunk_size[dim] = chunk_size[dim] // group.world_size
            recv_buf = torch.empty(chunk_size, dtype=tensor.dtype, device=group.device)
            return group.recv(recv_buf, src)

    @staticmethod
    def gather(
        group: CollectiveGroup,
        tensor: torch.Tensor,
        dst: int = 0,
        dim: int = 0,
    ) -> torch.Tensor | None:
        """
        Gather：将所有进程的 tensor 收集到 dst 进程。

        Returns:
            dst 进程返回拼接后的 tensor；其他进程返回 None
        """
        if group.rank == dst:
            chunks = [torch.empty_like(tensor) for _ in range(group.world_size)]
            chunks[dst] = tensor
            for src_rank in range(group.world_size):
                if src_rank != dst:
                    chunks[src_rank] = group.recv(torch.empty_like(tensor), src_rank)
            return torch.cat(chunks, dim=dim)
        else:
            group.send(tensor, dst)
            return None
