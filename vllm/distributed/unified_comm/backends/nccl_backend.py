# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL backend - NVIDIA GPU adapter for ``CommBackend``.

Adapts the NVIDIA Collective Communications Library (NCCL) to the
unified :class:`CommBackend` interface. The implementation mirrors
the call-pattern used by vLLM upstream's ``pynccl.py`` /
``pynccl_wrapper.py``.

Two execution modes are supported:
  Mode A (``torch.distributed``):
      Uses the NCCL backend through ``torch.distributed``. Recommended
      for general use.
  Mode B (direct C API):
      Calls into ``libnccl.so`` directly. Required for the CUDA-graph
      capture path. Enabled with ``UNIFIED_COMM_USE_DIRECT_NCCL=1``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.unified_comm.backend import (
    CommBackend,
    CommConfig,
    CommGroupInfo,
    ReduceOp,
)

logger = logging.getLogger(__name__)


# ============================================================
# ReduceOp 映射工具
# ============================================================

# 统一层 ReduceOp → NCCL C API 归约操作码
_REDUCE_OP_TO_NCCL_INT = {
    ReduceOp.SUM: 0,  # ncclSum
    ReduceOp.PRODUCT: 1,  # ncclProd
    ReduceOp.MAX: 2,  # ncclMax
    ReduceOp.MIN: 3,  # ncclMin
    ReduceOp.AVG: 4,  # ncclAvg
}


def _reduce_op_to_nccl(op: ReduceOp) -> int:
    """将统一 ReduceOp 转为 NCCL C API 的 int 枚举值"""
    return _REDUCE_OP_TO_NCCL_INT[op]


# ============================================================
# NCCL 通信句柄
# ============================================================


@dataclass
class NCCLCommHandle:
    """
    NCCL 通信组句柄，封装所有需要的状态。

    对应关系：
      - device_group: torch.distributed NCCL ProcessGroup (Mode A 使用)
      - cpu_group: torch.distributed Gloo ProcessGroup (辅助 CPU 通信)
      - nccl_comm: ncclComm_t 原始句柄 (Mode B 使用)
      - nccl_lib: NCCLLibrary 实例 (Mode B 使用)
      - owns_groups: 是否拥有 ProcessGroup 的生命周期
    """

    device_group: dist.ProcessGroup | None = None
    cpu_group: dist.ProcessGroup | None = None
    group_info: CommGroupInfo | None = None
    device: torch.device | None = None
    nccl_comm: Any = None  # ncclComm_t (ctypes.c_void_p)
    nccl_lib: Any = None  # NCCLLibrary instance
    use_direct: bool = False  # 是否使用直接 C API 模式
    owns_groups: bool = True  # 是否拥有 PG 生命周期管理权


# ============================================================
# NCCLBackend 实现
# ============================================================


class NCCLBackend(CommBackend):
    """
    NCCL 通信后端 - 适配 NVIDIA GPU。

    支持两种模式：
      Mode A: torch.distributed NCCL 后端 (默认，兼容性好)
      Mode B: 直接 libnccl.so C API (CUDA Graph 场景必需)

    典型用法:
        from vllm.distributed.unified_comm.backend import register_backend
        from vllm.distributed.unified_comm.backends.nccl_backend import (
            NCCLBackend,
        )

        # Mode A (默认)
        register_backend(NCCLBackend())

        # Mode B (直接 C API，用于 CUDA Graph)
        register_backend(NCCLBackend(use_direct_nccl=True))
    """

    def __init__(
        self,
        use_direct_nccl: bool = False,
        library_path: str | None = None,
    ):
        """
        Args:
            use_direct_nccl: True 时使用直接 C API 模式 (Mode B)
            library_path: NCCL 库路径 (None 则自动查找 libnccl.so)
        """
        self._use_direct_nccl = use_direct_nccl
        self._library_path = library_path

    @property
    def name(self) -> str:
        return "nccl"

    @property
    def device_type(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        """检查当前环境是否有 CUDA + NCCL"""
        try:
            return torch.cuda.is_available() and dist.is_nccl_available()
        except Exception:
            return False

    # ----------------------------------------------------------
    # 生命周期管理
    # ----------------------------------------------------------

    def init_comm_group(
        self, group_info: CommGroupInfo, config: CommConfig
    ) -> NCCLCommHandle:
        """
        初始化 NCCL 通信组。

        支持两种模式：
        A) 复用已有 ProcessGroup（通过 config.extra 传入）
        B) 新建 ProcessGroup（默认行为）
        """
        existing_device_group = config.extra.get("existing_device_group", None)
        existing_cpu_group = config.extra.get("existing_cpu_group", None)

        if existing_device_group is not None and existing_cpu_group is not None:
            device_group = existing_device_group
            cpu_group = existing_cpu_group
            owns_groups = False
        else:
            device_group = dist.new_group(
                ranks=group_info.ranks,
                backend="nccl",
            )
            cpu_group = dist.new_group(
                ranks=group_info.ranks,
                backend="gloo",
            )
            owns_groups = True

        # [可选] 初始化直接 NCCL C API
        nccl_comm = None
        nccl_lib = None
        if self._use_direct_nccl:
            nccl_comm, nccl_lib = self._init_direct_nccl(
                group_info=group_info,
                cpu_group=cpu_group,
                config=config,
            )

        handle = NCCLCommHandle(
            device_group=device_group,
            cpu_group=cpu_group,
            group_info=group_info,
            device=group_info.device,
            nccl_comm=nccl_comm,
            nccl_lib=nccl_lib,
            use_direct=(self._use_direct_nccl and nccl_comm is not None),
            owns_groups=owns_groups,
        )

        return handle

    def _init_direct_nccl(
        self,
        group_info: CommGroupInfo,
        cpu_group: dist.ProcessGroup,
        config: CommConfig,
    ) -> tuple[Any, Any]:
        """
        初始化直接 NCCL C API 通信。

        复刻 vLLM 上游 PyNcclCommunicator.__init__() 的逻辑：
        1. 加载 libnccl.so
        2. rank 0 生成 ncclUniqueId
        3. 通过 Gloo 广播 unique_id
        4. 所有 rank 调用 ncclCommInitRank
        5. warmup all_reduce
        """
        try:
            from vllm.distributed.device_communicators.pynccl_wrapper import (
                NCCLLibrary,
                buffer_type,
                cudaStream_t,
                ncclDataTypeEnum,
                ncclRedOpTypeEnum,
                ncclUniqueId,
            )
        except ImportError:
            logger.warning(
                "Cannot import pynccl_wrapper. "
                "Direct NCCL C API mode requires vllm internals. "
                "Falling back to torch.distributed mode."
            )
            return None, None

        library_path = config.library_path or self._library_path
        try:
            nccl_lib = NCCLLibrary(library_path)
        except Exception as e:
            logger.warning(
                "Failed to load NCCL library: %s. "
                "Falling back to torch.distributed mode.",
                e,
            )
            return None, None

        device = group_info.device
        logger.info(
            "Initializing direct NCCL comm: rank=%d, world_size=%d, device=%s",
            group_info.rank_in_group,
            group_info.world_size,
            device,
        )

        # 生成 / 广播 UniqueId
        if group_info.rank_in_group == 0:
            unique_id = nccl_lib.ncclGetUniqueId()
            nccl_version = nccl_lib.ncclGetVersion()
            logger.info("NCCL version: %s", nccl_version)
        else:
            unique_id = ncclUniqueId()

        tensor = torch.ByteTensor(list(unique_id.internal))
        dist.broadcast(tensor, src=group_info.ranks[0], group=cpu_group)
        byte_list = tensor.tolist()
        for i, byte_val in enumerate(byte_list):
            unique_id.internal[i] = byte_val

        # 初始化 NCCL Communicator
        with torch.cuda.device(device):
            nccl_comm = nccl_lib.ncclCommInitRank(
                group_info.world_size, unique_id, group_info.rank_in_group
            )

        # Warmup
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            warmup_data = torch.zeros(1, device=device)
            nccl_lib.ncclAllReduce(
                buffer_type(warmup_data.data_ptr()),
                buffer_type(warmup_data.data_ptr()),
                warmup_data.numel(),
                ncclDataTypeEnum.from_torch(warmup_data.dtype),
                ncclRedOpTypeEnum.from_torch(dist.ReduceOp.SUM),
                nccl_comm,
                cudaStream_t(stream.cuda_stream),
            )
            stream.synchronize()
            del warmup_data

        logger.info(
            "Direct NCCL comm initialized for rank %d",
            group_info.rank_in_group,
        )
        return nccl_comm, nccl_lib

    def destroy_comm_group(self, comm_handle: NCCLCommHandle) -> None:
        """销毁通信组，释放所有资源"""
        if comm_handle.nccl_comm is not None and comm_handle.nccl_lib is not None:
            try:
                with torch.cuda.device(comm_handle.device):
                    comm_handle.nccl_lib.ncclCommDestroy(comm_handle.nccl_comm)
            except Exception as e:
                logger.warning("Failed to destroy NCCL comm: %s", e)

        if comm_handle.owns_groups:
            if comm_handle.device_group is not None:
                with contextlib.suppress(Exception):
                    dist.destroy_process_group(comm_handle.device_group)
            if comm_handle.cpu_group is not None:
                with contextlib.suppress(Exception):
                    dist.destroy_process_group(comm_handle.cpu_group)

    # ----------------------------------------------------------
    # 集合通信原语
    # ----------------------------------------------------------

    def all_reduce(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """AllReduce: 所有 rank 归约 tensor，结果写回所有 rank。"""
        if comm_handle.use_direct:
            return self._all_reduce_direct(comm_handle, input_tensor, op, stream)

        # Mode A: torch.distributed — in-place all_reduce
        dist.all_reduce(input_tensor, op=op.to_torch(), group=comm_handle.device_group)
        return input_tensor

    def _all_reduce_direct(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        op: ReduceOp,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 NCCL C API 的 AllReduce"""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert input_tensor.device == comm_handle.device

        out_tensor = torch.empty_like(input_tensor)

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        comm_handle.nccl_lib.ncclAllReduce(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            _reduce_op_to_nccl(op),
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_gather(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """AllGather: 收集所有 rank 的 tensor，拼接返回。"""
        if comm_handle.use_direct:
            return self._all_gather_direct(
                comm_handle, input_tensor, world_size, stream
            )

        # Mode A: torch.distributed
        input_size = list(input_tensor.shape)
        output_size = [input_size[0] * world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=comm_handle.device_group
        )
        return output_tensor

    def _all_gather_direct(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 NCCL C API 的 AllGather"""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert input_tensor.device == comm_handle.device

        input_size = list(input_tensor.shape)
        output_size = [input_size[0] * world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        comm_handle.nccl_lib.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )
        return output_tensor

    def reduce_scatter(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """ReduceScatter: 先归约再分散。"""
        if comm_handle.use_direct:
            return self._reduce_scatter_direct(
                comm_handle, input_tensor, world_size, op, stream
            )

        # Mode A: torch.distributed
        input_size = list(input_tensor.shape)
        assert input_size[0] % world_size == 0, (
            f"ReduceScatter requires dim0 ({input_size[0]}) "
            f"divisible by world_size ({world_size})"
        )
        output_size = [input_size[0] // world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        dist.reduce_scatter_tensor(
            output_tensor,
            input_tensor,
            op=op.to_torch(),
            group=comm_handle.device_group,
        )
        return output_tensor

    def _reduce_scatter_direct(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        op: ReduceOp,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 NCCL C API 的 ReduceScatter"""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert input_tensor.device == comm_handle.device
        input_size = list(input_tensor.shape)
        assert input_size[0] % world_size == 0

        output_size = [input_size[0] // world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        comm_handle.nccl_lib.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            _reduce_op_to_nccl(op),
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )
        return output_tensor

    def broadcast(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """Broadcast: 从 src rank 广播 tensor 到所有 rank。"""
        assert comm_handle.group_info is not None
        if comm_handle.use_direct:
            return self._broadcast_direct(comm_handle, tensor, src, stream)

        # Mode A: torch.distributed
        global_src = comm_handle.group_info.ranks[src]
        output = (
            tensor.clone() if comm_handle.group_info.rank_in_group != src else tensor
        )
        dist.broadcast(output, src=global_src, group=comm_handle.device_group)
        return output

    def _broadcast_direct(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 NCCL C API 的 Broadcast"""
        assert comm_handle.group_info is not None
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert tensor.device == comm_handle.device

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        rank_in_group = comm_handle.group_info.rank_in_group
        if rank_in_group == src:
            sendbuff = buffer_type(tensor.data_ptr())
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()  # NULL
            recvbuff = buffer_type(tensor.data_ptr())

        comm_handle.nccl_lib.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )
        return tensor

    def all_to_all(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
        stream: Any = None,
    ) -> torch.Tensor:
        """All-to-All: 每个 rank 向每个 rank 发送不同块。"""
        assert comm_handle.group_info is not None
        if comm_handle.use_direct:
            return self._all_to_all_direct(
                comm_handle,
                input_tensor,
                scatter_dim,
                gather_dim,
                scatter_sizes,
                gather_sizes,
                stream,
            )

        # Mode A: torch.distributed
        world_size = comm_handle.group_info.world_size

        if scatter_dim < 0:
            scatter_dim += input_tensor.dim()
        if gather_dim < 0:
            gather_dim += input_tensor.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                t.contiguous()
                for t in torch.split(input_tensor, scatter_sizes, scatter_dim)
            ]
            rank = comm_handle.group_info.rank_in_group
            output_list = []
            tensor_shape_base = list(input_list[rank].shape)
            for i in range(world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(
                    torch.empty(
                        tensor_shape,
                        dtype=input_tensor.dtype,
                        device=input_tensor.device,
                    )
                )
        else:
            input_list = [
                t.contiguous()
                for t in torch.tensor_split(input_tensor, world_size, scatter_dim)
            ]
            output_list = [torch.empty_like(input_list[i]) for i in range(world_size)]

        dist.all_to_all(output_list, input_list, group=comm_handle.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    def _all_to_all_direct(
        self,
        comm_handle: NCCLCommHandle,
        input_tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        scatter_sizes: list[int] | None,
        gather_sizes: list[int] | None,
        stream: Any,
    ) -> torch.Tensor:
        """使用 ncclGroupStart/End + ncclSend/ncclRecv 实现 all_to_all。"""
        assert comm_handle.group_info is not None
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert input_tensor.device == comm_handle.device

        world_size = comm_handle.group_info.world_size
        rank = comm_handle.group_info.rank_in_group

        if scatter_dim < 0:
            scatter_dim += input_tensor.dim()
        if gather_dim < 0:
            gather_dim += input_tensor.dim()

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        # 切分 input
        if scatter_sizes is not None:
            input_list = [
                t.contiguous()
                for t in torch.split(input_tensor, scatter_sizes, scatter_dim)
            ]
        else:
            input_list = [
                t.contiguous()
                for t in torch.tensor_split(input_tensor, world_size, scatter_dim)
            ]

        # 准备 output buffers
        if gather_sizes is not None:
            tensor_shape_base = list(input_list[rank].shape)
            output_list = []
            for i in range(world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(
                    torch.empty(
                        tensor_shape,
                        dtype=input_tensor.dtype,
                        device=input_tensor.device,
                    )
                )
        else:
            output_list = [torch.empty_like(input_list[i]) for i in range(world_size)]

        # NCCL Group API
        nccl_dtype = ncclDataTypeEnum.from_torch(input_tensor.dtype)
        cuda_stream = cudaStream_t(stream.cuda_stream)

        comm_handle.nccl_lib.ncclGroupStart()
        for i in range(world_size):
            comm_handle.nccl_lib.ncclSend(
                buffer_type(input_list[i].data_ptr()),
                input_list[i].numel(),
                nccl_dtype,
                i,
                comm_handle.nccl_comm,
                cuda_stream,
            )
            comm_handle.nccl_lib.ncclRecv(
                buffer_type(output_list[i].data_ptr()),
                output_list[i].numel(),
                nccl_dtype,
                i,
                comm_handle.nccl_comm,
                cuda_stream,
            )
        comm_handle.nccl_lib.ncclGroupEnd()

        return torch.cat(output_list, dim=gather_dim).contiguous()

    # ----------------------------------------------------------
    # 点对点通信
    # ----------------------------------------------------------

    def send(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        dst: int,
        stream: Any = None,
    ) -> None:
        """点对点发送。dst: 组内 rank (0-based)"""
        assert comm_handle.group_info is not None
        if comm_handle.use_direct:
            self._send_direct(comm_handle, tensor, dst, stream)
            return

        global_dst = comm_handle.group_info.ranks[dst]
        dist.send(tensor, dst=global_dst, group=comm_handle.device_group)

    def _send_direct(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        dst: int,
        stream: Any,
    ) -> None:
        """直接调用 NCCL C API 的 Send"""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert tensor.device == comm_handle.device

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        dtype = tensor.dtype
        if "float8" in str(dtype):
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(dtype)

        comm_handle.nccl_lib.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            dst,
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """点对点接收。src: 组内 rank (0-based)"""
        assert comm_handle.group_info is not None
        if comm_handle.use_direct:
            self._recv_direct(comm_handle, tensor, src, stream)
            return tensor

        global_src = comm_handle.group_info.ranks[src]
        dist.recv(tensor, src=global_src, group=comm_handle.device_group)
        return tensor

    def _recv_direct(
        self,
        comm_handle: NCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any,
    ) -> None:
        """直接调用 NCCL C API 的 Recv"""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        assert tensor.device == comm_handle.device

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        dtype = tensor.dtype
        if "float8" in str(dtype):
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(dtype)

        comm_handle.nccl_lib.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            src,
            comm_handle.nccl_comm,
            cudaStream_t(stream.cuda_stream),
        )

    # ----------------------------------------------------------
    # 同步
    # ----------------------------------------------------------

    def barrier(self, comm_handle: NCCLCommHandle) -> None:
        """组内同步屏障"""
        dist.barrier(group=comm_handle.device_group)

    def synchronize(self, stream: Any = None) -> None:
        """同步 CUDA stream 上的所有操作。"""
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()

    # ----------------------------------------------------------
    # 高级功能：NCCL Group API (仅 Mode B)
    # ----------------------------------------------------------

    def group_start(self, comm_handle: NCCLCommHandle) -> None:
        """开始一个 NCCL group 操作。"""
        if not comm_handle.use_direct:
            raise RuntimeError(
                "group_start/group_end only available in direct C API mode."
            )
        comm_handle.nccl_lib.ncclGroupStart()

    def group_end(self, comm_handle: NCCLCommHandle) -> None:
        """结束一个 NCCL group 操作。"""
        if not comm_handle.use_direct:
            raise RuntimeError(
                "group_start/group_end only available in direct C API mode."
            )
        comm_handle.nccl_lib.ncclGroupEnd()

    def batch_send_recv(
        self,
        comm_handle: NCCLCommHandle,
        send_list: list[tuple[torch.Tensor, int]],
        recv_list: list[tuple[torch.Tensor, int]],
        stream: Any = None,
    ) -> None:
        """
        批量 Send/Recv 操作（使用 NCCL Group API）。

        Args:
            send_list: [(tensor, dst_rank), ...]
            recv_list: [(tensor, src_rank), ...]
            stream: 可选 CUDA stream
        """
        if not comm_handle.use_direct:
            # Mode A fallback: 逐个调用
            for tensor, dst in send_list:
                self.send(comm_handle, tensor, dst, stream)
            for tensor, src in recv_list:
                self.recv(comm_handle, tensor, src, stream)
            return

        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )

        if stream is None:
            stream = torch.cuda.current_stream(comm_handle.device)

        cuda_stream = cudaStream_t(stream.cuda_stream)

        comm_handle.nccl_lib.ncclGroupStart()

        for tensor, dst in send_list:
            assert tensor.device == comm_handle.device
            dtype = tensor.dtype
            if "float8" in str(dtype):
                nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
            else:
                nccl_dtype = ncclDataTypeEnum.from_torch(dtype)
            comm_handle.nccl_lib.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                nccl_dtype,
                dst,
                comm_handle.nccl_comm,
                cuda_stream,
            )

        for tensor, src in recv_list:
            assert tensor.device == comm_handle.device
            dtype = tensor.dtype
            if "float8" in str(dtype):
                nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
            else:
                nccl_dtype = ncclDataTypeEnum.from_torch(dtype)
            comm_handle.nccl_lib.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                nccl_dtype,
                src,
                comm_handle.nccl_comm,
                cuda_stream,
            )

        comm_handle.nccl_lib.ncclGroupEnd()
