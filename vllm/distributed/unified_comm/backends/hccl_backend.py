# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL backend - Huawei Ascend NPU adapter for ``CommBackend``.

Adapts the Huawei Collective Communications Library (HCCL) to the
unified :class:`CommBackend` interface. The low-level C bindings are
reused from the existing ``pyhccl_wrapper`` shipped with vllm-hust /
vllm-ascend-hust.

Two execution modes are supported:
  Mode A (``torch.distributed``):
      Uses the HCCL backend through ``torch.distributed``. Recommended
      because it is the most compatible.
  Mode B (direct C API):
      Calls into ``libhccl.so`` directly via the ``pyhccl_wrapper``.
      Useful for latency-sensitive paths because it avoids implicit
      ``torch.distributed`` overhead and gives finer-grained stream
      control. Enabled with ``UNIFIED_COMM_USE_DIRECT_HCCL=1``.
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

# 统一层 ReduceOp → HCCL C API 归约操作码
_REDUCE_OP_TO_HCCL_INT = {
    ReduceOp.SUM: 0,  # hcclSum
    ReduceOp.PRODUCT: 1,  # hcclProd
    ReduceOp.MAX: 2,  # hcclMax
    ReduceOp.MIN: 3,  # hcclMin
}


def _reduce_op_to_hccl(op: ReduceOp) -> int:
    """将统一 ReduceOp 转为 HCCL C API 的 int 枚举值"""
    if op == ReduceOp.AVG:
        raise ValueError("HCCL does not natively support AVG, use SUM + divide")
    return _REDUCE_OP_TO_HCCL_INT[op]


# ============================================================
# HCCL 数据类型映射（仅 Mode B 需要）
# ============================================================

# PyTorch dtype → HCCL hcclDataType_t (int)
# 注：具体映射值参考 HCCL 官方头文件
_TORCH_DTYPE_TO_HCCL = {
    torch.int8: 1,  # HCCL_DATA_TYPE_INT8
    torch.int16: 2,  # HCCL_DATA_TYPE_INT16
    torch.int32: 3,  # HCCL_DATA_TYPE_INT32
    torch.float16: 4,  # HCCL_DATA_TYPE_FP16
    torch.float32: 5,  # HCCL_DATA_TYPE_FP32
    torch.int64: 6,  # HCCL_DATA_TYPE_INT64
    torch.float64: 7,  # HCCL_DATA_TYPE_FP64
    torch.bfloat16: 9,  # HCCL_DATA_TYPE_BFP16
}


def _import_pyhccl_wrapper():
    """
    Try multiple known locations for the pyhccl wrapper module.

    Different vllm forks/branches put it under different namespaces:
    1. vllm.distributed.device_communicators.pyhccl_wrapper (if vendored
       into vllm-hust main repo)
    2. vllm_ascend.distributed.device_communicators.pyhccl_wrapper (lives
       in vllm-ascend-hust plugin)

    Returns the imported module on success, None on failure.
    """
    import importlib

    for mod_path in (
        "vllm.distributed.device_communicators.pyhccl_wrapper",
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper",
    ):
        try:
            return importlib.import_module(mod_path)
        except ImportError:
            continue
    return None


def _dtype_to_hccl(dtype: torch.dtype) -> int:
    """将 PyTorch dtype 转为 HCCL 数据类型 int"""
    if dtype in _TORCH_DTYPE_TO_HCCL:
        return _TORCH_DTYPE_TO_HCCL[dtype]
    # float8 变体统一映射为 int8 传输
    if "float8" in str(dtype):
        return _TORCH_DTYPE_TO_HCCL[torch.int8]
    raise ValueError(f"Unsupported dtype for HCCL: {dtype}")


# ============================================================
# HCCL 通信句柄
# ============================================================


@dataclass
class HCCLCommHandle:
    """
    HCCL 通信组句柄，封装所有需要的状态。

    对应关系：
      - device_group: torch.distributed HCCL ProcessGroup (Mode A 使用)
      - cpu_group: torch.distributed Gloo ProcessGroup (辅助 CPU 通信，如广播 unique_id)
      - hccl_comm: hcclComm_t 原始句柄 (Mode B 使用)
      - hccl_lib: HCCLLibrary 实例 (Mode B 使用)
      - owns_groups: 是否拥有 ProcessGroup 的生命周期
        （False 表示外部传入，不由我们销毁）
    """

    device_group: dist.ProcessGroup | None = None
    cpu_group: dist.ProcessGroup | None = None
    group_info: CommGroupInfo | None = None
    device: torch.device | None = None
    hccl_comm: Any = None  # hcclComm_t (ctypes.c_void_p)
    hccl_lib: Any = None  # HCCLLibrary instance
    use_direct: bool = False  # 是否使用直接 C API 模式
    owns_groups: bool = True  # 是否拥有 PG 生命周期管理权


# ============================================================
# HCCLBackend 实现
# ============================================================


class HCCLBackend(CommBackend):
    """
    HCCL 通信后端 - 适配华为 Ascend NPU。

    支持两种模式：
      Mode A: torch.distributed HCCL 后端 (默认，兼容性好，支持所有集合原语)
      Mode B: 直接 libhccl.so C API (低延迟，目前支持 AllReduce / Broadcast)

    典型用法:
        from vllm.distributed.unified_comm.backend import register_backend
        from vllm.distributed.unified_comm.backends.hccl_backend import HCCLBackend

        # Mode A (默认)
        register_backend(HCCLBackend())

        # Mode B (直接 C API)
        register_backend(HCCLBackend(use_direct_hccl=True))

    与 NCCL 后端的主要区别：
      1. 设备管理使用 torch.npu 而非 torch.cuda
      2. Stream 通过 torch_npu 的 current_stream 获取
      3. ProcessGroup 后端名称为 "hccl" 而非 "nccl"
      4. HCCL 不原生支持 AVG 归约操作
      5. All-to-All 和点对点通信目前仅支持 Mode A (torch.distributed)
    """

    def __init__(self, use_direct_hccl: bool = False, library_path: str | None = None):
        """
        Args:
            use_direct_hccl: True 时使用直接 C API 模式 (Mode B)
            library_path: HCCL 库路径 (None 则自动查找)
        """
        self._use_direct_hccl = use_direct_hccl
        self._library_path = library_path

    @property
    def name(self) -> str:
        return "hccl"

    @property
    def device_type(self) -> str:
        return "npu"

    def is_available(self) -> bool:
        """检查当前环境是否有 NPU + HCCL"""
        try:
            import torch_npu  # noqa: F401

            return torch.npu.is_available()
        except (ImportError, AttributeError):
            return False

    # ----------------------------------------------------------
    # 生命周期管理
    # ----------------------------------------------------------

    def init_comm_group(
        self, group_info: CommGroupInfo, config: CommConfig
    ) -> HCCLCommHandle:
        """
        初始化 HCCL 通信组。

        支持两种模式：
        A) 复用已有 ProcessGroup（通过 config.extra 传入）—— 推荐
           config.extra["existing_device_group"] = <已创建的 HCCL PG>
           config.extra["existing_cpu_group"] = <已创建的 Gloo PG>

        B) 新建 ProcessGroup（默认行为）

        Args:
            group_info: 通信组描述
            config: 通信配置
                - config.extra["group_name"]: HCCL ProcessGroup 名称 (可选)
                - config.extra["existing_device_group"]: 可选，已有的 HCCL PG
                - config.extra["existing_cpu_group"]: 可选，已有的 Gloo PG

        Returns:
            HCCLCommHandle: 通信句柄
        """
        import torch_npu  # noqa: F401 - 确保 torch_npu 已加载

        group_name = config.extra.get("group_name", "unified_comm")

        # 优先复用已有的 ProcessGroup，避免重复创建导致死锁
        existing_device_group = config.extra.get("existing_device_group", None)
        existing_cpu_group = config.extra.get("existing_cpu_group", None)

        if existing_device_group is not None and existing_cpu_group is not None:
            # 模式 A：复用已有 PG
            device_group = existing_device_group
            cpu_group = existing_cpu_group
            owns_groups = False
        else:
            # 模式 B：新建 PG
            device_group = self._create_device_group(group_info.ranks, group_name)
            cpu_group = dist.new_group(ranks=group_info.ranks, backend="gloo")
            owns_groups = True

        # [可选] 初始化直接 HCCL C API
        hccl_comm = None
        hccl_lib = None
        if self._use_direct_hccl:
            hccl_comm, hccl_lib = self._init_direct_hccl(
                group_info=group_info,
                cpu_group=cpu_group,
                config=config,
            )

        handle = HCCLCommHandle(
            device_group=device_group,
            cpu_group=cpu_group,
            group_info=group_info,
            device=group_info.device,
            hccl_comm=hccl_comm,
            hccl_lib=hccl_lib,
            use_direct=self._use_direct_hccl and hccl_comm is not None,
            owns_groups=owns_groups,
        )

        logger.info(
            "HCCL comm group initialized: rank=%d, world_size=%d, mode=%s",
            group_info.rank_in_group,
            group_info.world_size,
            "direct" if handle.use_direct else "torch.distributed",
        )

        return handle

    def _create_device_group(
        self, ranks: list[int], group_name: str
    ) -> dist.ProcessGroup:
        """
        创建 HCCL ProcessGroup。

        尝试使用项目中已有的 HCCL pg_options 逻辑（如果可用），
        否则使用默认参数创建。
        """
        try:
            # 尝试使用 vllm 中可能存在的 HCCL 配置工具
            from vllm.platforms import current_platform

            pg_options = current_platform.create_hccl_pg_options(group_name)
            device_group = dist.new_group(
                ranks=ranks,
                backend="hccl",
                pg_options=pg_options,
            )
        except (ImportError, AttributeError):
            # Fallback: 直接创建，不带额外选项
            device_group = dist.new_group(
                ranks=ranks,
                backend="hccl",
            )

        return device_group

    def _init_direct_hccl(
        self,
        group_info: CommGroupInfo,
        cpu_group: dist.ProcessGroup,
        config: CommConfig,
    ) -> tuple[Any, Any]:
        """
        初始化直接 HCCL C API 通信。

        复刻自 PyHcclCommunicator.__init__() 的逻辑：
        1. 加载 libhccl.so (通过 HCCLLibrary)
        2. rank 0 生成 hcclUniqueId
        3. 通过 Gloo (cpu_group) 广播 unique_id 到所有 rank
        4. 所有 rank 调用 hcclCommInitRank
        5. 做一次 warmup all_reduce
        """
        try:
            mod = _import_pyhccl_wrapper()
            if mod is None:
                raise ImportError("pyhccl_wrapper not found")
            HCCLLibrary = mod.HCCLLibrary
            aclrtStream_t = mod.aclrtStream_t
            buffer_type = mod.buffer_type
            hcclDataTypeEnum = mod.hcclDataTypeEnum
            hcclUniqueId = mod.hcclUniqueId
        except ImportError:
            logger.warning(
                "Cannot import pyhccl_wrapper from any known location "
                "(vllm.distributed.device_communicators / "
                "vllm_ascend.distributed.device_communicators). "
                "Direct HCCL C API mode requires the HCCL wrapper to be available. "
                "Falling back to torch.distributed mode."
            )
            return None, None

        # 确定库路径
        library_path = config.library_path or self._library_path
        try:
            # HCCLLibrary 内部会自动 find_hccl_library() 如果传 None
            hccl_lib = HCCLLibrary(library_path)
        except Exception as e:
            logger.warning(
                "Failed to load HCCL library: %s. "
                "Falling back to torch.distributed mode.",
                e,
            )
            return None, None

        device = group_info.device
        logger.info(
            "Initializing direct HCCL comm: rank=%d, world_size=%d, device=%s",
            group_info.rank_in_group,
            group_info.world_size,
            device,
        )

        # --- 生成 / 广播 UniqueId ---
        if group_info.rank_in_group == 0:
            with torch.npu.device(device):
                unique_id = hccl_lib.hcclGetUniqueId()
        else:
            unique_id = hcclUniqueId()

        # 将 unique_id 序列化为 ByteTensor，通过 Gloo cpu_group 广播
        # hcclUniqueId.internal 是 ctypes.c_byte * 4108
        tensor = torch.ByteTensor(list(unique_id.internal))
        # 注意：dist.broadcast 的 src 需要全局 rank
        dist.broadcast(tensor, src=group_info.ranks[0], group=cpu_group)
        byte_list = tensor.tolist()
        for i, byte_val in enumerate(byte_list):
            unique_id.internal[i] = byte_val

        # --- 初始化 HCCL Communicator ---
        with torch.npu.device(device):
            hccl_comm = hccl_lib.hcclCommInitRank(
                group_info.world_size, unique_id, group_info.rank_in_group
            )

        # --- Warmup: 一次 small AllReduce 确保通信链路就绪 ---
        with torch.npu.device(device):
            stream = torch.npu.current_stream(device)
            warmup_data = torch.zeros(1, device=device)
            hccl_lib.hcclAllReduce(
                buffer_type(warmup_data.data_ptr()),
                buffer_type(warmup_data.data_ptr()),  # in-place
                warmup_data.numel(),
                hcclDataTypeEnum.from_torch(warmup_data.dtype),
                0,  # hcclSum
                hccl_comm,
                aclrtStream_t(stream.npu_stream),
            )
            stream.synchronize()
            del warmup_data

        logger.info(
            "Direct HCCL comm initialized and warmed up for rank %d",
            group_info.rank_in_group,
        )
        return hccl_comm, hccl_lib

    def destroy_comm_group(self, comm_handle: HCCLCommHandle) -> None:
        """销毁通信组，释放所有资源"""
        # 销毁直接 HCCL comm (如果有)
        if comm_handle.hccl_comm is not None and comm_handle.hccl_lib is not None:
            try:
                comm_handle.hccl_lib.hcclCommDestroy(comm_handle.hccl_comm)
            except Exception as e:
                logger.warning("Failed to destroy HCCL comm: %s", e)

        # 只有当我们拥有 PG 生命周期时才销毁
        # 如果是外部传入的 PG（owns_groups=False），由外部管理生命周期
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
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        AllReduce: 所有 rank 归约 tensor，结果写回所有 rank。

        - Mode B (direct): 调用 libhccl.so hcclAllReduce，返回新 tensor
        - Mode A (torch.distributed): 调用 dist.all_reduce (in-place)

        注意：Mode A 采用 in-place 语义，与 vLLM GroupCoordinator.all_reduce 一致。
        """
        if comm_handle.use_direct:
            return self._all_reduce_direct(comm_handle, input_tensor, op, stream)

        # Mode A: torch.distributed — in-place all_reduce
        dist.all_reduce(input_tensor, op=op.to_torch(), group=comm_handle.device_group)
        return input_tensor

    def _all_reduce_direct(
        self,
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        op: ReduceOp,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 HCCL C API 的 AllReduce（对齐 PyHcclCommunicator.all_reduce）"""
        _pyhccl_mod = _import_pyhccl_wrapper()
        aclrtStream_t = _pyhccl_mod.aclrtStream_t
        buffer_type = _pyhccl_mod.buffer_type
        hcclDataTypeEnum = _pyhccl_mod.hcclDataTypeEnum

        # 设备检查
        assert input_tensor.device == comm_handle.device, (
            f"HCCL communicator bound to {comm_handle.device}, "
            f"but input tensor is on {input_tensor.device}"
        )

        out_tensor = torch.empty_like(input_tensor)

        if stream is None:
            stream = torch.npu.current_stream(comm_handle.device)

        comm_handle.hccl_lib.hcclAllReduce(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            input_tensor.numel(),
            hcclDataTypeEnum.from_torch(input_tensor.dtype),
            _reduce_op_to_hccl(op),
            comm_handle.hccl_comm,
            aclrtStream_t(stream.npu_stream),
        )
        return out_tensor

    def all_gather(
        self,
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        AllGather: 收集所有 rank 的 tensor，拼接返回。

        输出 shape: [world_size * input_dim0, ...]

        注：目前 HCCL 直接 C API 未完整导出 AllGather，统一走 torch.distributed。
        """
        if comm_handle.use_direct:
            return self._all_gather_direct(
                comm_handle, input_tensor, world_size, stream
            )

        # Mode A: torch.distributed
        input_size = list(input_tensor.shape)
        output_size = [input_size[0] * world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_tensor.dtype, device=input_tensor.device
        )
        dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=comm_handle.device_group
        )
        return output_tensor

    def _all_gather_direct(
        self,
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        stream: Any,
    ) -> torch.Tensor:
        """
        直接调用 HCCL C API 的 AllGather。

        如果 HCCL wrapper 不支持此操作，回退到 torch.distributed。
        """
        try:
            _pyhccl_mod = _import_pyhccl_wrapper()
            if _pyhccl_mod is None:
                raise ImportError("pyhccl_wrapper not found")
            aclrtStream_t = _pyhccl_mod.aclrtStream_t
            buffer_type = _pyhccl_mod.buffer_type
            hcclDataTypeEnum = _pyhccl_mod.hcclDataTypeEnum
        except ImportError:
            # Fallback to torch.distributed
            return self.all_gather.__wrapped__(  # type: ignore[attr-defined]
                self, comm_handle, input_tensor, world_size, None
            )

        assert input_tensor.device == comm_handle.device

        input_size = list(input_tensor.shape)
        output_size = [input_size[0] * world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if stream is None:
            stream = torch.npu.current_stream(comm_handle.device)

        try:
            # hcclAllGather 的 count 参数是 sendcount (每个 rank 贡献的元素数)
            comm_handle.hccl_lib.hcclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                hcclDataTypeEnum.from_torch(input_tensor.dtype),
                comm_handle.hccl_comm,
                aclrtStream_t(stream.npu_stream),
            )
        except (AttributeError, NotImplementedError):
            # 如果 HCCL 版本不支持 AllGather C API，回退到 torch.distributed
            logger.debug(
                "HCCL C API AllGather not available, falling back to torch.distributed"
            )
            dist.all_gather_into_tensor(
                output_tensor, input_tensor, group=comm_handle.device_group
            )

        return output_tensor

    def reduce_scatter(
        self,
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        ReduceScatter: 先归约再分散，每个 rank 得到 1/world_size 的结果。

        输入 shape: [N, ...]，要求 N % world_size == 0
        输出 shape: [N // world_size, ...]

        注：目前统一走 torch.distributed。HCCL C API 的 ReduceScatter 在未来版本中支持。
        """
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
            output_size, dtype=input_tensor.dtype, device=input_tensor.device
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
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        world_size: int,
        op: ReduceOp,
        stream: Any,
    ) -> torch.Tensor:
        """
        直接调用 HCCL C API 的 ReduceScatter。

        如果 HCCL wrapper 不支持此操作，回退到 torch.distributed。
        """
        try:
            _pyhccl_mod = _import_pyhccl_wrapper()
            if _pyhccl_mod is None:
                raise ImportError("pyhccl_wrapper not found")
            aclrtStream_t = _pyhccl_mod.aclrtStream_t
            buffer_type = _pyhccl_mod.buffer_type
            hcclDataTypeEnum = _pyhccl_mod.hcclDataTypeEnum
        except ImportError:
            input_size = list(input_tensor.shape)
            output_size = [input_size[0] // world_size] + input_size[1:]
            output_tensor = torch.empty(
                output_size, dtype=input_tensor.dtype, device=input_tensor.device
            )
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                op=op.to_torch(),
                group=comm_handle.device_group,
            )
            return output_tensor

        assert input_tensor.device == comm_handle.device
        input_size = list(input_tensor.shape)
        assert input_size[0] % world_size == 0

        output_size = [input_size[0] // world_size] + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if stream is None:
            stream = torch.npu.current_stream(comm_handle.device)

        try:
            # hcclReduceScatter 的 count 参数是 recvcount
            comm_handle.hccl_lib.hcclReduceScatter(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                output_tensor.numel(),
                hcclDataTypeEnum.from_torch(input_tensor.dtype),
                _reduce_op_to_hccl(op),
                comm_handle.hccl_comm,
                aclrtStream_t(stream.npu_stream),
            )
        except (AttributeError, NotImplementedError):
            logger.debug(
                "HCCL C API ReduceScatter not available, "
                "falling back to torch.distributed"
            )
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                op=op.to_torch(),
                group=comm_handle.device_group,
            )

        return output_tensor

    def broadcast(
        self,
        comm_handle: HCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        Broadcast: 从 src rank 广播 tensor 到所有 rank。

        注意: src 是组内 rank (0-based)，不是全局 rank。

        - Mode B (direct): 调用 libhccl.so hcclBroadcast (in-place)
        - Mode A (torch.distributed): 调用 dist.broadcast
        """
        assert comm_handle.group_info is not None
        if comm_handle.use_direct:
            return self._broadcast_direct(comm_handle, tensor, src, stream)

        # Mode A: torch.distributed
        # dist.broadcast 需要全局 rank
        global_src = comm_handle.group_info.ranks[src]
        dist.broadcast(tensor, src=global_src, group=comm_handle.device_group)
        return tensor

    def _broadcast_direct(
        self,
        comm_handle: HCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any,
    ) -> torch.Tensor:
        """直接调用 HCCL C API 的 Broadcast（对齐 PyHcclCommunicator.broadcast）"""
        _pyhccl_mod = _import_pyhccl_wrapper()
        aclrtStream_t = _pyhccl_mod.aclrtStream_t
        buffer_type = _pyhccl_mod.buffer_type
        hcclDataTypeEnum = _pyhccl_mod.hcclDataTypeEnum

        assert tensor.device == comm_handle.device, (
            f"HCCL communicator bound to {comm_handle.device}, "
            f"but tensor is on {tensor.device}"
        )

        if stream is None:
            stream = torch.npu.current_stream(comm_handle.device)

        comm_handle.hccl_lib.hcclBroadcast(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            src,  # HCCL Broadcast 接受组内 rank
            comm_handle.hccl_comm,
            aclrtStream_t(stream.npu_stream),
        )
        return tensor

    def all_to_all(
        self,
        comm_handle: HCCLCommHandle,
        input_tensor: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        All-to-All: 每个 rank 向每个 rank 发送不同块。

        支持均匀切分和非均匀切分 (通过 scatter_sizes/gather_sizes)。

        注：HCCL 目前仅通过 torch.distributed 支持 all_to_all，
        无直接 C API 路径。对齐 NPUCommunicator.all_to_all() 的逻辑。
        """
        assert comm_handle.group_info is not None
        world_size = comm_handle.group_info.world_size

        # 处理负维度
        if scatter_dim < 0:
            scatter_dim += input_tensor.dim()
        if gather_dim < 0:
            gather_dim += input_tensor.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            # 非均匀切分
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
            # 均匀切分
            input_list = [
                t.contiguous()
                for t in torch.tensor_split(input_tensor, world_size, scatter_dim)
            ]
            output_list = [torch.empty_like(input_list[i]) for i in range(world_size)]

        dist.all_to_all(output_list, input_list, group=comm_handle.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    # ----------------------------------------------------------
    # 点对点通信
    # ----------------------------------------------------------

    def send(
        self,
        comm_handle: HCCLCommHandle,
        tensor: torch.Tensor,
        dst: int,
        stream: Any = None,
    ) -> None:
        """
        点对点发送。

        dst: 组内 rank (0-based)

        注：目前 HCCL 的 P2P 统一通过 torch.distributed 完成。
        HCCL C API 层面的 send/recv 在 CANN 版本 >= 7.0 中支持。
        """
        assert comm_handle.group_info is not None
        global_dst = comm_handle.group_info.ranks[dst]
        dist.send(tensor, dst=global_dst, group=comm_handle.device_group)

    def recv(
        self,
        comm_handle: HCCLCommHandle,
        tensor: torch.Tensor,
        src: int,
        stream: Any = None,
    ) -> torch.Tensor:
        """
        点对点接收。

        src: 组内 rank (0-based)
        tensor: 预分配的接收 buffer
        """
        assert comm_handle.group_info is not None
        global_src = comm_handle.group_info.ranks[src]
        dist.recv(tensor, src=global_src, group=comm_handle.device_group)
        return tensor

    # ----------------------------------------------------------
    # 同步
    # ----------------------------------------------------------

    def barrier(self, comm_handle: HCCLCommHandle) -> None:
        """组内同步屏障"""
        dist.barrier(group=comm_handle.device_group)

    def synchronize(self, stream: Any = None) -> None:
        """
        同步 NPU stream 上的所有操作。

        如果不传 stream，则同步当前设备的默认 stream。
        """
        if stream is not None:
            stream.synchronize()
        else:
            torch.npu.synchronize()

    # ----------------------------------------------------------
    # 高级功能：批量 Send/Recv (未来扩展)
    # ----------------------------------------------------------

    def batch_send_recv(
        self,
        comm_handle: HCCLCommHandle,
        send_list: list[tuple[torch.Tensor, int]],
        recv_list: list[tuple[torch.Tensor, int]],
        stream: Any = None,
    ) -> None:
        """
        批量 Send/Recv 操作。

        当前实现：逐个调用 send/recv。
        未来（CANN >= 7.0）可升级为使用 hcclGroupStart/hcclGroupEnd 合并操作。

        Args:
            send_list: [(tensor, dst_rank), ...] 要发送的 tensor 和目标 rank 列表
            recv_list: [(tensor, src_rank), ...] 要接收的 tensor 和来源 rank 列表
            stream: 可选 NPU stream
        """
        for tensor, dst in send_list:
            self.send(comm_handle, tensor, dst, stream)
        for tensor, src in recv_list:
            self.recv(comm_handle, tensor, src, stream)
