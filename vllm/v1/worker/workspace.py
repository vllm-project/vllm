# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""工作区（Workspace）管理模块。

本模块实现了 DBO（Dual Batch Overlap）执行的工作区内存管理，负责：
- 管理 GPU 工作区缓冲区分配
- 支持微批次（micro-batching）的内存隔离
- 提供工作区锁定机制以防止运行时增长
- 优化内存分配效率

主要类：
- WorkspaceManager: 工作区管理器

使用示例：
    # 初始化
    init_workspace_manager(device, num_ubatches=2)

    # 获取工作区（自动扩展）
    workspace = current_workspace_manager().get_simultaneous(
        ((1024,), torch.float32),
        ((512,), torch.int32),
    )

    # 锁定工作区（防止进一步增长）
    lock_workspace()
"""

import inspect
import os
from itertools import accumulate
from math import prod

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """计算张量所需字节数。

    Args:
        shape: 张量形状
        dtype: 数据类型

    Returns:
        字节数
    """
    return prod(shape) * dtype.item_size


# 常量
_MB = 1024**2
_GiB = 1024**3

# 全局工作区管理器实例
_manager: "WorkspaceManager | None" = None


class WorkspaceManager:
    """工作区分配管理器。

    管理 DBO（Dual Batch Overlap）执行的工作区缓冲区。
    可以锁定以防止在执行期间进一步增长。

    Attributes:
        _device: 设备
        _num_ubatches: 微批次数量
        _current_workspaces: 当前工作区列表（每个 ubatch 一个）
        _locked: 是否已锁定
    """

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        """初始化工作区管理器。

        Args:
            device: 设备
            num_ubatches: 微批次数量，默认为 1
        """
        self._device = device
        # 基于配置缓存微批次数量（默认为 1）
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        self._current_workspaces: list[torch.Tensor | None] = [None, None]
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        """获取工作区大小（字节）。

        Args:
            workspace: 工作区张量

        Returns:
            字节数
        """
        if workspace is None:
            return 0
        return workspace.numel() * workspace.element_size()

    def lock(self) -> None:
        """锁定工作区以防止进一步增长。

        锁定后，任何尝试分配更大型区的请求都会引发断言错误。
        这确保在执行期间工作区大小是固定的。
        """
        self._locked = True
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] 工作区已锁定。当前大小：%s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def unlock(self) -> None:
        """解锁工作区以允许增长。

        这用于弹性 EP 扩展，当专家数量变化导致工作区大小需要增长时。
        """
        self._locked = False
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] 工作区已解锁。当前大小：%s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        """检查工作区是否已锁定。

        Returns:
            是否已锁定
        """
        return self._locked

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """从单个分配中同时获取多个工作区张量。

        Args:
            *shapes_and_dtypes: 一个或多个 (shape, dtype) 元组

        Returns:
            工作区缓冲区的张量视图列表，每个 shape/dtype 对应对应一个
        """
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # 使用 itertools.accumulate 计算累积偏移量
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        current_workspace = self._ensure_workspace_size(total_bytes)

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        """确保工作区已分配且足够大，返回当前工作区。

        Args:
            required_bytes: 所需字节数

        Returns:
            当前工作区张量
        """
        ubatch_id = dbo_current_ubatch_id()
        current_workspace = self._current_workspaces[ubatch_id]
        current_size = self._workspace_size_bytes(current_workspace)

        if current_size < required_bytes:

            def get_caller_info() -> str:
                """查找第一个在 WorkspaceManager 外部的帧。"""
                curr_frame = inspect.currentframe()
                if curr_frame is None:
                    return "unknown"
                # 向上遍历堆栈，跳过 WorkspaceManager 帧
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: 这只捕获实例方法（self），缺少 classmethods 和 staticmethods
                    # 一旦 Python 3.11+ 是最小支持版本，使用 co_qualname：
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        curr_frame = curr_frame.f_back
                        continue
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                return "unknown"

            if self._locked:
                raise AssertionError(
                    f"工作区已锁定，但来自 '{get_caller_info()}' 的分配需要 "
                    f"{required_bytes / _MB:.2f} MB，当前大小为 "
                    f"{current_size / _MB:.2f} MB。"
                    "锁定后不允许工作区增长。"
                )

            # 为所有 ubatch 分配工作区
            for ubatch_id in range(self._num_ubatches):
                current_workspace = self._current_workspaces[ubatch_id]
                if (
                    current_workspace is None
                    or self._workspace_size_bytes(current_workspace) < required_bytes
                ):
                    # 在分配新张量之前删除旧张量，以避免 resize_() 的内存峰值
                    # resize_() 在释放旧内存之前分配新内存，可能导致 OOM
                    # 必须先清除列表引用，因为局部变量只是引用的副本
                    self._current_workspaces[ubatch_id] = None
                    del current_workspace
                    self._current_workspaces[ubatch_id] = torch.empty(
                        (required_bytes,), dtype=torch.uint8, device=self._device
                    )

            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] 从 '%s' 调整工作区大小：%.2f MB -> "
                    "%.2f MB (%d 个微批次，总内存 %.2f MB)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    self._num_ubatches,
                    required_bytes * self._num_ubatches / _MB,
                )

            current_workspace = self._current_workspaces[dbo_current_ubatch_id()]

        return current_workspace


def is_workspace_manager_initialized() -> bool:
    """检查工作区管理器是否已初始化。

    Returns:
        如果已初始化返回 True，否则返回 False
    """
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    """获取当前工作区管理器实例。

    Returns:
        工作区管理器实例

    Raises:
        AssertionError: 如果工作区管理器未初始化
    """
    assert _manager is not None, (
        "WorkspaceManager 未初始化。在使用任何工作区函数之前"
        "请调用 init_workspace_manager() 并传入设备。"
    )
    return _manager


def init_workspace_manager(
    device: torch.device, num_ubatches: int | None = None
) -> None:
    """使用设备初始化工作区管理器。

    必须在使用任何工作区函数之前调用。通常在 GPUModelRunner.__init__ 中调用。

    Args:
        device: 要分配工作区的设备
        num_ubatches: 微批次数量，默认为 1
    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager 已经在设备 %s 上初始化，"
            "正在重新初始化到设备 %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, num_ubatches)


def lock_workspace() -> None:
    """锁定工作区以防止进一步增长。

    调用此函数后，任何尝试分配比当前大小更大的工作区的请求
    都会引发 AssertionError。这确保在执行期间工作区大小是固定的，
    并防止在热路径中出现意外的内存分配。

    示例：
        # 初始化期间
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # 在热身/分析后锁定
        lock_workspace()

        # 现在所有 get_workspace 调用必须适应预分配的大小
    """
    current_workspace_manager().lock()


def unlock_workspace() -> None:
    """解锁工作区以允许增长。

    这用于弹性 EP 扩展，当专家数量变化导致工作区大小需要增长时。
    在扩展操作完成后，应再次调用 lock_workspace() 以防止意外分配。
    """
    current_workspace_manager().unlock()


def reset_workspace_manager() -> None:
    """将工作区管理器重置为未初始化状态。

    这主要用于测试目的，允许测试干净地重新初始化工作区管理器。
    """
    global _manager
    _manager = None
