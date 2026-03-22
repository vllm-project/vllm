# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 Worker 模块。

本模块定义了 KV 卸载处理程序的抽象接口和管理类，负责：
- 定义异步数据传输的处理程序接口
- 提供传输结果数据类
- 管理多个处理程序的 Worker 类

主要类：
- TransferResult: 传输结果数据类
- OffloadingHandler: 卸载处理程序抽象基类
- OffloadingWorker: 卸载处理程序管理器

类型定义：
- TransferSpec: 传输规范类型（源规范，目标规范）
- TransferType: 传输类型（源介质，目标介质）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec

# 单个传输规范（源块规范，目标块规范）
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# 传输通过转移类型转发给 worker
TransferType = tuple[str, str]

logger = init_logger(__name__)


@dataclass
class TransferResult:
    """传输结果数据类。

    封装单次传输作业的完成状态和统计信息。

    Attributes:
        job_id: 传输作业的唯一 ID
        success: 传输是否成功
        transfer_size: 传输的数据大小（字节），可选
        transfer_time: 传输耗时（秒），可选
        transfer_type: 传输类型（源介质，目标介质），可选
    """

    job_id: int
    success: bool
    transfer_size: int | None = None  # 字节大小
    transfer_time: float | None = None
    transfer_type: TransferType | None = None


class OffloadingHandler(ABC):
    """卸载处理程序抽象基类。

    用于管理异步 KV 数据传输。

    此类在 worker 中运行，提供以下原语：
        transfer_async() - 发起新的传输作业
        get_finished() - 返回已完成的作业 ID 列表
        wait() - 阻塞等待指定作业完成
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """发起异步 KV 数据传输。

        Args:
            job_id: 唯一的作业 ID，用于在传输完成时通知
            spec: KV 数据传输的（源，目标）规范

        Returns:
            如果传输提交成功则返回 True
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """获取自上次调用以来完成的传输。

        Returns:
            传输结果列表，包含作业 ID 和成功状态
        """
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """等待作业完成（阻塞）。

        Args:
            job_ids: 要等待的作业 ID 集合
        """


class OffloadingWorker:
    """卸载处理程序管理器。

    使用多个 OffloadingHandler 管理异步 KV 数据传输。

    此类在 worker 中运行，根据传输类型委托给注册的
    OffloadingHandler 之一来发起异步 KV 数据传输请求。

    此类提供以下原语：
        register_handler() - 注册新的处理程序以处理特定传输类型
        transfer_async() - 使用注册的处理程序之一发起新的传输作业
        get_finished() - 从所有处理程序获取已完成的作业 ID

    Attributes:
        handlers: 注册的卸载处理程序集合
        transfer_type_to_handler: 传输类型到处理程序的映射
    """

    def __init__(self):
        """初始化卸载 worker。"""
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """注册新的处理程序。

        Args:
            src_cls: 处理程序处理的源传输类型
            dst_cls: 处理程序处理的目标传输类型
            handler: 处理传输的处理程序

        Raises:
            AssertionError: 如果该传输类型已注册了处理程序
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """发起异步 KV 数据传输。

        根据传输类型（源介质 -> 目标介质）查找对应的处理程序，
        并委托给它发起传输。

        Args:
            job_id: 唯一的作业 ID，用于在传输完成时通知
            spec: KV 数据传输的（源，目标）规范

        Returns:
            如果传输提交成功则返回 True
        """
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None
        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        return success

    def get_finished(self) -> list[TransferResult]:
        """获取自上次调用以来完成的传输。

        从所有注册的处理程序中收集完成的传输结果。

        Returns:
            TransferResult 列表
        """
        finished = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """等待作业完成（阻塞）。

        委托给所有注册的处理程序等待指定的作业 ID。

        Args:
            job_ids: 要等待的作业 ID 集合
        """
        for handler in self.handlers:
            handler.wait(job_ids)
