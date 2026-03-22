# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC 连接器模型运行器混入模块。

本模块定义 EC（Encoder Cache）连接器功能混入类，负责：
- 保存编码器缓存到连接器
- 获取已完成的 EC 传输
- 管理 EC 连接器的生命周期

主要类：
- ECConnectorModelRunnerMixin: EC 连接器功能混入类
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# 定义为模型运行器（GPU、TPU）的 EC 连接器功能混入类
class ECConnectorModelRunnerMixin:
    """EC 连接器功能混入类。

    提供 EC 连接器的标准接口，用于管理编码器缓存的传输。
    """

    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ) -> None:
        """保存编码器缓存到连接器（如果有 EC 传输）。

        Args:
            encoder_cache: 编码器缓存字典
            mm_hash: 多模态哈希值
        """
        if not has_ec_transfer():
            logger.debug("没有 EC 传输，请检查")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[set[str] | None, set[str] | None]:
        """获取已完成的 EC 传输。

        Args:
            scheduler_output: 调度器输出

        Returns:
            (已完成的发送请求 ID 集合，已完成的接收请求 ID 集合) 元组
        """
        if has_ec_transfer():
            return get_ec_transfer().get_finished(scheduler_output.finished_req_ids)
        return None, None

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        """获取 EC 连接器输出上下文管理器（如果有 EC 传输）。

        Args:
            scheduler_output: 调度器输出
            encoder_cache: 编码器缓存字典
            **kwargs: 其他参数

        Returns:
            EC 连接器输出上下文管理器
        """
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    # 此上下文管理器必须在活动的前向上下文中使用
    # 它封装了 execute_model 内的整个 EC 连接器生命周期
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        """获取 EC 连接器输出的上下文管理器。

        管理 EC 连接器的完整生命周期：
        1. 绑定连接器元数据
        2. 为 consumer 或 both roles 加载缓存
        3. 获取已完成的传输
        4. 清除连接器元数据

        Args:
            scheduler_output: 调度器输出
            encoder_cache: 编码器缓存字典
            **kwargs: 其他参数

        Yields:
            EC 连接器输出
        """
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # 为 consumer 或 both roles 加载缓存
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            ec_connector.clear_connector_metadata()
