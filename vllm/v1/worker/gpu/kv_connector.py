# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 连接器模块（GPU 版本）。

本模块提供 KV 连接器接口和实现，负责：
- 管理 KV 缓存的传输（加载和保存）
- 与分布式 KV 传输组协作
- 处理抢占和块错误

主要类：
- KVConnector: KV 连接器基类（无操作）
- ActiveKVConnector: 活动 KV 连接器实现
"""
import copy
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class KVConnector:
    """KV 连接器基类（无操作）。

    GPUModelRunner 使用的 KVConnector 接口。
    默认实现所有方法为空操作。
    """

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        """前向传播前的钩子。"""
        pass

    def post_forward(
        self, scheduler_output: "SchedulerOutput", wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        """前向传播后的钩子。

        Args:
            scheduler_output: 调度器输出
            wait_for_save: 是否等待保存完成

        Returns:
            KV 连接器输出（如果有）
        """
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        """无前向传播时的钩子。

        Args:
            scheduler_output: 调度器输出

        Returns:
            空的模型运行器输出
        """
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        """设置连接器是否被禁用。

        Args:
            disabled: 是否禁用
        """
        pass


class ActiveKVConnector(KVConnector):
    """活动 KV 连接器实现。

    负责实际的 KV 缓存传输操作，包括：
    - 注册 KV 缓存
    - 在前向传播前加载 KV 缓存
    - 在前向传播后保存 KV 缓存
    - 处理抢占和块错误

    Attributes:
        vllm_config: vLLM 配置
        kv_connector: KV 传输组
        _disabled: 是否被禁用
    """

    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        """初始化活动 KV 连接器。

        Args:
            vllm_config: vLLM 配置
            kv_caches_dict: KV 缓存字典
        """
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # 注册 KV 缓存到 KV 连接器（如果适用）
        # TODO: 支持 cross_layers_kv_cache
        # (见 https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        """前向传播前的钩子，启动 KV 加载。

        Args:
            scheduler_output: 调度器输出
        """
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        self.kv_connector.handle_preemptions(kv_connector_metadata)

        # TODO: 整理 KV 连接器对 forward_context 的使用
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self,
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        clear_metadata: bool = True,
    ) -> KVConnectorOutput | None:
        """前向传播后的钩子，等待保存完成并获取状态。

        Args:
            scheduler_output: 调度器输出
            wait_for_save: 是否等待保存完成
            clear_metadata: 是否清除元数据

        Returns:
            KV 连接器输出
        """
        if self._disabled:
            return None

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        if clear_metadata:
            self.kv_connector.clear_connector_metadata()
        return output

    def clear_metadata(self) -> None:
        """清除连接器元数据。在 draft model 运行后调用。"""
        if not self._disabled:
            self.kv_connector.clear_connector_metadata()

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        """无前向传播时的钩子。

        Args:
            scheduler_output: 调度器输出

        Returns:
            模型运行器输出
        """
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        kv_connector_output = self.post_forward(scheduler_output, wait_for_save=False)
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def set_disabled(self, disabled: bool) -> None:
        """设置连接器是否被禁用。

        Args:
            disabled: 是否禁用
        """
        # 确保禁用时不会调用逐层连接器钩子
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()
"""无操作 KV 连接器单例。"""


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    """获取 KV 连接器。

    如果没有 KV 传输组，返回无操作连接器；
    否则返回活动 KV 连接器。

    Args:
        vllm_config: vLLM 配置
        kv_caches_dict: KV 缓存字典

    Returns:
        KV 连接器实例
    """
    if not has_kv_transfer_group():
        # 无操作连接器
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
