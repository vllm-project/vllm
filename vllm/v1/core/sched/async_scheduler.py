# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""异步调度器模块。

本模块实现了异步调度器，负责：
- 继承 Scheduler 基类
- 支持异步调度（async scheduling）
- 处理推测解码的占位符
- 管理结构化输出的输出占位符

主要类：
- AsyncScheduler: 异步调度器实现
"""

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    """异步调度器实现。

    继承自 Scheduler，增加对异步调度的支持：
    - 为推测解码维护可复用的占位符列表
    - 跟踪结构化输出请求的输出占位符数量
    - 在调度后更新请求的占位符状态

    异步调度允许模型在输出尚未完全生成时继续调度新的步骤，
    提高 GPU 利用率。
    """

    def __init__(self, *args, **kwargs) -> None:
        """初始化异步调度器。

        Args:
            *args: 传递给 Scheduler 的位置参数
            **kwargs: 传递给 Scheduler 的关键字参数
        """
        super().__init__(*args, **kwargs)
        # 用于推测解码的可复用只读占位符列表
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """在调度后更新内部状态。

        此方法在每次调度后调用，用于：
        - 更新结构化输出的 token 状态
        - 为新的 draft/spec token 添加占位符

        Args:
            scheduler_output: 调度器输出
        """
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            # 更新结构化输出 token 状态
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # 请求将在此调度步骤生成一个新 token 加上 num_spec_tokens
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # 为新的 draft/spec token 添加占位符
            # 实际的 spec token id 将在 worker 进程中更新
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """根据模型输出更新请求状态。

        此方法在模型输出返回后调用，用于：
        - 处理 force preempted 请求的 token 丢弃
        - 更新输出占位符数量
        - 缓存新 token 到 KV 缓存

        Args:
            request: 要更新的请求
            new_token_ids: 新生成的 token ID 列表

        Returns:
            (实际更新的 token ID 列表，是否停止) 元组
        """
        if request.discard_latest_async_tokens:
            # 如果请求在 reset_prefix_cache 中被 force preempted，
            # 应该丢弃最新的异步 token
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # 更新输出占位符数量
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # 缓存新 token。被抢占的请求应该跳过
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
