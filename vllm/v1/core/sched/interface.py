# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度器接口模块。

本模块定义了调度器的抽象接口和暂停状态枚举，负责：
- 定义调度器必须实现的抽象方法
- 提供暂停状态枚举（UNPAUSED、PAUSED_NEW、PAUSED_ALL）

主要类：
- PauseState: 调度器暂停状态枚举
- SchedulerInterface: 调度器抽象基类

使用说明：
    SchedulerInterface 定义了调度器的完整接口，包括：
    - schedule(): 执行调度决策
    - get_grammar_bitmask(): 获取文法位掩码（用于结构化输出）
    - update_from_output(): 根据模型输出更新状态
    - add_request(): 添加新请求
    - finish_requests(): 完成请求
    - reset_prefix_cache(): 重置前缀缓存
    等等。
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class PauseState(enum.IntEnum):
    """调度器暂停状态枚举。

    用于控制调度器的行为：
    - UNPAUSED: 正常运行状态
    - PAUSED_NEW: 不调度新请求，但继续处理已运行的请求
    - PAUSED_ALL: 完全暂停，不调度任何请求
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulerInterface(ABC):
    """调度器抽象基类。

    定义了 vLLM V1 调度器必须实现的完整接口。
    调度器负责管理请求的生命周期、分配 KV 缓存资源、
    处理推测解码和前缀缓存等功能。
    """

    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """初始化调度器。

        Args:
            vllm_config: vLLM 配置
            kv_cache_config: KV 缓存配置
            structured_output_manager: 结构化输出管理器
            block_size: KV 缓存块大小
            mm_registry: 多模态注册表
            include_finished_set: 是否包含已完成的请求集合
            log_stats: 是否记录统计信息
        """
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """执行请求调度，为每个步骤分配合适的 token 和 KV 缓存块。

        调度决策在迭代级别进行。每个调度步骤对应模型的一次前向传播。
        因此，该方法由引擎中的忙循环重复调用。

        本质上，调度器产生一个 {req_id: num_tokens} 的字典，
        指定每个请求在此调度步骤中处理多少个 token。例如：
        - num_tokens 可以大到新请求的 prompt token 数量
        - 或者为 1（自回归逐个生成新 token 的请求）
        - 或者介于两者之间（chunked prefills、前缀缓存、推测解码等情况）

        此外，调度器还返回关于每个请求或整个 batch 的有用信息。
        模型 runner 将使用这些信息准备模型输入。

        Returns:
            SchedulerOutput 对象，包含有关调度请求的信息
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        """获取结构化输出的文法位掩码。

        Args:
            scheduler_output: 调度器输出

        Returns:
            GrammarOutput 对象或 None（如果没有结构化输出请求）
        """
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """根据模型 runner 输出更新调度器状态。

        该方法在模型 runner 处理完调度的请求后调用。
        模型 runner 输出包括生成的 token ID、下一步的 draft token ID 等。
        调度器使用这些信息更新其状态、检查已完成的请求，
        并返回每个请求的输出。

        Returns:
            从 client 索引到 EngineCoreOutputs 对象的字典，
            包含来自该客户端的每个请求的输出
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """用新生成的 draft token ID 更新请求，必要时应用结构化输出文法验证。

        Args:
            draft_token_ids: 每个请求的 draft token ID
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """用新生成的 draft token ID 更新调度器输出，必要时应用结构化输出文法验证。

        Args:
            draft_token_ids: 每个请求的 draft token ID
            scheduler_output: 用相应的 draft token ID 更新给定的 scheduler_output
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """将新请求添加到调度器的内部队列。

        Args:
            request: 新请求
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> list[tuple[str, int]]:
        """完成调度器内部队列中的请求。如果请求不在队列中，则不执行任何操作。

        该方法在两种情况下调用：
        1. 当请求被客户端中止时
        2. 当 frontend 进程在对请求生成的 token 进行去 tokenize 后检测到停止字符串时

        Args:
            request_ids: 单个请求 ID、请求 ID 列表，或 None（完成所有请求）
            finished_status: 给定请求的完成状态

        Returns:
            (req_id, client_index) 元组列表，包含被中止的请求
            不包括已完成的请求
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """返回调度器内部队列中未完成请求的数量。

        Returns:
            未完成请求的数量
        """
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """返回调度器内部队列中是否有未完成请求。

        Returns:
            如果有未完成请求则返回 True
        """
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """返回是否有已完成的请求需要清除。

        注意：这与 `not self.has_unfinished_requests()` 不同。

        调度器维护一个内部列表，包含上一步完成的请求。
        该列表在下次调用 schedule() 时返回，发送到模型 runner 以
        清除这些已完成请求的缓存状态。

        该方法检查此内部已完成请求列表是否非空。
        此信息对 DP attention 很有用。

        Returns:
            如果有已完成请求则返回 True
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """返回是否有未完成请求，或是否有尚未在 SchedulerOutputs 中返回的已完成请求。

        Returns:
            如果有请求需要处理则返回 True
        """
        return self.has_unfinished_requests() or self.has_finished_requests()

    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """调度器当前的暂停状态。

        Returns:
            PauseState 枚举值
        """
        raise NotImplementedError

    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        """设置调度器的暂停状态。

        Args:
            pause_state: 新的暂停状态
        """
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置 KV 缓存的前缀缓存。

        当模型权重实时更新时尤其需要此方法。

        Args:
            reset_running_requests: 如果为 True，所有运行中的请求将被抢占并
                移回等待队列。否则，只有当没有运行中的请求占用 KV 缓存时，
                才会重置 KV 前缀缓存。
            reset_connector: 是否重置连接器状态

        Returns:
            是否成功重置
        """
        raise NotImplementedError

    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """重置编码器缓存以使所有缓存的编码器输出失效。

        当模型权重更新时应调用此方法，确保过时的视觉嵌入不会被复用。
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """返回 (num_running_reqs, num_waiting_reqs)。

        Returns:
            (运行中请求数，等待中请求数) 元组
        """
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """创建用于日志的 SchedulerStats 对象。

        SchedulerStats 对象在每个调度步骤创建。

        Returns:
            SchedulerStats 对象或 None
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """关闭调度器。"""
        raise NotImplementedError

    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        """获取 KV 连接器。

        Returns:
            KV 连接器或 None
        """
        return None
