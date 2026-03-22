# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 连接器模型运行器混入模块。

本模块定义 KV 连接器功能混入类，负责：
- 管理 KV 连接器的生命周期
- 处理 KV 缓存的异步传输
- 支持 uniform KV cache 布局

主要类：
- KVConnectorModelRunnerMixin: KV 连接器功能混入类
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# 定义为模型运行器（GPU、TPU）的 KV 连接器功能混入类
class KVConnectorModelRunnerMixin:
    """KV 连接器功能混入类。

    提供 KV 连接器的标准接口，用于管理 KV 缓存的传输。
    """

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        """KV 连接器无 forward 时的处理。

        即使没有工作要做，也执行 KV 发送/接收。

        Args:
            scheduler_output: 调度器输出
            vllm_config: vLLM 配置

        Returns:
            模型运行器输出
        """
        # 即使没有工作要做，也执行 KV 发送/接收
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        """获取 KV 连接器输出上下文管理器（如果有 KV 传输）。

        Args:
            scheduler_output: 调度器输出
            defer_finalize: 是否延迟完成

        Returns:
            KV 连接器输出上下文管理器
        """
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        """完成 KV 连接器：等待保存完成并清除元数据。

        当 defer_finalize=True 时，在 draft model forward 后调用。
        """
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # 此上下文管理器必须在活动的前向上下文中使用
    # 它封装了 execute_model 内的整个 KV 连接器生命周期
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        """获取 KV 连接器输出的上下文管理器。

        管理 KV 连接器的完整生命周期：
        1. 绑定连接器元数据
        2. 启动异步 KV 缓存传输
        3. 获取已完成的传输和统计信息
        4. 清除连接器元数据

        Args:
            scheduler_output: 调度器输出
            wait_for_save: 是否等待保存完成
            defer_finalize: 是否延迟完成

        Yields:
            KV 连接器输出
        """
        output = KVConnectorOutput()

        # 使用 forward() 的 KVConnector 元数据更新 KVConnector
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # 后台 KV 缓存传输发生在这里
        # 这些传输设计为异步的，涉及的请求可能与运行中的请求不重叠
        # 在这里执行以节省 collective_rpc
        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if wait_for_save and not defer_finalize:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """确定是否应该使用统一的 KV 布局。

        统一布局意味着所有层的 KV 缓存将共享相同的底层张量，
        其中对于给定的块号，所有层的相应 KV 数据将是连续的。
        这将允许一次性高效地传输所有层的每块 KV 数据。

        注意：只有在满足以下 3 个条件时才会应用此布局：
        1. KV 缓存配置只包含一个组，其中所有层具有相同的页面大小
        2. 配置了 KV 连接器，并且 KV 连接器实例更喜欢使用此布局
           (prefer_cross_layer_blocks() 返回 True)
        3. flash attention 后端支持此布局
           (get_kv_cache_stride_order(True) 包含 num_layers 维度的放置)

        注意：num_layers 维度在统一层张量中的实际放置将由
        注意力后端决定。因此，如果注意力后端不支持，
        层的 KV 数据可能仍然不是每块连续的。

        Args:
            attn_groups: 此模型的注意力组列表
            cache_dtype: KV 缓存数据类型

        Returns:
            如果应该使用统一 KV 缓存布局则返回 True
        """

        if not has_kv_transfer_group():
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

        if len(attn_groups) != 1 or len(attn_groups[0]) != 1:
            return False

        attn_group = attn_groups[0][0]
        kv_cache_spec = attn_group.kv_cache_spec
        if not isinstance(kv_cache_spec, AttentionSpec):
            return False

        attn_backend = attn_group.backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            1234,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError):
            return False

        # 检查注意力后端是否包含层维度
        if len(kv_cache_stride_order) != len(kv_cache_shape) + 1:
            return False

        # stride_order[0] == 0 表示 num_layers 在物理布局中保持第一
        # （恒等排列），因此不支持跨层
        return kv_cache_stride_order[0] != 0

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        """为所有层具有相同布局的简单情况初始化和重塑 KV 缓存。

        此函数假设 use_uniform_kv_cache() 返回 True。

        Args:
            kv_cache_config: KV 缓存配置
            attn_groups: 此模型的注意力组列表
            cache_dtype: KV 缓存数据类型
            device: 要分配的 torch 设备
            kernel_block_sizes: 每个 KV 缓存组的内核块大小

        Returns:
            元组 (kv_caches, cross_layers_kv_cache, attn_backend)，其中：
                kv_caches 是从层名映射到其相应 KV 缓存内存缓冲区的字典
                cross_layers_kv_cache 是跨层 KV 缓存张量
                attn_backend 是匹配此张量的注意力后端
        """
        attn_group = attn_groups[0][0]
        kv_cache_spec = attn_group.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec)

        tensor_sizes = set(
            kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        )
        assert len(tensor_sizes) == 1
        tensor_size = tensor_sizes.pop()

        page_size = kv_cache_spec.page_size_bytes
        assert tensor_size % page_size == 0
        num_blocks = tensor_size // page_size
        num_layers = len(kv_cache_config.kv_cache_tensors)
        total_size = tensor_size * num_layers

        assert len(kernel_block_sizes) == 1
        kernel_block_size = kernel_block_sizes[0]
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        attn_backend = attn_group.backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # 在形状前添加 num_layers 维度
        kv_cache_shape = (num_layers,) + kv_cache_shape

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        logger.info("分配形状为 %s 的跨层 KV 缓存", kv_cache_shape)

        # 为所有层分配一个连续缓冲区
        cross_layers_kv_cache = (
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            .view(kv_cache_shape)
        )

        # 保持原始 KV 形状视图
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

        kv_caches = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            tensor = permuted_kv_cache[i]
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        return kv_caches, cross_layers_kv_cache, attn_backend
