# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA 模型运行器混入模块。

本模块定义 LoRA 功能混入类，负责：
- 加载 LoRA 模型
- 设置活动的 LoRA 适配器
- 管理 LoRA 缓存

主要类：
- LoRAModelRunnerMixin: LoRA 功能混入类
"""

from contextlib import contextmanager
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch

logger = init_logger(__name__)


# 定义为 GPUModelRunner 的混入类
class LoRAModelRunnerMixin:
    """LoRA 功能混入类。

    提供 LoRA 适配器的管理和执行功能。
    """

    def load_lora_model(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> nn.Module:
        """加载支持 LoRA 的模型。

        Args:
            model: 要加载的模型
            vllm_config: vLLM 配置
            device: 设备类型

        Returns:
            加载了 LoRA 管理器的模型

        Raises:
            ValueError: 如果模型不支持 LoRA
        """
        if not supports_lora(model):
            raise ValueError(f"{model.__class__.__name__} 尚不支持 LoRA。")

        # 添加 LoRA 管理器到模型运行器
        self.lora_manager = LRUCacheWorkerLoRAManager(
            vllm_config,
            device,
            model.embedding_modules,
        )
        return self.lora_manager.create_lora_manager(model, vllm_config)

    def _set_active_loras(
        self,
        prompt_lora_mapping: tuple[int, ...],
        token_lora_mapping: tuple[int, ...],
        lora_requests: set[LoRARequest],
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        """设置活动的 LoRA 适配器。

        Args:
            prompt_lora_mapping: prompt LoRA 映射
            token_lora_mapping: token LoRA 映射
            lora_requests: LoRA 请求集合
            mapping_type: LoRA 映射类型
        """
        self._ensure_lora_enabled()

        # 设置 is_prefill 为 True，因此我们总是在非 cuda 平台上使用 SGMV 内核
        # 在 cuda 平台上，我们对 prefill 和 decode 使用相同的内核
        # 这个标志通常被忽略
        lora_mapping = LoRAMapping(
            token_lora_mapping,
            prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
        )
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def _ensure_lora_enabled(self) -> None:
        """确保 LoRA 已启用。

        Raises:
            RuntimeError: 如果 LoRA 未启用
        """
        if not hasattr(self, "lora_manager"):
            raise RuntimeError("LoRA 未启用。请使用 --enable-lora 启用 LoRA。")

    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        """设置活动的 LoRA 适配器。

        Args:
            input_batch: 输入批次
            num_scheduled_tokens: 每个请求调度的 token 数量
            num_sampled_tokens: 每个请求采样的 token 数量（可选）
            mapping_type: LoRA 映射类型
        """
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        prompt_lora_mapping: tuple[int, ...]  # 大小为 np.sum(num_sampled_tokens)
        token_lora_mapping: tuple[int, ...]  # 大小为 np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = (
            input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        return self._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
        )

    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ):
        """设置虚拟 LoRA 用于预热（如果有 LoRA 配置）。

        Args:
            lora_config: LoRA 配置
            remove_lora: 是否在退出后移除 LoRA
        """
        if lora_config is None:
            yield
        else:
            # __enter__ 代码
            assert self.lora_manager is not None, "LoRA 未启用"

            num_loras = lora_config.max_loras
            lora_warmup_rank = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            # 创建虚拟 lora 请求
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # 在这里添加虚拟 LoRA，这样_set_active_loras 就不会尝试从
                # 磁盘加载
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)

                yield

            # __exit__ 代码
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
        num_sampled_tokens: np.ndarray | None = None,
        num_active_loras: int = 0,
    ):
        """选择虚拟 LoRA 用于捕获/预热的上下文管理器。

        Args:
            lora_config: LoRA 配置，或者 None 如果 LoRA 已禁用
            num_scheduled_tokens: 每个请求的调度 token 数量数组
            num_sampled_tokens: 每个请求的采样 token 数量数组
            num_active_loras: 要使用的不同活动 LoRA 数量
                - 0：没有 LoRA 活动（设置零映射）
                - >0：使用正好这个数量的不同 LoRA
        """
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        # 仅在没有任何 LoRA 配置时跳过 LoRA 设置
        if lora_config is None:
            yield
        else:
            # __enter__ 代码
            assert self.lora_manager is not None, "LoRA 未启用"

            num_reqs = len(num_scheduled_tokens)
            max_loras = lora_config.max_loras

            # 确定使用多少个不同的 LoRA 以及是否包含
            # 无 LoRA token（-1 条目）
            # 当 num_active_loras > max_loras 时（例如 max_loras + 1），我们需要
            # 包含 -1 条目来模拟同时包含 LoRA 和无 LoRA token 的批次
            # 这确保 prepare_tensors 计算正确的 num_active_loras
            # 与 cudagraph 捕获键匹配
            if num_active_loras == 0:
                # 没有 LoRA 活动 - 使用 0 映射，如原始代码
                effective_num_loras = 0
                include_no_lora = False
            elif num_active_loras > max_loras:
                # num_active_loras > max_loras 表示我们想要 max_loras 个适配器
                # 加上无 LoRA token（-1）。这是 max_loras + 1 的情况
                effective_num_loras = max_loras
                include_no_lora = True
            else:
                # 请求特定数量的活动 LoRA
                effective_num_loras = min(num_active_loras, max_loras)
                include_no_lora = False

            # 创建 prompt lora 映射
            # 循环分配 LoRA ID 以模拟最坏情况
            # LoRA ID 是 1 索引的（1 到 max_loras），如 LoRARequest 所要求
            # convert_mapping() 将这些转换为 0 索引的槽索引
            if effective_num_loras > 0:
                if include_no_lora:
                    # 包含 -1（无 LoRA）条目，循环通过
                    # -1, 1, 2, ..., effective_num_loras
                    # 这确保 prepare_tensors 同时看到 LoRA 和无 LoRA token
                    # 计算 num_active_loras = effective_num_loras+1
                    cycle_values = np.array(
                        list(range(1, effective_num_loras + 1)),
                        dtype=np.int32,
                    )
                    prompt_lora_mapping = cycle_values[
                        np.arange(num_reqs, dtype=np.int32) % len(cycle_values)
                    ]
                else:
                    # 使用 1 到 effective_num_loras（1 索引的 lora ID）
                    prompt_lora_mapping = (
                        np.arange(num_reqs, dtype=np.int32) % effective_num_loras
                    ) + 1
            else:
                # 没有 LoRA 活动 - 对所有 token 使用 0（原始行为）
                prompt_lora_mapping = np.zeros(num_reqs, dtype=np.int32)

            # 创建 sample lora 映射
            sample_lora_mapping = np.repeat(prompt_lora_mapping, num_sampled_tokens)

            # 创建 token lora 映射
            token_lora_mapping = np.repeat(prompt_lora_mapping, num_scheduled_tokens)

            # 创建虚拟 lora 请求（仅针对活动的 LoRA）
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, effective_num_loras + 1)
            }

            self._set_active_loras(
                tuple(sample_lora_mapping),
                tuple(token_lora_mapping),
                lora_requests,
                mapping_type,
            )

            yield

    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        remove_lora: bool = True,
        num_active_loras: int = 0,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ):
        """使用 LoRA 进行虚拟运行的上下文管理器。

        Args:
            lora_config: LoRA 配置
            num_scheduled_tokens: 每个请求的调度 token 数量数组
            num_sampled_tokens: 每个请求的采样 token 数量数组
            remove_lora: 是否在上下文退出后移除 LoRA
            num_active_loras: 要使用的不同活动 LoRA 数量
                当 num_active_loras > 0 时激活 LoRA
        """
        with (
            self.maybe_setup_dummy_loras(lora_config, remove_lora),
            self.maybe_select_dummy_loras(
                lora_config,
                num_scheduled_tokens,
                mapping_type,
                num_sampled_tokens,
                num_active_loras,
            ),
        ):
            yield

    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
        """移除所有 LoRA（如果有 LoRA 配置）。

        Args:
            lora_config: LoRA 配置
        """
        if lora_config is None:
            return
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """添加 LoRA 适配器。

        Args:
            lora_request: LoRA 请求

        Returns:
            如果成功添加则返回 True

        Raises:
            RuntimeError: 如果 LoRA 未启用
        """
        self._ensure_lora_enabled()
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """移除 LoRA 适配器。

        Args:
            lora_id: LoRA ID

        Returns:
            如果成功移除则返回 True

        Raises:
            RuntimeError: 如果 LoRA 未启用
        """
        self._ensure_lora_enabled()
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        """固定 LoRA 适配器（防止被驱逐）。

        Args:
            lora_id: LoRA ID

        Returns:
            如果成功固定则返回 True

        Raises:
            RuntimeError: 如果 LoRA 未启用
        """
        self._ensure_lora_enabled()
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        """列出所有 LoRA ID。

        Returns:
            LoRA ID 集合

        Raises:
            RuntimeError: 如果 LoRA 未启用
        """
        self._ensure_lora_enabled()
        return self.lora_manager.list_adapters()
