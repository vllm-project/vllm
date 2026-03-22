# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载 CPU-GPU 处理程序模块。

本模块实现了 CPU 和 GPU 之间的 KV 数据传输处理程序，负责：
- 提供单向传输处理程序（CPU->GPU 或 GPU->CPU）
- 管理 CUDA 流和事件的异步传输
- 处理不同块大小的转换和映射
- 提供双向传输处理程序管理器

主要类：
- Transfer: 传输作业数据类
- SingleDirectionOffloadingHandler: 单向卸载处理程序
- CpuGpuOffloadingHandlers: CPU-GPU 双向处理程序管理器

主要函数：
- expand_block_ids: 将块 ID 扩展为子块 ID 列表
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


@dataclass
class Transfer:
    """传输作业数据类。

    记录异步传输作业的详细信息，用于跟踪完成状态和统计。

    Attributes:
        job_id: 传输作业的唯一 ID
        stream: 执行传输的 CUDA 流
        start_event: 记录传输开始的 CUDA 事件
        end_event: 记录传输结束的 CUDA 事件
        num_bytes: 传输的字节数
    """

    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """将块 ID 转换为匹配的子块 ID 列表。

    假设每个块由实际的 block_size_factor 个子块组成。
    输出到 output 张量。
    前 skip_count 个块将被跳过。
    注意 skip_count 必须小于 block_size_factor。

    例如，如果 block_ids = [0, 1, 3] 且 block_size_factor = 4，
    则输出 [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    因为 0 映射到 [0, 1, 2, 3]
    1 映射到 [4, 5, 6, 7]
    3 映射到 [12, 13, 14, 15]

    第一个块可以跳过前 skip_count 个子块，用于处理不对齐的情况。

    Args:
        block_ids: 块 ID 的一维 numpy 数组
        block_size_factor: 每个 KV 块包含的子块数量
        output: 输出数组，用于存储扩展后的子块 ID
        skip_count: 第一个块要跳过的子块数量，必须 < block_size_factor
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        # 第一个块使用 first_range（可能跳过前面的子块），后续块使用完整范围
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """单向卸载处理程序。

    处理单个方向的传输，要么 CPU->GPU，要么 GPU->CPU。
    传输保证按照提交的顺序执行。
    每个传输使用一个唯一的 CUDA 流，该流只有在前一个传输的流完成后才会开始执行。

    实现细节：
        - 使用 CUDA 流池和事件池来复用资源
        - 传输按顺序提交，通过 stream.wait_event() 保证依赖
        - GPU->CPU 传输会等待模型计算完成后再开始卸载

    Attributes:
        src_tensors: 源 KV 缓存张量列表
        dst_tensors: 目标 KV 缓存张量列表
        src_block_size_factor: 源张量中每个 KV 块的子块数量
        dst_block_size_factor: 目标张量中每个 KV 块的子块数量
        block_size_in_bytes: 每个块的字节数列表
        gpu_to_cpu: 是否为 GPU->CPU 传输
        _transfer_events: 作业 ID 到结束事件的映射
        _transfers: 传输队列
        _stream_pool: 可用的 CUDA 流池
        _event_pool: 可用的 CUDA 事件池
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        """初始化单向卸载处理程序。

        Args:
            src_tensors: 要复制的 KV 缓存张量列表
            dst_tensors: 要复制到的 KV 缓存张量列表，顺序应与 src_tensors 匹配
            src_block_size_factor: 源张量中每个 KV 块的子块数量
            dst_block_size_factor: 目标张量中每个 KV 块的子块数量
        """
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor: int = src_block_size_factor // min_block_size_factor
        self.dst_block_size_factor: int = dst_block_size_factor // min_block_size_factor

        self.block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0) * min_block_size_factor
            for tensor in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

        assert len(src_tensors) > 0
        self.gpu_to_cpu: bool = self.src_tensors[0].is_cuda
        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # 传输队列（job_id, stream, event）
        self._transfers: deque[Transfer] = deque()
        # 可复用的 CUDA 流列表
        self._stream_pool: list[torch.cuda.Stream] = []
        # 可复用的 CUDA 事件列表
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """发起异步传输。

        1. 解析源和目标规范，获取块 ID 列表
        2. 计算需要跳过的子块数量（处理不对齐）
        3. 构建源到目标的块 ID 映射
        4. 获取或创建 CUDA 流和事件
        5. 如果是 GPU->CPU，等待模型计算完成
        6. 如果有前一个传输，等待其完成
        7. 在 CUDA 流上执行 swap_blocks 操作
        8. 记录传输事件并加入队列

        Args:
            job_id: 唯一的作业 ID
            transfer_spec: （源规范，目标规范）元组

        Returns:
            True 表示传输提交成功

        Raises:
            AssertionError: 如果规范类型不匹配或块数量不匹配
        """
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        # 计算需要跳过的源子块数量，以确保源和目标数量匹配
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        # 构建源到目标的块 ID 映射
        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # 获取或创建 CUDA 流和事件
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            # GPU->CPU 传输：等待模型计算完成后再开始卸载
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # 保证作业在前一个作业完成后才开始
            stream.wait_event(last_event)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for src_tensor, dst_tensor, block_size_in_bytes in zip(
                self.src_tensors,
                self.dst_tensors,
                self.block_size_in_bytes,
            ):
                ops.swap_blocks(
                    src_tensor,
                    dst_tensor,
                    block_size_in_bytes,
                    src_to_dst_tensor,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=dst_sub_block_count * self.total_block_size_in_bytes,
            )
        )

        # 成功
        return True

    def get_finished(self) -> list[TransferResult]:
        """获取自上次调用以来完成的传输。

        查询队列头部的传输是否完成（end_event.query()），
        如果完成则创建 TransferResult 并回收资源（流和事件）。

        Returns:
            完成的传输结果列表
        """
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time 单位是毫秒

            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )

            results.append(result)
            # 回收 CUDA 资源
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        """等待指定的作业完成（阻塞）。

        Args:
            job_ids: 要等待的作业 ID 集合
        """
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()


class CpuGpuOffloadingHandlers:
    """CPU-GPU 双向卸载处理程序管理器。

    创建并管理两个方向的传输处理程序：
    - GPU->CPU：用于将 KV 块从 GPU 卸载到 CPU
    - CPU->GPU：用于将 KV 块从 CPU 加载回 GPU

    自动检测 KV 缓存张量的布局（是否有 layers 维、是否 K/V 分离），
    并据此分配匹配的 CPU 张量。

    Attributes:
        gpu_to_cpu_handler: GPU->CPU 传输处理程序
        cpu_to_gpu_handler: CPU->GPU 传输处理程序
    """

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        """初始化 CPU-GPU 卸载处理程序。

        Args:
            gpu_block_size: GPU 块大小
            cpu_block_size: CPU 块大小，必须是 gpu_block_size 的倍数
            num_cpu_blocks: CPU 块数量
            gpu_caches: layer_name -> gpu_kv_cache 张量的字典
            attn_backends: layer_name -> AttentionBackend 类型的字典
        """
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        # 查找 kernel 块大小并确定每个 gpu 张量的布局
        kernel_block_size: int | None = None
        # (gpu_tensor, split_k_and_v) 列表
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers 张量
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # 在 test_shape 前添加 dummy num_layers=80
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # shape 应该是 (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            if has_layers_dim:
                # 在 cross-layers 情况下，注册的 kv cache 张量
                # shape 匹配物理布局，而 test_shape 是逻辑布局
                # 为了匹配它们，需要置换 test_shape
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=has_layers_dim
                    )
                    assert len(kv_cache_stride_order) == len(gpu_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(gpu_shape)))

                test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # 查找 block_size (16) 维度索引
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        # 分配 CPU 张量
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            cpu_tensor = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

            gpu_tensors.extend(gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])
            cpu_tensors.extend(cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor])

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )
