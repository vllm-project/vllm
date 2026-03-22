# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""结构化输出工人模块。

本模块提供结构化输出的位掩码应用功能，负责：
- 管理语法位掩码缓冲区
- 异步复制位掩码到 GPU
- 使用 Triton 内核应用语法位掩码到 logits

主要类：
- StructuredOutputsWorker: 结构化输出工人类
"""
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    """结构化输出工人类。

    负责将语法位掩码应用到 logits 上，用于结构化输出。
    使用异步 GPU 复制和 Triton 内核进行高效处理。

    Attributes:
        logits_indices: logits 索引张量
        grammar_bitmask: 语法位掩码张量
        device: 设备类型
        copy_stream: CUDA 复制流
    """

    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        """初始化结构化输出工人。

        Args:
            max_num_logits: 最大 logits 数量
            vocab_size: 词表大小
            device: 设备类型
        """
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        """应用语法位掩码到 logits。

        Args:
            logits: logits 张量
            input_batch: 输入批次
            grammar_req_ids: 语法请求 ID 列表
            grammar_bitmask: 语法位掩码 numpy 数组
        """
        if not grammar_req_ids:
            return

        # 异步复制位掩码到 GPU
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # 构建 bitmask -> logits 映射
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # 异步复制映射到 GPU
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # 确保所有异步复制完成后再启动内核
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # 确保复制流等待设备张量使用完毕后再重用或释放它们
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton 内核：应用语法位掩码到 logits。

    将不符合语法的 token 的 logits 设置为 -inf，
    从而在采样时排除这些 token。

    Args:
        logits_ptr: logits 张量指针
        logits_stride: logits 步幅
        logits_indices_ptr: logits 索引指针
        bitmask_ptr: 位掩码指针
        bitmask_stride: 位掩码步幅
        vocab_size: 词表大小
        BLOCK_SIZE: 块大小（编译时常量）
    """
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # 加载位掩码
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # 解包位掩码
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # 应用位掩码到 logits
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
