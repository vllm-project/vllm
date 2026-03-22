# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""块表模块。

本模块提供块表和 slot mapping 管理功能，负责：
- 管理 KV 缓存块的分配和映射
- 计算 slot mapping（token 到 KV 缓存位置的映射）
- 支持混合块大小（hybrid block sizes）
- 支持上下文并行（CP）的交错存储

主要类：
- BlockTable: 单个 KV 缓存组的块表
- MultiGroupBlockTable: 多个 KV 缓存组的块表
"""

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


class BlockTable:
    """块表类。

    管理 KV 缓存块的分配、映射和 slot mapping 计算。
    支持标准块大小和混合块大小（内存分配块大小与内核计算块大小不同）
    两种模式，以及上下文并行的交错存储。

    Attributes:
        max_num_reqs: 最大请求数量
        max_num_batched_tokens: 最大批次 token 数
        pin_memory: 是否锁定内存以加速 GPU 传输
        device: 目标设备
        block_size: 块大小（用于内核计算）
        blocks_per_kv_block: 每个 KV 缓存块对应的内核块数量
        use_hybrid_blocks: 是否使用混合块
        max_num_blocks_per_req: 每个请求的最大块数
        block_table: 块表缓冲区
        num_blocks_per_row: 每行的块数
        slot_mapping: slot mapping 缓冲区
        pcp_world_size: PCP 世界大小
        pcp_rank: PCP rank
        dcp_world_size: DCP 世界大小
        dcp_rank: DCP rank
        cp_kv_cache_interleave_size: CP KV 缓存交错大小
    """

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ):
        """初始化块表。

        Args:
            block_size: KV 缓存内存分配的块大小
            max_num_reqs: 支持的最大并发请求数
            max_num_blocks_per_req: 每个请求的最大块数
            max_num_batched_tokens: 批次中最大 token 数
            pin_memory: 是否锁定内存以加速 GPU 传输
            device: 目标设备
            kernel_block_size: 底层注意力内核的块大小
            cp_kv_cache_interleave_size: CP KV 缓存交错大小
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # 标准情况：分配和计算使用相同的块大小
            # 不需要块拆分，直接映射
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # 混合情况：分配块大小与内核块大小不同
            # 内存块被细分以匹配内核要求
            # 例如：32-token 内存块与 16-token 内核块
            # → 每个内存块对应 2 个内核块
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP 可能在测试中未初始化
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP 可能在测试中未初始化
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        """追加块 ID 到行。

        Args:
            block_ids: 块 ID 列表
            row_idx: 行索引
        """
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """添加行（重置后追加）。

        Args:
            block_ids: 块 ID 列表
            row_idx: 行索引
        """
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """移动行。

        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        """交换行。

        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        """计算 slot mapping。

        将 token 位置映射到 KV 缓存 slot。
        例如：[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        其中 K 是 max_num_blocks_per_req，块大小为 2。

        Args:
            req_indices: 请求索引数组
            positions: token 位置数组
        """
        # NOTE(woosuk): 我们不能简单地使用 `token_indices // block_size`
        # 因为 M (max_model_len) 不一定能被 block_size 整除。
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): DCP 实现以交错方式存储 kvcache，
            # token_idx 为 i 的 token 的 kvcache 始终存储在
            # dcp_rank 等于 i % cp_world_size 的 GPU 上：

            # 使用"虚拟块"（等于 world_size * block_size）
            # 计算 block_table_indices
            virtual_block_size = self.block_size * total_cp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # 使用 virtual_block_size 进行 mask 计算，标记本地 token
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # 计算本地 block_offsets
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # 计算 slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # 写入最终 slot，非本地使用 -1
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )

    def commit_block_table(self, num_reqs: int) -> None:
        """提交块表到 GPU。

        Args:
            num_reqs: 请求数量
        """
        self.block_table.copy_to_gpu(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        """提交 slot mapping 到 GPU。

        Args:
            num_tokens: token 数量
        """
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        """清除块表和 slot mapping。"""
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """将 kv_manager_block_ids 转换为内核块 ID。

        示例：
            # kv_manager_block_ids: 32 tokens
            # 内核块大小：16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # 每个 kv_manager_block_id 映射到 2 个内核块 ID：
            # kv_manager_block_id 0 → 内核块 ID [0, 1]
            # kv_manager_block_id 1 → 内核块 ID [2, 3]
            # kv_manager_block_id 2 → 内核块 ID [4, 5]

        Args:
            kv_manager_block_ids: KV 管理器块 ID 数组
            blocks_per_kv_block: 每个 KV 缓存块的内核块数量
            kernel_block_arange: 内核块范围数组

        Returns:
            内核块 ID 数组
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """获取块表的设备张量。

        Args:
            num_reqs: 请求数量

        Returns:
            设备上的块表张量
        """
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """获取块表的 CPU 张量。

        Returns:
            CPU 上的块表张量
        """
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """获取块表的 numpy 数组。

        Returns:
            块表 numpy 数组
        """
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        """创建 CPU/GPU 缓冲区。

        Args:
            *size: 缓冲区大小
            dtype: 数据类型

        Returns:
            CpuGpuBuffer 实例
        """
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """多 KV 缓存组的块表类。

    为每个 KV 缓存组管理独立的块表，支持不同的块大小配置。

    Attributes:
        block_tables: 每个 KV 缓存组的块表列表
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        """初始化多组块表。

        Args:
            max_num_reqs: 最大请求数量
            max_model_len: 最大模型长度
            max_num_batched_tokens: 最大批次 token 数
            pin_memory: 是否锁定内存
            device: 目标设备
            block_sizes: 每个组的块大小列表
            kernel_block_sizes: 每个组的内核块大小列表
            max_num_blocks: 每个组的最大块数列表（可选）
            cp_kv_cache_interleave_size: CP KV 缓存交错大小
        """
        """
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): 每个 dcp rank 只存储
            # (max_model_len//dcp_world_size) 个 token 在 kvcache 中，
            # 所以用于计算 max_num_blocks_per_req 的 block_size
            # 必须乘以 dcp_world_size。
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        """追加块 ID 到行。

        Args:
            block_ids: 每个组的块 ID 元组
            row_idx: 行索引
        """
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        """添加行（重置后追加）。

        Args:
            block_ids: 每个组的块 ID 元组
            row_idx: 行索引
        """
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """移动行。

        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        """交换行。

        Args:
            src: 源行索引
            tgt: 目标行索引
        """
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        """计算 slot mapping。

        Args:
            req_indices: 请求索引数组
            positions: token 位置数组
        """
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        """提交块表到 GPU。

        Args:
            num_reqs: 请求数量
        """
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        """提交 slot mapping 到 GPU。

        Args:
            num_tokens: token 数量
        """
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        """清除所有块表。"""
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """获取第 idx 个 KV 缓存组的块表。

        Args:
            idx: KV 缓存组索引

        Returns:
            第 idx 个块表
        """
        return self.block_tables[idx]
