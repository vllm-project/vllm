# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 注意力后端模块。

本模块实现了基于 Mamba2 的注意力后端，负责：
- 实现 Mamba2 注意力后端类
- 构建 Mamba2 专用的元数据
- 支持分块对齐的变长元数据

主要类：
- Mamba2AttentionBackend: Mamba2 注意力后端类
- Mamba2AttentionMetadata: Mamba2 元数据类
- Mamba2AttentionMetadataBuilder: 元数据构建器

辅助函数：
- compute_varlen_chunk_metadata: 计算变长分块元数据
"""
import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建用于 Mamba2 SSD kernel 的分块对齐变长元数据。

    给定形状的序列累积 token 起始位置 `query_start_loc` [B+1] 和
    物理 `chunk_size`，返回三个相同设备上的张量：
      - cu_chunk_seqlens: (nchunks+1,) int32
        逻辑块长度的独占前缀和（每个逻辑块不会跨越序列或物理块边界）
      - last_chunk_indices: (B,) int32
        每个序列的最后一个逻辑块的索引（空序列为 -1）
      - seq_idx_chunks: (nchunks,) int32
        按顺序排列的每个逻辑块的序列索引

    此函数设计为轻量级且在 CPU 上运行；它镜像了 V1 Mamba2 元数据构建器
    生成的元数据并被导出，因此测试和其他调用者可以避免重复此逻辑。

    Args:
        query_start_loc: 累积 token 起始位置，形状 [B+1]
        chunk_size: 块大小

    Returns:
        (cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks) 元组
    """
    assert query_start_loc.ndim == 1, "query_start_loc 必须是 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] 必须是 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # 空序列
            continue
        pos = s
        while pos < e:
            # 在序列边界和物理块边界处分割
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # 对逻辑块长度进行独占前缀和
    if chunk_lens:
        cu_chunk_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(chunk_lens)),
            device=device,
            dtype=torch.int32,
        )
        # 最后一个边界必须等于总 token 数
        assert int(cu_chunk_seqlens[-1].item()) == total
    else:
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)

    last_chunk_indices_t = (
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
        if len(starts) > 0
        else torch.empty((0,), device=device, dtype=torch.int32)
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    """Mamba2 注意力后端类。

    基于 Mamba2 实现的注意力后端。
    """
    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "MAMBA2_ATTN"
        """
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            Mamba2AttentionMetadataBuilder 类
        """
        return Mamba2AttentionMetadataBuilder


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    """Mamba2 注意力元数据类。

    继承自 BaseMambaAttentionMetadata，添加了 Mamba2 专用的元数据字段。

    Attributes:
        prep_initial_states: 是否准备初始状态
        chunk_size: 块大小
        seq_idx_p: 预填充的序列索引
    """
    prep_initial_states: bool = False
    """是否准备初始状态。"""

    chunk_size: int = 0
    """块大小。"""

    # 分块相关的元数据（仅用于预填充）
    seq_idx_p: torch.Tensor | None = None
    """预填充的序列索引张量。"""


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    """Mamba2 元数据构建器类。

    负责构建 Mamba2 注意力运行所需的元数据对象。
    继承自 BaseMambaAttentionMetadataBuilder。

    Class Attributes:
        metadata_cls: 元数据类
    """
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Mamba2 元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # 获取 Mamba2 的块大小配置
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "Mamba2 模型需要在模型配置中设置 chunk_size"
        )
        self.chunk_size: int = chunk_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        """构建 Mamba2 注意力元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建
            **kwargs: 其他参数（如 num_accepted_tokens）

        Returns:
            构建的 Mamba2AttentionMetadata 对象
        """
        # 计算通用元数据
        common = self._compute_common_metadata(
            common_attn_metadata, num_accepted_tokens=kwargs.get("num_accepted_tokens")
        )

        # 初始化 Mamba2 专用字段
        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # 仅为预填充计算 seq_idx
        if common.num_prefills > 0:
            # 检查是否有预填充请求需要初始状态
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            # 构建分块元数据张量
            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

        # 返回添加了 Mamba2 专用字段的元数据
        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )
