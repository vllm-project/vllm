# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""延迟交互（Late Interaction）池化模块。

本模块实现了延迟交互（Late Interaction）池化方法，主要用于：
- ColBERT 风格的延迟交互池化
- 查询/文档嵌入的缓存和评分
- MaxSim 相似度计算

延迟交互是一种高效的检索方法，它分别编码查询和文档，
然后在最后阶段进行细粒度的交互计算。

主要函数：
- get_late_interaction_engine_index: 获取延迟交互引擎索引
- build_late_interaction_query_params: 构建查询参数
- build_late_interaction_doc_params: 构建文档参数
- compute_maxsim_score: 计算单个 MaxSim 分数
- compute_maxsim_scores: 批量计算 MaxSim 分数

ColBERT MaxSim 说明：
MaxSim 是 ColBERT 使用的相似度计算方法，对于查询 token 的每个嵌入，
找到与之最相似的文档 token 嵌入，然后求和得到最终分数。
"""

import zlib
from collections.abc import Sequence

import torch

from vllm.pooling_params import LateInteractionParams, PoolingParams

# 延迟交互模式：缓存查询
LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"
# 延迟交互模式：评分文档
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"


def get_late_interaction_engine_index(
    pooling_params: PoolingParams | None,
    num_engines: int,
) -> int | None:
    """获取延迟交互引擎索引。

    根据查询键（query_key）的 CRC32 哈希值，将请求分配到固定的引擎。
    这样可以确保相同查询键的请求总是被路由到同一个引擎，
    从而实现查询嵌入的本地缓存。

    Args:
        pooling_params: 池化参数
        num_engines: 引擎数量

    Returns:
        引擎索引（0 到 num_engines-1），如果不符合条件则返回 None
    """
    if pooling_params is None or pooling_params.late_interaction_params is None:
        return None

    late_interaction_params = pooling_params.late_interaction_params
    mode = late_interaction_params.mode
    if mode not in (
        LATE_INTERACTION_MODE_CACHE_QUERY,
        LATE_INTERACTION_MODE_SCORE_DOC,
    ):
        return None

    query_key = late_interaction_params.query_key
    if not isinstance(query_key, str) or not query_key:
        return None

    # 查询嵌入缓存在进程本地 worker 内存中，
    # 将共享相同查询键的请求固定到同一个引擎
    return zlib.crc32(query_key.encode("utf-8")) % num_engines


def build_late_interaction_query_params(
    query_key: str,
    query_uses: int,
) -> LateInteractionParams:
    """构建延迟交互查询参数。

    用于缓存查询嵌入，以便后续文档评分时复用。

    Args:
        query_key: 查询键，用于标识和缓存查询嵌入
        query_uses: 查询使用次数，至少为 1

    Returns:
        延迟交互参数，模式为 CACHE_QUERY
    """
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,
        query_key=query_key,
        query_uses=max(1, int(query_uses)),
    )


def build_late_interaction_doc_params(
    query_key: str,
) -> LateInteractionParams:
    """构建延迟交互文档参数。

    用于对文档进行评分，使用已缓存的查询嵌入。

    Args:
        query_key: 查询键，用于查找已缓存的查询嵌入

    Returns:
        延迟交互参数，模式为 SCORE_DOC
    """
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_SCORE_DOC,
        query_key=query_key,
    )


def compute_maxsim_score(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
) -> torch.Tensor:
    """计算单个查询 - 文档对的 MaxSim 分数。

    MaxSim 是 ColBERT 使用的相似度计算方法：
    1. 计算查询 token 和文档 token 之间的相似度矩阵
    2. 对每个查询 token，找到最相似的文档 token（max）
    3. 将所有查询 token 的最大相似度求和

    Args:
        q_emb: 查询嵌入 [num_query_tokens, dim]
        d_emb: 文档嵌入 [num_doc_tokens, dim]

    Returns:
        MaxSim 分数（标量张量）
    """
    # 使用 float32 计算以保证数值稳定性
    token_scores = torch.matmul(q_emb.float(), d_emb.float().T)
    # 对每个查询 token 取最大相似度，然后求和
    return token_scores.amax(dim=-1).sum()


def compute_maxsim_scores(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 64,
    max_score_matrix_elements: int = 64_000_000,
) -> list[torch.Tensor]:
    """批量计算多个查询 - 文档对的 MaxSim 分数。

    使用 mini-batch 方式处理多对查询/文档嵌入，以避免内存溢出。
    每批会自动调整大小以确保分数矩阵不超过最大元素数量限制。

    Args:
        q_embs: 查询嵌入序列，每个元素形状为 [num_query_tokens, dim]
        d_embs: 文档嵌入序列，每个元素形状为 [num_doc_tokens, dim]
        max_batch_size: 最大批次大小
        max_score_matrix_elements: 分数矩阵最大元素数量（用于内存控制）

    Returns:
        MaxSim 分数列表，每个元素是一个标量张量

    Raises:
        ValueError: 如果输入不合法（长度不匹配、维度不匹配、设备不同等）
    """
    if len(q_embs) != len(d_embs):
        raise ValueError("q_embs and d_embs must have the same length")

    num_pairs = len(q_embs)
    if num_pairs == 0:
        return []

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")
    if max_score_matrix_elements <= 0:
        raise ValueError("max_score_matrix_elements must be greater than 0")

    # 验证所有输入的有效性
    for q_emb, d_emb in zip(q_embs, d_embs):
        if q_emb.ndim != 2 or d_emb.ndim != 2:
            raise ValueError("Each embedding tensor must be 2-D")
        if q_emb.shape[1] != d_emb.shape[1]:
            raise ValueError("Query and document embeddings must have same dim")
        if q_emb.device != d_emb.device:
            raise ValueError("Query and document embeddings must be on same device")

    scores: list[torch.Tensor] = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(int(x.shape[0]) for x in q_embs[start:end])
        max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        # 保持分数矩阵有界，避免过大分配
        while (
            end - start > 1
            and (end - start) * max_q * max_d > max_score_matrix_elements
        ):
            end -= 1
            max_q = max(int(x.shape[0]) for x in q_embs[start:end])
            max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        batch_q = q_embs[start:end]
        batch_d = d_embs[start:end]
        batch_size = end - start
        device = batch_q[0].device
        dim = int(batch_q[0].shape[1])

        # 创建填充批次张量
        q_batch = torch.zeros(
            (batch_size, max_q, dim), dtype=torch.float32, device=device
        )
        d_batch = torch.zeros(
            (batch_size, max_d, dim), dtype=torch.float32, device=device
        )
        q_mask = torch.zeros((batch_size, max_q), dtype=torch.bool, device=device)
        d_mask = torch.zeros((batch_size, max_d), dtype=torch.bool, device=device)

        # 复制到填充张量
        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):
            q_len = int(q_emb.shape[0])
            d_len = int(d_emb.shape[0])
            q_batch[i, :q_len] = q_emb.to(device=device, dtype=torch.float32)
            d_batch[i, :d_len] = d_emb.to(device=device, dtype=torch.float32)
            q_mask[i, :q_len] = True
            d_mask[i, :d_len] = True

        # 批量计算 token 相似度
        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        # 掩码无效的文档 token
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        # 对每个查询 token 取最大值
        max_per_query = token_scores.amax(dim=-1)
        # 掩码无效的查询 token
        max_per_query.masked_fill_(~q_mask, 0.0)
        # 求和得到最终分数
        batch_scores = max_per_query.sum(dim=-1)
        scores.extend(batch_scores.unbind(0))
        start = end

    return scores
