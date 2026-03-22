# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""后缀解码（Suffix Decoding）推测解码 proposer 模块。

本模块实现了基于后缀树的推测解码 proposer，负责：
- 使用后缀树缓存历史请求的输出
- 基于当前序列后缀在后缀树中查找匹配
- 动态生成可变数量的草稿 token

主要类：
- SuffixDecodingProposer: 后缀解码推测 proposer

算法说明：
后缀解码（https://arxiv.org/pdf/2411.04975）是一种基于历史缓存的
推测解码方法。它维护一个后缀树，缓存之前请求的输出序列。
当新请求到来时，通过后缀树查找与当前序列后缀匹配的路径，
并使用匹配路径后的 token 作为草稿 token。

该方法特别适合重复模式多的场景，如代码生成、对话等。
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """后缀解码（Suffix Decoding）的推测解码 proposer。

    后缀解码是一种基于历史缓存的推测解码方法，参考论文：
    https://arxiv.org/pdf/2411.04975

    该类导入并使用 Arctic Inference 的官方实现：
    https://github.com/snowflakedb/ArcticInference

    SuffixDecodingCache 负责：
    - 缓存请求的输出序列
    - 驱逐旧的请求以释放空间
    - 管理每个 prompt 的后缀树

    Attributes:
        num_speculative_tokens: 最大草稿 token 数量
        max_tree_depth: 后缀树最大深度
        max_spec_factor: 最大推测因子
        min_token_prob: 最小 token 概率阈值
        max_model_len: 模型最大长度
        suffix_cache: 后缀缓存对象
    """

    def __init__(self, vllm_config: VllmConfig):
        """初始化 SuffixDecodingProposer。

        Args:
            vllm_config: vLLM 配置

        Raises:
            AssertionError: 如果 speculative_config 未设置
        """
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # 延迟导入以避免在未使用后缀解码时报错
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # 初始化后缀缓存
        # 该对象负责缓存请求输出、驱逐旧请求、管理每个 prompt 的后缀树
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """为输入批次中的每个请求提议草稿 token。

        后缀解码会为每个请求在每个解码步骤中动态生成不同数量的
        token，因此返回列表中每个元素的长度可能不同。

        Args:
            input_batch: GPU 输入批次
            sampled_token_ids: 采样的 token ID 列表
            slot_mappings: 槽映射（未使用）

        Returns:
            每个请求的草稿 token ID 列表
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # 跳过部分预填充请求的推测解码
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # 跳过已达到最大模型长度的请求
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # 重置该请求的后缀缓存
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # 启动新请求，这将构建该 prompt 的后缀树
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # 将新采样的 token 添加到该请求的后缀缓存
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # 后缀解码只使用最多 max_tree_depth 的最新 token，
            # 所以我们从输入末尾提取模式
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]

            # 使用后缀缓存进行推测
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # 停止未在当前输入批次中出现的请求
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """加载模型（后缀解码 proposer 无需加载模型）。"""
        # 无需加载模型
        pass
