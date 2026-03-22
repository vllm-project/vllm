# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""请求状态管理模块。

本模块提供请求状态管理类，负责：
- 管理每个请求的 token ID 序列
- 跟踪 prompt 长度、prefill 长度和总长度
- 管理已计算的 token 数量
- 管理采样和 draft token

主要类：
- RequestState: 请求状态管理类
"""
import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RequestState:
    """请求状态管理类。

    管理每个请求的状态信息，包括 token ID 序列、长度信息和采样状态。
    使用 UVA（Unified Virtual Addressing）来节省 GPU 内存。

    Attributes:
        max_num_reqs: 最大请求数量
        max_model_len: 最大模型长度
        max_num_batched_tokens: 最大批次 token 数
        num_speculative_steps: 推测解码步数
        vocab_size: 词表大小
        device: 设备类型
        req_id_to_index: 请求 ID 到索引的映射
        index_to_req_id: 索引到请求 ID 的映射
        free_indices: 空闲索引列表
        all_token_ids: 所有 token ID（UVA  backed）
        prompt_len: prompt 长度（UVA backed）
        prefill_len: prefill 长度（UVA backed）
        total_len: 总长度（staged write）
        num_computed_prefill_tokens: 已计算的 preflight token 数量（numpy）
        num_computed_tokens: 已计算的 token 数量（staged write）
        last_sampled_tokens: 最后采样的 token
        draft_tokens: draft token
        next_prefill_tokens: 下一个 prefill token
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ):
        """初始化请求状态管理。

        Args:
            max_num_reqs: 最大请求数量
            max_model_len: 最大模型长度
            max_num_batched_tokens: 最大批次 token 数
            num_speculative_steps: 推测解码步数
            vocab_size: 词表大小
            device: 设备类型
        """
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # NOTE(woosuk): 此张量可能非常大（例如数 GB）
        # 取决于配置的 max_num_reqs 和 max_model_len。
        # 为节省 GPU 内存，我们使用 UVA 而不是 GPU 来存储此张量。
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        # NOTE(woosuk): 明确区分 prompt_len 和 prefill_len：
        # - prompt_len：用户提供的 prompt 中的 token 数量。
        # - prefill_len：传入 model runner 的 token 数量。
        #   这可能包括 prompt 和额外的部分输出 token，
        #   因此 prefill_len >= prompt_len。
        # 通常 prefill_len 等于 prompt_len，但在抢占后恢复等情况下，
        # prefill_len 可能更大。区分这些值至关重要，因为某些功能
        #（如 prompt logprobs 或 frequency penalties）必须分别处理
        # prompt 和 output token。
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len，随着请求的进行而增长
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # 已计算的 token 数量
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # 最后采样的 token
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Draft token
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

    @property
    def num_reqs(self) -> int:
        """返回当前请求数量。

        Returns:
            当前请求数量
        """
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        """添加新请求。

        Args:
            req_id: 请求 ID
            prompt_len: prompt 长度
            all_token_ids: 所有 token ID 列表
            num_computed_tokens: 已计算的 token 数量
        """
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

    def apply_staged_writes(self) -> None:
        """应用所有暂存的写入操作。

        将暂存的数据复制到 UVA 或应用 staged write。
        """
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> bool:
        """移除请求。

        Args:
            req_id: 请求 ID

        Returns:
            如果成功移除则返回 True，请求不存在则返回 False
        """
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            # 请求未找到
            return False
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return True

    def any_prefills(self, idx_mapping_np: np.ndarray) -> bool:
        """检查是否有任何请求仍在 prefill 阶段。

        Args:
            idx_mapping_np: 索引映射 numpy 数组

        Returns:
            如果有任何请求仍在 prefill 阶段则返回 True
        """
        return np.any(
            self.num_computed_prefill_tokens[idx_mapping_np]
            < self.prefill_len.np[idx_mapping_np]
        )
