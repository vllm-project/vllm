# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU 加速 N-gram 推测解码 proposer 模块。

本模块实现了基于 N-gram 匹配的 GPU 加速推测解码 proposer，负责：
- 使用完全向量化的 PyTorch 张量操作进行 N-gram 匹配
- 使用 unfold 和 argmax 并行查找所有序列的首次匹配
- 支持 torch.compile 编译优化

主要类：
- NgramGPUKernel: GPU 加速的 N-gram 内核（支持 torch.compile）
- NgramProposerGPU: GPU N-gram proposer

主要函数：
- update_scheduler_for_invalid_drafts: 更新调度器中无效的草稿
- update_ngram_gpu_tensors_incremental: 增量更新 GPU tensor
- copy_num_valid_draft_tokens: 异步复制有效草稿数量

算法说明：
GPU 版本使用完全向量化的张量操作，通过 unfold 创建滑动窗口，
并行比较所有序列的所有可能 N-gram 长度，使用 argmax 查找首次匹配。
相比 CPU 版本，GPU 版本适合大批量场景，可充分利用 GPU 并行性。
"""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@support_torch_compile()
class NgramGPUKernel(nn.Module):
    """GPU 加速的 N-gram proposer，使用完全向量化的张量操作。

    该类实现了 N-gram 匹配的 GPU 内核，支持 torch.compile 编译优化。
    通过 unfold 创建滑动窗口，并行比较所有序列的所有 N-gram 长度，
    使用 argmax 查找首次匹配位置。

    Attributes:
        min_n: N-gram 最小长度
        max_n: N-gram 最大长度
        k: 草稿 token 数量
        max_model_len: 模型最大长度
        max_num_seqs: 最大序列数量
        device: 设备（CUDA）
    """

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", device: torch.device = "cuda"
    ):
        """初始化 NgramGPUKernel。

        Args:
            vllm_config: vLLM 配置
            prefix: 前缀名称（未使用）
            device: 设备（CUDA）

        Raises:
            AssertionError: 如果 speculative_config、prompt_lookup_min、
                prompt_lookup_max 未设置
        """
        super().__init__()

        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

    def _find_first_and_extract_all_n_parallel(
        self,
        token_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
        min_ngram_len: int,
        max_ngram_len: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """查找后缀 N-gram 匹配并提取后续 token。

        搜索每个序列后缀在之前出现的最早位置，尝试多种长度，
        并选择最长的有效匹配。

        Args:
            token_ids: 每个序列的 token ID [batch_size, seq_len]
            seq_lengths: 每个序列的实际长度（不包括填充）[batch_size]
            min_ngram_len: 搜索的最小 N-gram 长度（如 2）
            max_ngram_len: 搜索的最大 N-gram 长度（如 5）
            num_draft_tokens: 匹配后提取的 token 数量（k）

        Returns:
            草稿 token 预测，-1 表示无效/无匹配 [batch_size, num_draft_tokens]
        """
        batch_size = token_ids.shape[0]
        max_seq_len = token_ids.shape[1]
        device = token_ids.device
        num_ngram_sizes = max_ngram_len - min_ngram_len + 1

        # 尝试的所有 N-gram 长度
        ngram_lengths = torch.arange(min_ngram_len, max_ngram_len + 1, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # 每个（序列，N-gram 长度）的首次匹配位置，-1 表示无匹配
        first_match_positions = torch.full(
            (batch_size, num_ngram_sizes), -1, dtype=torch.long, device=device
        )

        # 遍历每个 N-gram 长度
        for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
            # 大小为 ngram_len 的滑动窗口；unfold 是 O(1) 视图操作
            search_windows = token_ids.unfold(1, ngram_len, 1)
            num_windows = search_windows.shape[1]

            # 每个序列的后缀（最后 ngram_len 个 token）
            suffix_starts = seq_lengths - ngram_len
            suffix_indices = suffix_starts.unsqueeze(1) + torch.arange(
                ngram_len, device=device
            )
            suffix = torch.gather(token_ids, 1, suffix_indices.clamp(min=0))

            # 窗口匹配
            matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

            # 匹配位置必须至少留下一个草稿 token 的空间
            max_valid_suffix_start = seq_lengths - ngram_len - 1
            window_positions = torch.arange(num_windows, device=device)
            valid_mask = window_positions <= max_valid_suffix_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # 查找首次匹配（argmax=0 当为空时；用 has_match 验证）
            first_match_idx = torch.argmax(final_matches.int(), dim=1)
            has_match = final_matches[batch_indices, first_match_idx]

            # 存储有效的匹配位置（窗口索引 = 位置）
            first_match_positions[:, i] = torch.where(has_match, first_match_idx, -1)

        # 选择有匹配的最长 N-gram
        best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx  # 翻转回来

        # 最佳 N-gram 的匹配位置
        best_match_pos = first_match_positions[batch_indices, best_ngram_idx]

        # 避免数据依赖分支
        has_any_match = best_match_pos >= 0

        # 最佳匹配 N-gram 的长度
        best_ngram_lengths = ngram_lengths[best_ngram_idx]

        # 匹配后的起始位置
        draft_start = torch.where(
            has_any_match,
            best_match_pos + best_ngram_lengths,
            torch.zeros_like(best_match_pos),
        )
        tokens_available = seq_lengths - draft_start

        # 草稿 token 的索引
        draft_indices = draft_start.unsqueeze(1) + torch.arange(
            num_draft_tokens, device=device
        )
        draft_indices = draft_indices.clamp(min=0, max=max_seq_len - 1)

        # 提取草稿 token
        draft_tokens = torch.gather(token_ids, 1, draft_indices)

        # 掩码超出可用 token 数量的位置
        position_indices = torch.arange(num_draft_tokens, device=device).unsqueeze(0)
        valid_positions = position_indices < tokens_available.unsqueeze(1)

        draft_tokens = torch.where(
            valid_positions,
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        # 如果没有匹配，掩码所有位置
        draft_tokens = torch.where(
            has_any_match.unsqueeze(1),
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        return draft_tokens

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N-gram 提议的 GPU 前向传播。

        Args:
            num_tokens_no_spec: 每个序列的 token 数量 [batch_size]
            token_ids_gpu: token ID [batch_size, max_len]
            combined_mask: 每个序列是否有效进行推测解码 [batch_size]

        Returns:
            二元组：
                - draft_tokens: 草稿 token [batch_size, k]，在 GPU 上
                - num_valid_draft_tokens: 每个请求的有效草稿数量 [batch_size]，
                    int32，在 GPU 上，表示每个请求中连续有效（非 -1）token 数量
        """

        device = token_ids_gpu.device

        # 推断批次大小以保持动态形状
        actual_batch_size = token_ids_gpu.shape[0]

        # 在前向传播中分配，让 torch.compile 可以优化
        # 注意：不要预先分配为缓冲区，这会破坏 torch.compile
        draft_tokens = torch.full(
            (actual_batch_size, self.k), -1, dtype=torch.int32, device=device
        )

        # 执行 N-gram 匹配
        results = self._find_first_and_extract_all_n_parallel(
            token_ids_gpu,
            num_tokens_no_spec,
            min_ngram_len=self.min_n,
            max_ngram_len=self.max_n,
            num_draft_tokens=self.k,
        )

        # 应用掩码
        draft_tokens = torch.where(combined_mask.unsqueeze(1), results, -1)

        # 计算每个请求中连续有效（非 -1）token 的数量
        is_valid = draft_tokens != -1  # [batch, k]
        cum_valid = is_valid.int().cumsum(dim=1)  # [batch, k]
        positions = torch.arange(1, self.k + 1, device=device).unsqueeze(0)
        num_valid_draft_tokens = (cum_valid == positions).int().sum(dim=1)

        return draft_tokens, num_valid_draft_tokens

    def load_model(self, *args, **kwargs):
        """N-gram proposer 无需加载模型。"""
        pass


class NgramProposerGPU:
    """GPU 加速的 N-gram 推测解码 proposer。

    该类封装了 NgramGPUKernel，提供完整的推测解码 proposer 功能。
    支持 torch.compile 编译优化和 CUDA Graphs。

    Attributes:
        vllm_config: vLLM 配置
        min_n: N-gram 最小长度
        max_n: N-gram 最大长度
        k: 草稿 token 数量
        max_model_len: 模型最大长度
        max_num_seqs: 最大序列数量
        device: 设备（CUDA）
        kernel: GPU 内核
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        """初始化 NgramProposerGPU。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）
            runner: 模型运行器（未使用）

        Raises:
            AssertionError: 如果 speculative_config、prompt_lookup_min、
                prompt_lookup_max 未设置
        """
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # 编译配置
        compilation_config = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"],
            splitting_ops=[],
            compile_sizes=[],
            inductor_compile_config={
                "enable_auto_functionalized_v2": False,
                "max_autotune": True,
                "aggressive_fusion": True,
                "triton.autotune_pointwise": True,
                "coordinate_descent_tuning": True,
                "use_mixed_mm": False,
            },
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        model_config = vllm_config.model_config
        speculative_config = vllm_config.speculative_config
        scheduler_config = vllm_config.scheduler_config

        self.vllm_config = VllmConfig(
            compilation_config=compilation_config,
            model_config=model_config,
            speculative_config=speculative_config,
            scheduler_config=scheduler_config,
        )

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

        # 创建并初始化内核
        self.kernel = NgramGPUKernel(
            vllm_config=self.vllm_config, prefix="ngram_gpu_kernel", device=device
        )
        self.kernel.to(device)
        self.kernel.eval()

        # 运行虚拟推理初始化
        self._dummy_run()

    def _dummy_run(self):
        """运行虚拟推理以初始化模型。"""
        token_ids, num_tokens, sampled_flags, valid_mask = self._generate_dummy_data(
            batch_size=self.max_num_seqs,
            max_seq_len=self.max_model_len,
            pattern_len=self.k,
            device=self.device,
        )

        combined_mask = sampled_flags & valid_mask & (num_tokens >= self.min_n)

        for _ in range(3):
            with set_forward_context(None, self.vllm_config):
                _, _ = self.kernel(num_tokens, token_ids, combined_mask)

    def _generate_dummy_data(
        self,
        batch_size: int,
        max_seq_len: int,
        pattern_len: int,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成包含 N-gram 重复模式的随机测试数据。

        Args:
            batch_size: 批次中的序列数量
            max_seq_len: 最大序列长度
            pattern_len: 注入用于匹配的模式长度
            device: 放置数据的设备

        Returns:
            四元组：
                - token_ids: [batch_size, max_seq_len] token ID
                - num_tokens: [batch_size] 每个序列的 token 数量
                - sampled_flags: [batch_size] 采样标志
                - valid_mask: [batch_size] 有效掩码
        """
        token_ids = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )

        num_tokens = torch.randint(
            pattern_len, max_seq_len, (batch_size,), dtype=torch.int32, device=device
        )

        sampled_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        return token_ids, num_tokens, sampled_flags, valid_mask

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用 GPU 加速的 N-gram 匹配生成草稿 token。

        将采样的 token 分散到 token_ids_gpu 中，计算临时更新的长度，
        然后运行内核。

        Args:
            num_tokens_no_spec: 每个序列的 token 数量（只读）[batch_size]
            token_ids_gpu: token ID 张量（原地修改，写入新 token）
            valid_sampled_token_ids_gpu: 要分散的新采样 token
            valid_sampled_tokens_count: 每个序列的有效 token 数量

        Returns:
            二元组：
                - draft_tokens: 提议的草稿 token ID [batch_size, k]
                - num_valid_draft_tokens: 每个请求的有效草稿数量 [batch_size]
        """
        assert token_ids_gpu.device == self.device
        assert num_tokens_no_spec.device == self.device

        batch_size = num_tokens_no_spec.shape[0]
        max_seq_len = token_ids_gpu.shape[1]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]  # num_spec_tokens + 1

        # 将新采样的 token 分散到 token_ids_gpu 中
        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(
            1
        )
        in_bounds = write_positions < max_seq_len
        scatter_mask = (
            valid_write_mask & (valid_sampled_token_ids_gpu != -1) & in_bounds
        )

        write_positions_long = write_positions.clamp(max=max_seq_len - 1).long()
        existing_values = token_ids_gpu.gather(1, write_positions_long)

        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        tokens_to_scatter = torch.where(
            scatter_mask,
            tokens_cast,
            existing_values,
        )
        token_ids_gpu.scatter_(1, write_positions_long, tokens_to_scatter)

        # 计算临时 token 数量
        num_tokens_tmp = (num_tokens_no_spec + valid_sampled_tokens_count).to(
            torch.int32
        )

        # 计算有效性掩码
        sampled_flags = valid_sampled_tokens_count > 0
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        with set_forward_context(None, self.vllm_config):
            combined_mask = sampled_flags & valid_mask & (num_tokens_tmp >= self.min_n)

            with record_function_or_nullcontext("ngram_proposer_gpu: kernel"):
                draft_tokens, num_valid_draft_tokens = self.kernel(
                    num_tokens_tmp,
                    token_ids_gpu,
                    combined_mask,
                )

            return draft_tokens, num_valid_draft_tokens

    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在设备上准备推测解码输入。

        计算下一个 token ID 和有效数量，尊重被丢弃的请求
        和被拒绝的 token，无需 CPU-GPU 同步。

        Args:
            sampled_token_ids: 采样的 token ID（张量或列表）
            gpu_input_batch: GPU 输入批次
            token_ids_gpu: token ID 张量
            num_tokens_no_spec: 每个请求的 token 数量（无推测）
            discard_request_mask: 丢弃请求掩码

        Returns:
            三元组：
                - next_token_ids: 下一个 token ID
                - valid_sampled_tokens_count: 有效采样 token 数量
                - valid_sampled_token_ids_gpu: 有效采样 token ID
        """
        num_reqs = gpu_input_batch.num_reqs

        if isinstance(sampled_token_ids, list):
            # 当 disable_padded_drafter_batch=True 时，sampled_token_ids 是
            # 不规则的 list[list[int]]，子列表可能有不同长度
            # （包括被丢弃请求的空列表）。
            # 将所有子列表填充为相同长度（用 -1），然后转换为张量
            max_len = max(
                (len(sublist) for sublist in sampled_token_ids),
                default=0,
            )
            # 确保至少长度 1 以创建张量
            max_len = max(max_len, 1)
            padded_list = [
                sublist + [-1] * (max_len - len(sublist))
                for sublist in sampled_token_ids
            ]
            sampled_token_ids = torch.tensor(
                padded_list, dtype=torch.int32, device=self.device
            )
        assert isinstance(sampled_token_ids, torch.Tensor), (
            "sampled_token_ids should be a torch.Tensor for ngram_gpu"
        )

        # 在推测 token 之前备份最后一个有效 token
        backup_indices = (num_tokens_no_spec[:num_reqs] - 1).clamp(min=0).long()
        backup_next_token_ids = torch.gather(
            token_ids_gpu[:num_reqs], dim=1, index=backup_indices.unsqueeze(1)
        ).squeeze(1)

        # 克隆采样 token ID
        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        # 使被丢弃请求的采样 token 无效
        discard_mask_expanded = discard_request_mask[:num_reqs].unsqueeze(1)
        valid_sampled_token_ids_gpu.masked_fill_(discard_mask_expanded, -1)

        # 掩码每个请求内的有效 token
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # 计算每个请求的有效 token 数量
        valid_sampled_tokens_count = valid_mask.sum(dim=1).to(torch.int32)

        # 每行最右侧有效索引
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # 从每行选择最后一个有效 token
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # 如果有效则使用最后一个 token，否则使用备份
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            backup_next_token_ids,
        )

        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        """加载模型（调用内核的 load_model）。"""
        self.kernel.load_model(*args, **kwargs)


def update_scheduler_for_invalid_drafts(
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens_cpu: torch.Tensor,
    scheduler_output: "SchedulerOutput",
    req_id_to_index: dict[str, int],
) -> None:
    """使用每个请求的有效草稿数量修剪无效的推测槽位。

    Args:
        num_valid_draft_tokens_event: 用于异步 D2H 完成的 Event
        num_valid_draft_tokens_cpu: 有效草稿数量的 CPU 缓冲区
        scheduler_output: 要原地更新的调度器元数据
        req_id_to_index: 请求 ID 到批次索引的映射
    """
    req_data = scheduler_output.scheduled_cached_reqs
    # 同步事件
    num_valid_draft_tokens_event.synchronize()

    for req_id in req_data.req_ids:
        req_index = req_id_to_index.get(req_id)
        if req_index is None:
            continue

        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        if spec_token_ids is None:
            continue

        scheduled_k = len(spec_token_ids)

        # 获取有效数量并确保在有效范围内
        valid_k = int(num_valid_draft_tokens_cpu[req_index].item())
        valid_k = max(0, min(valid_k, scheduled_k))

        # 计算要修剪的 token 数量
        tokens_to_trim = scheduled_k - valid_k
        scheduler_output.total_num_scheduled_tokens -= tokens_to_trim
        scheduler_output.num_scheduled_tokens[req_id] -= tokens_to_trim

        # 如果没有有效 token，移除该请求的推测 token
        if valid_k == 0:
            scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
        else:
            scheduler_output.scheduled_spec_decode_tokens[req_id] = spec_token_ids[
                :valid_k
            ]


def update_ngram_gpu_tensors_incremental(
    input_batch: InputBatch,
    token_ids_gpu_tensor: torch.Tensor,
    num_tokens_no_spec_gpu: torch.Tensor,
    new_reqs: list[CachedRequestState],
    device: torch.device,
    _pinned_idx_buf: torch.Tensor,
    _pinned_val_buf: torch.Tensor,
) -> None:
    """为 N-gram GPU proposer 增量更新 token_ids_gpu_tensor 和
    num_tokens_no_spec_gpu。

    Args:
        input_batch: 输入批次
        token_ids_gpu_tensor: token ID GPU 张量
        num_tokens_no_spec_gpu: 无推测 token 数量 GPU 张量
        new_reqs: 新请求列表
        device: 设备
        _pinned_idx_buf: 固定的索引缓冲区
        _pinned_val_buf: 固定的值缓冲区
    """
    prev_req_id_to_index = input_batch.prev_req_id_to_index
    curr_req_id_to_index = input_batch.req_id_to_index

    if not curr_req_id_to_index:
        return

    active_indices = list(curr_req_id_to_index.values())
    n_active = len(active_indices)

    # 使用驻留的固定缓冲区避免每次调用分配
    active_idx_cpu = _pinned_idx_buf[:n_active]
    active_idx_cpu.copy_(torch.as_tensor(active_indices, dtype=torch.long))

    active_idx_gpu = active_idx_cpu.to(device=device, non_blocking=True)

    new_req_ids = {req.req_id for req in new_reqs}

    # 第一次运行，没有之前的状态
    if prev_req_id_to_index is None:
        for idx in active_indices:
            num_tokens = input_batch.num_tokens_no_spec[idx]
            if num_tokens > 0:
                token_ids_gpu_tensor[idx, :num_tokens].copy_(
                    input_batch.token_ids_cpu_tensor[idx, :num_tokens],
                    non_blocking=True,
                )

        _sync_num_tokens(
            input_batch,
            num_tokens_no_spec_gpu,
            active_idx_cpu,
            active_idx_gpu,
            n_active,
            device,
            _pinned_val_buf,
        )
        return

    # 检测索引变化以进行重排序
    reorder_src: list[int] = []
    reorder_dst: list[int] = []

    for req_id, curr_idx in curr_req_id_to_index.items():
        if req_id in new_req_ids:
            continue
        prev_idx = prev_req_id_to_index.get(req_id)
        if prev_idx is not None and prev_idx != curr_idx:
            reorder_src.append(prev_idx)
            reorder_dst.append(curr_idx)

    if reorder_src:
        src_tensor = torch.tensor(reorder_src, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(reorder_dst, dtype=torch.long, device=device)

        temp_token_ids = token_ids_gpu_tensor[src_tensor].clone()
        temp_num_tokens = num_tokens_no_spec_gpu[src_tensor].clone()

        token_ids_gpu_tensor[dst_tensor] = temp_token_ids
        num_tokens_no_spec_gpu[dst_tensor] = temp_num_tokens

    # 为新/恢复的请求完整复制
    for req_state in new_reqs:
        new_req_idx = curr_req_id_to_index.get(req_state.req_id)
        if new_req_idx is None:
            continue

        num_tokens = input_batch.num_tokens_no_spec[new_req_idx]
        if num_tokens > 0:
            token_ids_gpu_tensor[new_req_idx, :num_tokens].copy_(
                input_batch.token_ids_cpu_tensor[new_req_idx, :num_tokens],
                non_blocking=True,
            )

    # 始终为所有活动请求批量同步序列长度
    _sync_num_tokens(
        input_batch,
        num_tokens_no_spec_gpu,
        active_idx_cpu,
        active_idx_gpu,
        n_active,
        device,
        _pinned_val_buf,
    )


def _sync_num_tokens(
    input_batch: InputBatch,
    num_tokens_no_spec_gpu: torch.Tensor,
    active_idx_cpu: torch.Tensor,
    active_idx_gpu: torch.Tensor,
    n_active: int,
    device: torch.device,
    _pinned_val_buf: torch.Tensor,
) -> None:
    """从 CPU 数据源批量同步 GPU 序列长度。

    Args:
        input_batch: 包含 CPU 长度张量的批次容器
        num_tokens_no_spec_gpu: 目标 GPU 长度张量
        active_idx_cpu: CPU 上的活动请求索引
        active_idx_gpu: GPU 上的活动请求索引
        n_active: 活动请求数量
        device: 目标 CUDA 设备
        _pinned_val_buf: 驻留的 int32 固定 staging 缓冲区
    """
    src_cpu = input_batch.num_tokens_no_spec_cpu_tensor
    vals = _pinned_val_buf[:n_active]
    vals.copy_(src_cpu.index_select(0, active_idx_cpu))

    num_tokens_no_spec_gpu.index_copy_(
        0,
        active_idx_gpu,
        vals.to(device=device, non_blocking=True),
    )


def copy_num_valid_draft_tokens(
    num_valid_draft_tokens_cpu: torch.Tensor,
    num_valid_draft_tokens_copy_stream: torch.cuda.Stream,
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens: torch.Tensor | None,
    batch_size: int,
) -> None:
    """异步 D2H 复制每个请求的有效草稿数量。

    Args:
        num_valid_draft_tokens_cpu: 目标 CPU 缓冲区
        num_valid_draft_tokens_copy_stream: 用于复制的 CUDA 流
        num_valid_draft_tokens_event: 记录完成的事件
        num_valid_draft_tokens: 源 GPU tensor
        batch_size: 批次大小
    """
    if num_valid_draft_tokens is None:
        return

    num_reqs_to_copy = min(batch_size, num_valid_draft_tokens.shape[0])
    if num_reqs_to_copy <= 0:
        return

    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(num_valid_draft_tokens_copy_stream):
        num_valid_draft_tokens_copy_stream.wait_stream(default_stream)
        num_valid_draft_tokens_cpu[:num_reqs_to_copy].copy_(
            num_valid_draft_tokens[:num_reqs_to_copy], non_blocking=True
        )
        num_valid_draft_tokens_event.record()
