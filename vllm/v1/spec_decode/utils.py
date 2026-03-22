# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""推测解码工具函数模块。

本模块提供了 EAGLE 推测解码所需的工具函数和 Triton kernel，负责：
- 槽映射计算和更新
- 输入准备和扩展
- Token 索引计算
- 注意力元数据管理
- 草稿模型配置创建

主要函数：
- eagle_step_update_slot_mapping_and_metadata: 更新 EAGLE 步骤的槽映射和元数据
- eagle_prepare_inputs_padded_kernel: 准备填充模式的输入
- eagle_prepare_next_token_padded_kernel: 准备下一个 token
- compute_new_slot_mapping: 计算新的槽映射
- create_vllm_config_for_draft_model: 创建草稿模型的 vllm 配置
- extend_all_queries_by_N: 扩展所有查询长度
- copy_and_expand_eagle_inputs_kernel: 复制和扩展 EAGLE 输入
"""

import torch

from vllm.config import VllmConfig, replace
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

PADDING_SLOT_ID = -1


@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    positions_ptr,  # [batch_size] - 当前位置（M-RoPE 的 1D 视图）
    block_table_ptr,  # [batch_size, n_blocks_per_req]
    block_table_stride,  # block_table 第一维的步长
    seq_lens_ptr,  # [batch_size] - 读写
    out_clamped_positions_ptr,  # [batch_size] (输出)
    out_slot_mapping_ptr,  # [input_batch_size] (输出)
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    n_blocks_per_req: tl.constexpr,
    PAD_ID: tl.constexpr,
    batch_size,
):
    """EAGLE 自回归步骤的融合 kernel。

    在单个 kernel 中更新位置、槽映射和序列长度，以减少启动开销。

    使用 input_batch_size 个线程启动。req_idx >= batch_size 的线程是
    CUDA 图填充槽位，只写入 PADDING_SLOT_ID。

    每个真实线程处理批次中的一个请求。计算：
    - new_position = position + 1，如果超过 max_model_len 则钳制
    - 从块表查找计算 slot_mapping
    - seq_lens += 1，如果超过最大值则为 1

    Args:
        positions_ptr: 当前位置指针
        block_table_ptr: 块表指针
        block_table_stride: 块表步长
        seq_lens_ptr: 序列长度指针
        out_clamped_positions_ptr: 输出钳制位置指针
        out_slot_mapping_ptr: 输出槽映射指针
        block_size: KV 缓存块大小
        max_model_len: 最大模型长度
        n_blocks_per_req: 每个请求的块数
        PAD_ID: 填充槽位 ID
        batch_size: 批次大小
    """
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
        return

    # 加载当前位置并递增
    position = tl.load(positions_ptr + req_idx)
    new_position = position + 1

    # 检查边界并计算钳制位置
    exceeds_max = new_position >= max_model_len
    clamped_position = tl.where(exceeds_max, 0, new_position)

    # 块表查找：block_number = position // block_size
    # 钳制 block_number 避免越界
    block_number = clamped_position // block_size
    block_number = tl.minimum(block_number, n_blocks_per_req - 1)

    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
    slot_id = block_id * block_size + (clamped_position % block_size)
    slot_id = tl.where(exceeds_max, PAD_ID, slot_id)

    # 更新 seq_lens：通常 +1，如果超过则为 1
    seq_len = tl.load(seq_lens_ptr + req_idx)
    new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
    new_seq_len = tl.minimum(new_seq_len, max_model_len)

    # 存储输出
    tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
    tl.store(out_slot_mapping_ptr + req_idx, slot_id)
    tl.store(seq_lens_ptr + req_idx, new_seq_len)


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    """为 EAGLE 自回归步骤融合更新槽映射和元数据。

    原地更新 seq_lens，写入 out_clamped_positions 和 out_slot_mapping。

    当 input_batch_size > batch_size 时，超出 batch_size 的线程写入
    PADDING_SLOT_ID 到 out_slot_mapping 用于 CUDA 图填充。

    Args:
        positions_1d: [batch_size] 当前位置（对 M-RoPE 使用 positions[0]）
        block_table_tensor: [batch_size, n_blocks_per_req] 块表张量
        seq_lens: [batch_size] 原地更新
        block_size: KV 缓存块大小
        max_model_len: 最大模型长度用于钳制
        out_clamped_positions: [batch_size] 钳制位置输出缓冲区
        out_slot_mapping: [input_batch_size] 槽映射输出缓冲区
        input_batch_size: 包含 CUDA 图填充的总批次大小；
            默认为 batch_size（无填充）
    """
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size
    n_blocks_per_req = block_table_tensor.shape[1]

    eagle_step_slot_mapping_metadata_kernel[(input_batch_size,)](
        positions_1d,
        block_table_tensor,
        block_table_tensor.stride(0),
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        block_size=block_size,
        max_model_len=max_model_len,
        n_blocks_per_req=n_blocks_per_req,
        PAD_ID=PADDING_SLOT_ID,
        batch_size=batch_size,
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (输出)
    num_rejected_tokens_gpu_ptr,  # [num_reqs] (输出)
    num_reqs,  # tl.int32
):
    """EAGLE prepare_input_padded 的融合 kernel。

    此 kernel 为每个请求计算要采样的 token 索引，
    考虑草稿 token 数量和有效采样 token 数量
    （比接受 token 数量多一个）。

    Args:
        cu_num_draft_tokens_ptr: 累积草稿 token 数量指针
        valid_sampled_tokens_count_ptr: 有效采样 token 数量指针
        query_start_loc_gpu_ptr: 查询起始位置指针
        token_indices_to_sample_ptr: 要采样的 token 索引输出指针
        num_rejected_tokens_gpu_ptr: 被拒绝 token 数量输出指针
        num_reqs: 请求数量
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # 从 cu_num_draft_tokens 计算 num_draft_tokens，
    # 这是包含性累积和（第一项是第一个值，不是零）
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] 是下一个请求的起始位置，
    # 即当前请求最后一个 token 之后的位置
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (输出)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (输出)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (第一维的步长)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # 2 的幂 >= num_sampled_tokens_per_req
):
    """EAGLE prepare_next_token_ids_padded 的融合 kernel。

    此 kernel 计算每个请求的有效（1 + 接受）token 数量，
    以及推测解码期间要采样的对应"下一个"token ID。
    这是来自采样 token 的"最后一个接受的 token"，
    或者如果没有接受 token 或请求被标记为丢弃则使用备份 token。

    Args:
        sampled_token_ids_ptr: 采样 token ID 指针
        discard_request_mask_ptr: 丢弃请求掩码指针
        backup_next_token_ids_ptr: 备份下一个 token ID 指针
        next_token_ids_ptr: 下一个 token ID 输出指针
        valid_sampled_tokens_count_ptr: 有效采样 token 数量输出指针
        vocab_size: 词表大小
        num_sampled_tokens_per_req: 每个请求的采样 token 数量
        num_reqs: 请求数量
        stride_sampled_token_ids: 采样 token ID 的步长
        BLOCK_SIZE_TOKENS: token 维度的块大小（2 的幂）
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # 检查此请求是否被丢弃
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # 计算采样 token 中的有效 token 数量
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # 被拒绝的 token 是 -1，有效 token 在 [0, vocab_size) 范围内
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # 保证有定义，因为 valid_count > 0 意味着 is_valid_mask 非空
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # 选择该索引处的 token，使用求和技巧
            # 因为我们不想再次加载来访问 token_ids[last_valid_index]
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # 没有有效 token，使用备份 token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
):
    """计算新的槽映射。

    基于新位置和拒绝掩码计算槽映射。

    Args:
        cad: 通用注意力元数据
        new_positions: 新位置张量
        is_rejected_token_mask: 被拒绝 token 掩码
        block_size: KV 缓存块大小
        num_new_tokens: 新 token 数量
        max_model_len: 最大模型长度

    Returns:
        新的槽映射张量
    """
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        cad.naive_query_lens() + num_new_tokens,
        output_size=len(new_positions),
    )
    # 钳制位置以防止索引 block_table_tensor 时越界
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # 屏蔽超出最大模型长度的位置 ID
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # 屏蔽被拒绝的 token 以防止保存到 KV 缓存
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """为草稿模型创建 vllm 配置。

    vllm_config 是为目标模型配置的，例如其 quant_config 和 parallel_config。
    但草稿模型可能使用不同的量化方式和不同的 tensor_parallel_size。
    此函数创建一个为草稿模型配置的新 vllm_config。
    vllm_config 在通过 get_model() 加载草稿模型时有用。

    Args:
        target_model_vllm_config: 目标模型的 vllm 配置

    Returns:
        为草稿模型配置的 vllm 配置
    """
    old = target_model_vllm_config
    assert old.speculative_config is not None, "speculative_config is not set"
    old_spec_config = old.speculative_config
    new_parallel_config = replace(
        old_spec_config.draft_parallel_config, rank=old.parallel_config.rank
    )
    new: VllmConfig = replace(
        old,
        quant_config=None,
        parallel_config=new_parallel_config,
        model_config=old_spec_config.draft_model_config,
    )
    return new


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """创建所有查询长度增加 N 的新 CommonAttentionMetadata。

    所有序列长度也增加 N。
    这在并行推测解码中很有用，我们将每个序列扩展 N 个 token
    并在一次传递中预测所有 token。
    槽映射在外部计算，因为它需要更多信息。

    Args:
        common_attn_metadata: 通用注意力元数据
        N: 要增加的长度
        arange: 范围张量
        new_slot_mapping: 新的槽映射

    Returns:
        扩展后的通用注意力元数据
    """
    cad = common_attn_metadata
    # query start loc 必须增加 [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # 每个请求增加 N 个 token -> 增加 batch_size * N 个 token
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # 所有查询长度增加 N，所以最大查询长度增加 N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
    )
    return new_cad


# 统一的复制/扩展 kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # 来自目标模型的（填充）输入
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # 写入草稿缓冲区的输出
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (输出)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (输出)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (输出)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (输出)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (输出)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # 输入元数据
    query_start_loc_ptr,  # [num_reqs + 1], 最后一个值是总输入 token 数量
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # 尺寸信息
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # 沿 token 维度的块大小用于处理预填充
):
    """复制和扩展输入从目标模型到草稿缓冲区。

    用于 EAGLE 推测解码。此 kernel 处理填充槽位和并行推测 token（如果启用）。

    输出布局：
    - [0, num_valid_tokens): 从输入复制的有效 token
    - [num_valid_tokens]: 来自 next_token_ids 的 bonus token
    - (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
        并行推测槽位
    - [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
        被拒绝的槽位

    Args:
        target_token_ids_ptr: 目标 token ID 指针
        target_positions_ptr: 目标位置指针
        next_token_ids_ptr: 下一个 token ID 指针
        out_input_ids_ptr: 输出 input IDs 指针
        out_positions_ptr: 输出位置指针
        out_is_rejected_token_mask_ptr: 输出被拒绝 token 掩码指针
        out_is_masked_token_mask_ptr: 输出被屏蔽 token 掩码指针
        out_new_token_indices_ptr: 输出新 token 索引指针
        out_hidden_state_mapping_ptr: 输出隐藏状态映射指针
        query_start_loc_ptr: 查询起始位置指针
        query_end_loc_ptr: 查询结束位置指针
        padding_token_id: 填充 token ID
        parallel_drafting_token_id: 并行推测 token ID
        total_input_tokens: 总输入 token 数量
        num_padding_slots_per_request: 每个请求的填充槽位数量
        shift_input_ids: 是否移位 input IDs
        BLOCK_SIZE_TOKENS: token 维度的块大小（ constexpr）
    """
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # 加载查询位置
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # 计算要复制的有效 token 数量和输入偏移
    # shift_input_ids=True 时，我们跳过第一个 token
    # 每个请求获得 (input_len + num_padding_slots_per_request) 个输出槽位
    # 但有移位时，每个请求失去一个 token
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # 来自之前推测的被拒绝 token 数量
    num_rejected = next_query_start_loc - query_end_loc - 1

    # 此请求的总输出 token 数量
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # 在此块中处理 token
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # 计算不同输出区域的掩码：
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # 计算输出索引
    out_idx = output_start + j

    # 对于有效 token，计算输入索引
    in_idx = query_start_loc + input_offset + j
    # 钳制避免越界访问（掩码加载仍需要有效地址）
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # 加载输入 token（掩码到有效区域）
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # 加载此请求的起始位置（序列中的第一个位置）
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # 加载此请求的 bonus token
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # 基于区域构建最终的 token_ids
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # 构建最终位置：
    # 位置不移位 - 从第一个输入位置开始递增
    # 输出位置 j 获得 start_pos + j
    # （例如，输入位置 [5,6,7] -> 输出 [5,6,7,8,9,...]）
    positions = start_pos + j
    # 被拒绝的位置是无关紧要的，设置为 0
    positions = tl.where(is_rejected_region, 0, positions)

    # 计算输出掩码
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # 计算新 token 的索引（bonus + 并行推测）用于采样
    # 新 token 在位置
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # bonus 为 0，并行推测为 1, 2, ...
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # 计算隐藏状态映射（源索引 -> 目标索引）
    # 这将每个输入位置映射到其对应的输出位置
    # 隐藏状态不移位，所以我们映射所有输入 token（包括被拒绝的）
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # 存储输出
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )
