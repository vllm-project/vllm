# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)

def _min_p_torch(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """纯 torch 版本，替换 `_min_p_kernel`（平台不支持 triton）。

    对每个 token 取其请求的 min_p（min_p==0 的行跳过），在 log 空间计算阈值
    threshold = max(logits) + log(min_p)，把低于阈值的 logits 置为 -inf。
    """
    mp = min_p[expanded_idx_mapping.long()].to(torch.float32)  # [num_tokens]
    apply = mp != 0.0
    if not bool(apply.any()):
        return

    logits_f = logits.to(torch.float32)
    max_val = logits_f.max(dim=1, keepdim=True).values  # [num_tokens, 1]
    threshold = max_val + torch.log(mp).unsqueeze(1)  # [num_tokens, 1]

    drop = (logits_f < threshold) & apply.unsqueeze(1)
    logits[drop] = -float("inf")

def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _min_p_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    _min_p_torch(logits, expanded_idx_mapping, min_p)
