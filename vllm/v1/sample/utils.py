# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def build_logprob_token_ids_matrix(
    logprob_token_ids: dict[int, list[int]],
    num_rows: int,
    sampled: torch.Tensor,
    device: torch.device,
    pin_memory: bool,
    rows_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded token_ids matrix and valid_mask for specific-token logprobs.

    Args:
        logprob_token_ids: dict mapping req_index -> list of token IDs
        num_rows: total number of rows in the output matrix
        sampled: [num_rows] tensor of sampled token IDs (on GPU)
        device: target device
        pin_memory: whether to use pinned memory for CPU tensors
        rows_per_req: number of rows each request occupies (1 for normal
            sampling, max_spec_len+1 for speculative decoding)

    Returns:
        token_ids_tensor: [num_rows, max_num_tokens+1] int64 tensor on device.
            Column 0 = sampled token, columns 1..N = logprob_token_ids (padded).
        valid_mask: [num_rows, max_num_tokens+1] bool tensor on device.
            True = valid position, False = padding.
    """
    max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())

    token_ids_cpu = torch.zeros(
        num_rows, max_num_tokens + 1, dtype=torch.int64, pin_memory=pin_memory
    )
    valid_mask_cpu = torch.zeros(
        num_rows, max_num_tokens + 1, dtype=torch.bool, pin_memory=pin_memory
    )
    valid_mask_cpu[:, 0] = True

    for req_idx, token_ids in logprob_token_ids.items():
        num_tokens = len(token_ids)
        token_ids_t = torch.as_tensor(token_ids, dtype=torch.int64)
        start_row = req_idx * rows_per_req
        end_row = min(start_row + rows_per_req, num_rows)
        token_ids_cpu[start_row:end_row, 1 : num_tokens + 1] = token_ids_t
        valid_mask_cpu[start_row:end_row, 1 : num_tokens + 1] = True

    token_ids_tensor = token_ids_cpu.to(device, non_blocking=True)
    valid_mask = valid_mask_cpu.to(device, non_blocking=True)
    token_ids_tensor[:, 0] = sampled

    return token_ids_tensor, valid_mask
