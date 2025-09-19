# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed import tensor_model_parallel_all_gather


def pad_and_all_gather(
    x: torch.Tensor,
    padded_size: int,
) -> torch.Tensor:
    n = x.shape[0]
    if n != padded_size:
        padded_x = torch.empty(
            (padded_size, *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
        padded_x[:n] = x
    else:
        padded_x = x

    x = tensor_model_parallel_all_gather(padded_x)
    return x
