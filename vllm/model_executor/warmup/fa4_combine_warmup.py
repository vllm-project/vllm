# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the FA4 split-KV combine kernel."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        FA4MLAPrefillKernel,
    )


def warm_split_kv_combine(
    compile_keys: "Iterable[FA4MLAPrefillKernel.CompileKey]",
) -> None:
    """Compile the split-KV combine kernel, which compile-only forward
    warmup skips. Mirror FA4's call at the end of _flash_attn_fwd, not the
    public flash_attn_combine: the cache key also covers varlen, the dynamic
    split count from the scheduler metadata and no seqused / virtual batch
    index. FA4's num_splits_heuristic caps auto splits at
    num_SMs // num_heads, which is <= 32 for >= 5 heads per rank on B200, so
    ceil(log2(num_splits)) is always the kernel's floor of 5.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd_combine

    keys = list(compile_keys)
    combine_configs = {
        (key.q_dtype, key.v_shape[-1], key.q_shape[1])
        for key in keys
        if key.num_splits != 1
    }
    device = torch.device(current_platform.device_type)
    # FakeTensorMode populates the native cache without launching the kernel.
    with FakeTensorMode():
        for dtype, v_head_dim, num_heads in combine_configs:
            for return_lse in (True, False):
                _flash_attn_fwd_combine(
                    torch.empty(
                        (2, 1, num_heads, v_head_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        (2, num_heads, 1), dtype=torch.float32, device=device
                    ).transpose(-1, -2),
                    torch.empty((1, num_heads, v_head_dim), dtype=dtype, device=device),
                    torch.empty(
                        (num_heads, 1), dtype=torch.float32, device=device
                    ).transpose(-1, -2)
                    if return_lse
                    else None,
                    cu_seqlens=torch.empty((2,), dtype=torch.int32, device=device),
                    num_splits_dynamic_ptr=torch.empty(
                        (1,), dtype=torch.int32, device=device
                    ),
                )
