# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FA4 MLA prefill warmup's split-KV combine coverage."""

import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.model_executor.warmup import fa4_mla_prefill_warmup


@pytest.mark.skip_global_cleanup
def test_split_kv_combine_warmup_matches_runtime_compile_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_keys: list[tuple[Any, ...]] = []

    def _flash_attn_fwd_combine(
        out_partial: torch.Tensor,
        lse_partial: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        seqused: torch.Tensor | None = None,
        num_splits_dynamic_ptr: torch.Tensor | None = None,
        virtual_batch_idx: torch.Tensor | None = None,
        semaphore_to_reset: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # Compile-only: no real tensor is allocated and no kernel launches.
        assert isinstance(out_partial, FakeTensor)
        # compile_key of flash-attention 9cd61de cute/interface.py
        # _flash_attn_fwd_combine, without the arch and the head-dim-derived
        # tile_m=8 / k_block_size=128 (which raise log_max_splits to >= 5).
        compile_keys.append(
            (
                out.dtype,
                out_partial.dtype,
                out_partial.shape[-1],
                out_partial.shape[-2],
                max(math.ceil(math.log2(out_partial.shape[0])), 5),
                cu_seqlens is not None,
                seqused is not None,
                lse is not None,
                virtual_batch_idx is not None,
                num_splits_dynamic_ptr is not None,
                semaphore_to_reset is not None,
                output_scale is not None,
            )
        )

    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.cute.interface",
        SimpleNamespace(_flash_attn_fwd_combine=_flash_attn_fwd_combine),
    )
    monkeypatch.setattr(
        fa4_mla_prefill_warmup, "current_platform", SimpleNamespace(device_type="cpu")
    )
    keys: list[Any] = [
        SimpleNamespace(
            q_dtype=torch.bfloat16,
            q_shape=(8, num_heads, 192),
            v_shape=(8, num_heads, 128),
            num_splits=num_splits,
        )
        for num_heads, num_splits in ((16, 0), (32, 0), (64, 1))
    ]

    fa4_mla_prefill_warmup._warm_split_kv_combine(keys)

    # _flash_attn_fwd's split-KV combine call: varlen, scheduler-metadata
    # num_splits_dynamic_ptr, no seqused / virtual_batch_idx / semaphore /
    # output_scale, LSE iff requested, num_head = q heads, auto splits < 32.
    # num_splits=1 keys never combine.
    assert sorted(compile_keys, key=str) == sorted(
        (
            (
                torch.bfloat16,
                torch.float32,
                128,
                num_heads,
                5,
                True,
                False,
                return_lse,
                False,
                True,
                False,
                False,
            )
            for num_heads in (16, 32)
            for return_lse in (True, False)
        ),
        key=str,
    )
