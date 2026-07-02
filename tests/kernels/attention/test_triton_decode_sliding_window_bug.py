# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-only reproducer for the TRITON split-KV decode sliding-window bug.

The split-KV decode kernels disagree on how the KV range is tiled when a
sliding window is active:

* stage 1 (``_fwd_kernel_stage1`` / ``_fwd_grouped_kernel_stage1``) tiles only
  the *windowed* range ``[seq_len - W, seq_len)`` -- so trailing splits can be
  empty and are left unwritten in ``attn_logits``.
* stage 2 (``_fwd_kernel_stage2``) recomputes split boundaries from the *full*
  ``seq_len`` (it is never told the window) and merges every split whose
  full-seq range is non-empty -- including the ones stage 1 left unwritten.

``attn_logits`` is allocated with ``torch.empty`` (uninitialized), so stage 2
folds garbage LSE/values into the softmax merge. In production this is
*intermittent* (depends on what is in memory) and only appears once the dynamic
``num_kv_splits`` grows past the window (long context). Here we make it
deterministic by poisoning ``attn_logits`` with a large finite value before the
call, then assert the property the kernel must satisfy: **the result must not
depend on num_kv_splits.**

Run: pytest -q tests/kernels/attention/test_triton_decode_sliding_window_bug.py
Fails on buggy code (the >window split count diverges), passes once stage 2 is
made window-aware.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

DEVICE = current_platform.device_type

# Small, model-independent MHA setup (kv_group_num == 1 -> normal kernel path).
B, H, D = 1, 1, 16
SEQ_LEN = 64
WINDOW = 8
POISON = 1.0e30


def _run(num_kv_splits: int) -> torch.Tensor:
    torch.manual_seed(0)
    q = torch.randn(B, H, D, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(SEQ_LEN, H, D, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(SEQ_LEN, H, D, dtype=torch.bfloat16, device=DEVICE)
    # Identity paged mapping, page_size = 1.
    req_to_token = torch.arange(SEQ_LEN, device=DEVICE).view(1, SEQ_LEN)
    b_seq_len = torch.full((B,), SEQ_LEN, device=DEVICE)

    o = torch.zeros(B, H, D, dtype=torch.bfloat16, device=DEVICE)
    lse = torch.zeros(B, H, dtype=torch.bfloat16, device=DEVICE)
    # Poison the scratch buffer to model worst-case uninitialized memory.
    attn_logits = torch.full(
        (B, H, num_kv_splits, D + 1), POISON, dtype=torch.float32, device=DEVICE
    )

    decode_attention_fwd(
        q,
        k,
        v,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        1.0 / (D**0.5),
        page_size=1,
        sliding_window=WINDOW,
    )
    return o


def _torch_windowed_ref() -> torch.Tensor:
    torch.manual_seed(0)
    q = torch.randn(B, H, D, dtype=torch.bfloat16, device=DEVICE).float()
    k = torch.randn(SEQ_LEN, H, D, dtype=torch.bfloat16, device=DEVICE).float()
    v = torch.randn(SEQ_LEN, H, D, dtype=torch.bfloat16, device=DEVICE).float()
    sw_start = max(SEQ_LEN - WINDOW, 0)
    kw = k[sw_start:SEQ_LEN, 0]  # [W, D]
    vw = v[sw_start:SEQ_LEN, 0]
    scores = (q[0, 0] @ kw.T) * (1.0 / (D**0.5))  # [W]
    p = torch.softmax(scores, dim=-1)
    return (p @ vw).view(B, H, D)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton decode kernel needs GPU"
)
def test_sliding_window_decode_independent_of_num_kv_splits():
    ref = _torch_windowed_ref()
    out_1 = _run(num_kv_splits=1)  # 1 split: no empty split -> correct
    out_4 = _run(num_kv_splits=4)  # 4 <= W, divides W: no empty split -> correct
    out_16 = _run(num_kv_splits=16)  # 16 > W: trailing empty splits -> bug

    # Baselines agree with the windowed reference.
    torch.testing.assert_close(out_1.float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_4.float(), ref, atol=2e-2, rtol=2e-2)
    # The property under test: the result must not depend on the split count.
    # On buggy code out_16 is dominated by the poisoned (unwritten) splits.
    torch.testing.assert_close(out_16.float(), out_1.float(), atol=2e-2, rtol=2e-2)
