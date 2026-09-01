# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Once-per-step block-index prep kernel for GDN ``mamba_cache_mode='all'``.

Fuses the per-request block-anchor arithmetic of
``compute_mamba_prefix_caching_block_indices`` (mamba_attn.py) and the
spec-decode read-anchor resolution (gdn_attn.py) into a single Triton launch
that writes runner-owned persistent buffers shared by every GDN metadata
builder — replacing the eager cdiv/sub/clamp tensor chains that otherwise run
once per builder (3x per step) between cudagraph replays. Gated by
``VLLM_GDN_INKERNEL_CKPT_IDX`` (default on; "0" restores the eager
per-builder math bit-for-bit).
"""

import os

import torch

from vllm.triton_utils import tl, triton


def gdn_inkernel_ckpt_idx_enabled() -> bool:
    return os.environ.get("VLLM_GDN_INKERNEL_CKPT_IDX", "1") != "0"


@triton.jit
def _gdn_block_idx_prep_kernel(
    seq_lens_ptr,  # (num_reqs,) int32
    num_computed_ptr,  # (num_reqs,) int32 (seq_lens - query_lens)
    prev_idx_ptr,  # (num_reqs,) int32, prev-step anchors or dummy
    out_last_computed_ptr,  # (num_reqs,) int32
    out_first_scheduled_ptr,  # (num_reqs,) int32
    out_last_scheduled_ptr,  # (num_reqs,) int32
    out_prev_step_ptr,  # (num_reqs,) int32 (spec only)
    out_packed_ptr,  # (num_reqs, 2) int32: [last_computed, last_scheduled]
    out_packed_spec_ptr,  # (num_reqs, 2) int32: [prev_step, last_scheduled]
    num_reqs,
    BLOCK_SIZE: tl.constexpr,  # mamba_block_size
    HAS_SPEC: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < num_reqs
    sl = tl.load(seq_lens_ptr + offs, mask=mask, other=0)
    nct = tl.load(num_computed_ptr + offs, mask=mask, other=0)
    # Verbatim semantics of compute_mamba_prefix_caching_block_indices
    # (mamba_attn.py:91-101): cdiv(n, B) - 1 with the same clamp placement
    # (first_scheduled deliberately unclamped). Domain restriction: the
    # formulas require nct >= 0 and sl >= 0 (for nct <= -B the unclamped fs
    # would diverge from torch's floor division under Triton's truncating
    # `//`). This always holds in the shipped wiring — nct = seq_lens -
    # query_lens with both >= 0 and padded rows zeroed by the runner — but
    # unit tests feeding raw tensors must respect it too.
    lc = tl.maximum((nct + BLOCK_SIZE - 1) // BLOCK_SIZE - 1, 0)
    fs = (nct + BLOCK_SIZE) // BLOCK_SIZE - 1
    ls = tl.maximum((sl + BLOCK_SIZE - 1) // BLOCK_SIZE - 1, 0)
    tl.store(out_last_computed_ptr + offs, lc, mask=mask)
    tl.store(out_first_scheduled_ptr + offs, fs, mask=mask)
    tl.store(out_last_scheduled_ptr + offs, ls, mask=mask)
    tl.store(out_packed_ptr + offs * 2, lc, mask=mask)
    tl.store(out_packed_ptr + offs * 2 + 1, ls, mask=mask)
    if HAS_SPEC:
        # Spec read-anchor resolution (gdn_attn.py:497-506): prev-step
        # anchor when tracked, else clamp((nct - 1) // B, 0). The lone
        # negative numerator (nct == 0) lands at 0 after the clamp under
        # both floor and trunc division.
        prev = tl.load(prev_idx_ptr + offs, mask=mask, other=-1)
        fb = tl.maximum((nct - 1) // BLOCK_SIZE, 0)
        ra = tl.where(prev >= 0, prev, fb)
        tl.store(out_prev_step_ptr + offs, ra, mask=mask)
        tl.store(out_packed_spec_ptr + offs * 2, ra, mask=mask)
        tl.store(out_packed_spec_ptr + offs * 2 + 1, ls, mask=mask)


class GDNBlockIdxPrepBuffers:
    """Runner-owned persistent buffers written once per step by the prep
    kernel and consumed by every GDN metadata builder (and, through the
    metadata, by the captured decode kernels — stable device addresses).
    ``packed_anchors[*]`` hold (read, write) pairs for the kernels' single
    64-bit anchor load: nospec read = last computed block, spec read = the
    resolved prev-step anchor; write = last scheduled block for both.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        kw = dict(dtype=torch.int32, device=device)
        self.block_idx_last_computed_token = torch.zeros(max_num_reqs, **kw)
        self.block_idx_first_scheduled_token = torch.zeros(max_num_reqs, **kw)
        self.block_idx_last_scheduled_token = torch.zeros(max_num_reqs, **kw)
        self.block_idx_last_scheduled_token_prev_step = torch.zeros(
            max_num_reqs, **kw
        )
        self.packed_anchors = torch.zeros((max_num_reqs, 2), **kw)
        self.packed_anchors_spec = torch.zeros((max_num_reqs, 2), **kw)

    def prepare(
        self,
        seq_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        prev_last_scheduled_idx: torch.Tensor | None,
        mamba_block_size: int,
        num_reqs: int,
    ) -> None:
        assert num_reqs <= self.max_num_reqs
        has_spec = prev_last_scheduled_idx is not None
        grid = (triton.cdiv(num_reqs, 256),)
        _gdn_block_idx_prep_kernel[grid](
            seq_lens,
            num_computed_tokens,
            prev_last_scheduled_idx if has_spec else seq_lens,
            self.block_idx_last_computed_token,
            self.block_idx_first_scheduled_token,
            self.block_idx_last_scheduled_token,
            self.block_idx_last_scheduled_token_prev_step,
            self.packed_anchors,
            self.packed_anchors_spec,
            num_reqs,
            BLOCK_SIZE=mamba_block_size,
            HAS_SPEC=has_spec,
            BLOCK_N=256,
        )
