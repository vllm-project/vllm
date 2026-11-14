# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-form replacement for aiter's serial MLA work planner at batch size 1.

aiter's ``get_mla_metadata_v1`` plans a uniform-qo dense decode with
``kn_get_mla_metadata_v1_2``: one 64-lane wavefront that walks every CU in a
serial loop. That walk costs ~70 us at long context, which is more than the
draft attention kernel it schedules. With one request, no qo tiling
(``kQoSplits`` off), ``page_size == 1``, ``qk_batch_ratio == 1`` and the auto
split count, the walk has a closed form: every split takes ``c`` KV blocks
(``c = ceil(sum_blocks / eff_splits)``), the last split absorbs the remainder
and a tail shorter than ``tail_done_threshold`` (== ``max_seqlen_qo``) is
folded into the last split. This kernel writes that plan with one program and
matches aiter's output entry for entry; ``AiterMlaBs1Planner`` verifies that
against aiter on the live device before it is ever used.

``VLLM_K31_BEAT_DRAFTMETA`` under ``decode_backend=latency_specialized``: the
builder side is ``levers.mla``.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton

logger = init_logger(__name__)

# aiter v1_2: fixed_over_head_num_blocks = max(1, cdiv(16, page_size)).
_FIXED_OVERHEAD_BLOCKS = 16
# aiter v1_2 MLA_V12_SPLIT_COEF; Triton lowers it to the same fp32 1.2f that
# aiter multiplies sqrtf() by. Must track aiter's value (verify catches drift).
_SPLIT_COEF = 1.2


@triton.jit
def _mla_bs1_meta_kernel(
    kv_indptr_ptr,
    work_meta_ptr,
    work_indptr_ptr,
    work_info_ptr,
    reduce_indptr_ptr,
    reduce_final_map_ptr,
    reduce_partial_map_ptr,
    work_indptr_addr,
    work_info_addr,
    NUM_CU: tl.constexpr,
    QLEN: tl.constexpr,
    KV_GRAN: tl.constexpr,
    FIXED_OH: tl.constexpr,
    TAIL_THR: tl.constexpr,
    SPLIT_COEF: tl.constexpr,
    REDUCE_INDPTR_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    kv_begin = tl.load(kv_indptr_ptr)
    kv_len = tl.load(kv_indptr_ptr + 1) - kv_begin
    num_blocks = (kv_len + KV_GRAN - 1) // KV_GRAN
    sum_blocks = num_blocks + FIXED_OH
    # eff = clamp(lrintf(1.2f * sqrtf(float(max(1, sum)) * float(num_splits))))
    work = tl.maximum(sum_blocks, 1).to(tl.float32) * NUM_CU
    eff = tldevice.rint(SPLIT_COEF * tl.sqrt_rn(work)).to(tl.int32)
    eff = tl.minimum(tl.maximum(eff, 1), NUM_CU)
    c = (sum_blocks + eff - 1) // eff
    # Last split index if no tail folds earlier.
    k0 = tl.maximum((num_blocks + c - 1) // c - 1, 0)

    lanes = tl.arange(0, BLOCK)
    split_end = (lanes + 1) * c * KV_GRAN
    # A split whose end leaves less than TAIL_THR tokens becomes the last one.
    folds = (lanes < k0) & (kv_len - split_end < TAIL_THR)
    k = tl.minimum(tl.min(tl.where(folds, lanes, k0), axis=0), k0)
    num_works = k + 1

    # work_indptr[j] = min(j, num_works), j in [0, NUM_CU].
    wi_mask = lanes < NUM_CU + 1
    tl.store(work_indptr_ptr + lanes, tl.minimum(lanes, num_works), mask=wi_mask)

    # MlaWorkInfo rows: batch, partial_qo_loc, qo_start, qo_end, kv_start,
    # kv_end, kv_offset, pad.
    row_mask = lanes <= k
    is_last = lanes == k
    kv_start = lanes * c * KV_GRAN
    kv_end = tl.where(is_last, kv_len, split_end)
    partial = tl.where(is_last & (k == 0), -1, lanes * QLEN)
    zeros = tl.zeros([BLOCK], dtype=tl.int32)
    base = work_info_ptr + lanes * 8
    tl.store(base + 0, zeros, mask=row_mask)
    tl.store(base + 1, partial, mask=row_mask)
    tl.store(base + 2, zeros, mask=row_mask)
    tl.store(base + 3, zeros + QLEN, mask=row_mask)
    tl.store(base + 4, kv_start, mask=row_mask)
    tl.store(base + 5, kv_end, mask=row_mask)
    tl.store(base + 6, kv_len - kv_end, mask=row_mask)
    tl.store(base + 7, zeros, mask=row_mask)

    # Reduce plan: one qo tile, split num_works ways when k > 0. aiter fills
    # reduce_indptr to its full size (max_num_seqs * qo tiles + 1, which can
    # exceed BLOCK), so the tail past the first BLOCK lanes is written too.
    split = k > 0
    ri_tail = tl.where(split, num_works, 0)
    ri = tl.where(lanes == 0, 0, ri_tail)
    tl.store(reduce_indptr_ptr + lanes, ri, mask=lanes < REDUCE_INDPTR_SIZE)
    for off in range(BLOCK, REDUCE_INDPTR_SIZE, BLOCK):
        tl.store(
            reduce_indptr_ptr + off + lanes,
            zeros + ri_tail,
            mask=off + lanes < REDUCE_INDPTR_SIZE,
        )
    tl.store(
        reduce_final_map_ptr + lanes,
        tl.where(lanes == 0, 0, QLEN),
        mask=(lanes < 2) & split,
    )
    tl.store(reduce_partial_map_ptr + lanes, lanes * QLEN, mask=row_mask & split)

    tl.store(work_meta_ptr, work_indptr_addr)
    tl.store(work_meta_ptr + 1, work_info_addr)


def default_verify_kv_lens() -> list[int]:
    """KV lengths covering the unsaturated split law (short contexts), the
    tail-fold boundaries and long contexts."""
    return sorted(
        set(
            list(range(0, 320))
            + list(range(320, 6000, 23))
            + list(range(6000, 400000, 7919))
            + [2**i + d for i in range(11, 21) for d in (-17, -1, 0, 1, 7)]
            + [360000, 360007, 1000000]
        )
    )


class AiterMlaBs1Planner:
    """Writes aiter's persistent MLA schedule for one uniform-qo request.

    Bound to one builder's persistent buffers. ``verify`` compares against
    aiter's planner on the current device and must pass before ``plan`` is
    used; any mismatch (e.g. a different aiter planner variant or a folded
    head plan) leaves the planner disabled.
    """

    def __init__(
        self,
        work_meta_data: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info_set: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
    ):
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        self.num_cu = work_indptr.numel() - 1
        self.verified: set[tuple] = set()
        self.failed: set[tuple] = set()

    def supported_buffers(self) -> bool:
        return (
            self.work_info_set.dim() == 2
            and self.work_info_set.shape[1] == 8
            and self.work_info_set.shape[0] >= self.num_cu
            and self.reduce_partial_map.numel() >= self.num_cu
            and self.reduce_final_map.numel() >= 2
            and self.reduce_indptr.numel() >= 2
            and self.work_meta_data.numel() >= 2
            and self.work_meta_data.element_size() == 8
        )

    def plan(self, kv_indptr: torch.Tensor, qlen: int) -> None:
        block = triton.next_power_of_2(self.num_cu + 1)
        _mla_bs1_meta_kernel[(1,)](
            kv_indptr,
            self.work_meta_data.view(torch.int64),
            self.work_indptr,
            self.work_info_set,
            self.reduce_indptr,
            self.reduce_final_map,
            self.reduce_partial_map,
            self.work_indptr.data_ptr(),
            self.work_info_set.data_ptr(),
            NUM_CU=self.num_cu,
            QLEN=qlen,
            KV_GRAN=16,
            FIXED_OH=_FIXED_OVERHEAD_BLOCKS,
            TAIL_THR=qlen,
            SPLIT_COEF=_SPLIT_COEF,
            REDUCE_INDPTR_SIZE=self.reduce_indptr.numel(),
            BLOCK=block,
            num_warps=4,
        )

    def verify(
        self,
        qlen: int,
        num_heads: int,
        is_causal: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        kv_lens: list[int] | None = None,
    ) -> bool:
        """Compare against aiter for a sweep of KV lengths. Uses scratch
        copies, so the live buffers are untouched. Results are cached per
        configuration; call outside CUDA graph capture.

        The default sweep samples the split law (dense at short contexts, the
        tail-fold boundaries, strided to 1M); it is a check, not a proof. A
        missed divergence would still plan a contiguous KV partition of at
        most num_cu splits, only not aiter's. It takes ~0.5 s, so a draft
        that runs eagerly pays it on its first single-request step."""
        key = (qlen, num_heads, is_causal, q_dtype, kv_dtype)
        if key in self.verified:
            return True
        if key in self.failed or qlen < 2 or not self.supported_buffers():
            return False
        from aiter import get_mla_metadata_v1

        # ``mla.install`` wraps aiter's planner; check against aiter's own.
        get_mla_metadata_v1 = getattr(
            get_mla_metadata_v1, "aiter_planner", get_mla_metadata_v1
        )
        dev = self.work_indptr.device
        if kv_lens is None:
            kv_lens = default_verify_kv_lens()
        logger.info(
            "Checking the bs1 MLA planner against aiter (qlen=%d, heads=%d, "
            "causal=%s) over %d KV lengths; runs once per shape.",
            qlen,
            num_heads,
            is_causal,
            len(kv_lens),
        )

        def scratch(t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(t, -7) if t.dtype != torch.uint64 else t.clone()

        ref = [
            scratch(t)
            for t in (
                self.work_meta_data,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            )
        ]
        mine = AiterMlaBs1Planner(*[scratch(t) for t in ref])
        qo_indptr = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
        last_page = torch.ones(1, dtype=torch.int32, device=dev)
        kv_indptr = torch.zeros(2, dtype=torch.int32, device=dev)
        ok = True
        for kv_len in kv_lens:
            kv_indptr[1] = kv_len
            for t in ref[1:] + [
                mine.work_indptr,
                mine.work_info_set,
                mine.reduce_indptr,
                mine.reduce_final_map,
                mine.reduce_partial_map,
            ]:
                t.fill_(-7)
            wmd, wind, wis, rind, rfm, rpm = ref
            get_mla_metadata_v1(
                qo_indptr,
                kv_indptr,
                last_page,
                num_heads,
                1,
                is_causal,
                wmd,
                wis,
                wind,
                rind,
                rfm,
                rpm,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=qlen,
                uni_seqlen_qo=qlen,
                fast_mode=True,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
            )
            mine.plan(kv_indptr, qlen)
            pairs = (
                (wind, mine.work_indptr),
                (wis, mine.work_info_set),
                (rind, mine.reduce_indptr),
                (rfm, mine.reduce_final_map),
                (rpm, mine.reduce_partial_map),
            )
            if not all(torch.equal(a, b) for a, b in pairs):
                logger.warning(
                    "bs1 MLA planner mismatches aiter at kv_len=%d qlen=%d; "
                    "keeping aiter's planner.",
                    kv_len,
                    qlen,
                )
                ok = False
                break
            # Pointers differ by design (scratch buffers): check they point to
            # the planner's own buffers.
            ptrs = mine.work_meta_data.view(torch.int64).tolist()
            if ptrs[:2] != [
                mine.work_indptr.data_ptr(),
                mine.work_info_set.data_ptr(),
            ]:
                ok = False
                break
        if ok:
            self.verified.add(key)
            logger.info_once(
                "bs1 MLA planner matches aiter (qlen=%d, heads=%d, causal=%s) "
                "over %d KV lengths; using it for single-request decode.",
                qlen,
                num_heads,
                is_causal,
                len(kv_lens),
            )
        else:
            self.failed.add(key)
        return ok

    def is_verified(
        self,
        qlen: int,
        num_heads: int,
        is_causal: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> bool:
        return (qlen, num_heads, is_causal, q_dtype, kv_dtype) in self.verified
