# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE top-k finalize + shared-expert add + TP reduce-scatter + residual add +
Gemma RMSNorm (+ MXFP8 quantization) for token-sharded batches.

Under MiniMax-M3's large-batch sequence parallelism, each rank owns a
contiguous shard of ``S`` tokens of the residual stream, while the
tensor-parallel experts see the whole batch. Unfused, a MoE layer ends with the
experts' top-k finalize, the shared-expert add, a reduce-scatter of the
TP-partial rows and the next norm. Here two kernels and a barrier do it:

1. ``_finalize_scatter_kernel`` sums each token's routed expert rows (as
   ``do_finalize=False`` left them) and its shared-expert row, and stores the
   BF16 partial row straight into its owner rank's (``t // S``) symmetric
   mailbox ``[generations, tp, S_max, hidden]``, in this rank's slot.
2. A symmetric-memory barrier.
3. ``_reduce_norm_kernel`` sums the ``tp`` slots of each owned row in rank
   order, adds the residual (stored back), and writes the Gemma RMSNorm as
   MXFP8 (E4M3 + 128x4-swizzled UE8M0 scales) or BF16.

The partial rows are rounded to BF16 where the unfused path rounds them: the
routed sum, the shared add and the reduction. Consecutive layers alternate
between two mailbox generations, so a rank may write layer L+1's rows while a
peer still reads layer L's; every layer in between has a TP collective, which
orders L+2's writes after L's reads.

The mailbox holds 2 x max_num_batched_tokens BF16 rows of symmetric memory per
GPU whatever the TP size: 201 MB for hidden 6144 and 8192 tokens.
"""

import torch
import torch.distributed._symmetric_memory as symm_mem

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.minimax_m3.nvidia.ops.norm_mxfp8 import (
    MXFP8_BLOCK,
    SF_ROW_TILE,
    _store_mxfp8_row,
    _swizzled_scale_offsets,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

GENERATIONS = 2
_FINALIZE_BLOCK = 1024


@triton.jit(do_not_specialize=["S", "rank", "generation"])
def _finalize_scatter_kernel(
    routed_ptr,
    weights_ptr,
    permuted_idx_ptr,
    shared_ptr,
    mailbox_ptrs,
    S,
    S_MAX: tl.constexpr,
    rank,
    generation,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    routed = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(TOP_K):
        row = tl.load(permuted_idx_ptr + token * TOP_K + k)
        local = row >= 0  # -1: the expert is not on this rank
        weight = tl.load(weights_ptr + token * TOP_K + k).to(tl.float32)
        x = tl.load(
            routed_ptr + tl.where(local, row, 0).to(tl.int64) * H + cols,
            mask=mask & local,
            other=0.0,
        )
        routed += tl.where(local, weight, 0.0) * x.to(tl.float32)
    shared = tl.load(shared_ptr + token.to(tl.int64) * H + cols, mask=mask, other=0.0)
    partial = (routed.to(tl.bfloat16).to(tl.float32) + shared.to(tl.float32)).to(
        tl.bfloat16
    )
    owner = token // S
    owner_mailbox = tl.load(mailbox_ptrs + owner).to(tl.pointer_type(tl.bfloat16))
    slot = ((generation * TP + rank) * S_MAX + token - owner * S).to(tl.int64)
    tl.store(owner_mailbox + slot * H + cols, partial, mask=mask)


@triton.jit(do_not_specialize=["S", "generation"])
def _reduce_norm_kernel(
    mailbox_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    sf_ptr,
    S,
    S_MAX: tl.constexpr,
    generation,
    eps,
    H: tl.constexpr,
    TP: tl.constexpr,
):
    G: tl.constexpr = H // 32
    GP: tl.constexpr = triton.next_power_of_2(H) // 32
    PADDED_COLS: tl.constexpr = (G + 3) // 4 * 4
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, GP)
    if sf_ptr is not None:
        sf_off = _swizzled_scale_offsets(row, g, PADDED_COLS)
        if row >= S:
            # Zero the padding rows of the swizzled scale tensor.
            tl.store(sf_ptr + sf_off, tl.zeros((GP,), tl.uint8), mask=g < G)
            return
    cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < H
    total = tl.zeros((GP, 32), tl.float32)
    for r in tl.static_range(TP):
        slot = (generation * TP + r) * S_MAX + row
        total += tl.load(mailbox_ptr + slot * H + cols, mask=mask, other=0.0).to(
            tl.float32
        )
    residual = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0)
    h = total.to(tl.bfloat16).to(tl.float32) + residual.to(tl.float32)
    tl.store(residual_ptr + row * H + cols, h, mask=mask)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(h * h) / H
    y = (h * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16)
    if sf_ptr is not None:
        _store_mxfp8_row(y, out_ptr, sf_ptr, sf_off, row, cols, mask, g, G)
    else:
        tl.store(out_ptr + row * H + cols, y, mask=mask)


class FinalizeReduceScatterGemmaRMSNorm:
    """The MoE layer tail of a token-sharded batch, see the module docstring.

    Construction is collective over the TP group (symmetric-memory rendezvous).
    """

    def __init__(
        self, *, hidden_size: int, max_num_tokens: int, device: torch.device
    ) -> None:
        group = get_tp_group()
        self.tp = group.world_size
        self.rank = group.rank_in_group
        self.hidden_size = hidden_size
        self.max_shard = (max_num_tokens + self.tp - 1) // self.tp
        self.mailbox = symm_mem.empty(
            (GENERATIONS, self.tp, self.max_shard, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self.handle = symm_mem.rendezvous(self.mailbox, group.device_group)
        self.mailbox_ptrs = torch.tensor(
            [self.handle.buffer_ptrs[r] for r in range(self.tp)],
            dtype=torch.int64,
            device=device,
        )

    def __call__(
        self,
        moe_output: MoEOutput,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        *,
        mxfp8: bool,
        generation: int,
    ) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
        """Return (normalized shard rows, new residual shard).

        Args:
            moe_output: The whole batch's unfinalized TP-partial routed output
                and TP-partial shared-expert output, with routed_scaling_factor
                in the routing weights.
            residual: [S, hidden] BF16 residual shard of this rank, updated in
                place. The batch's tokens [rank * S, (rank + 1) * S) are its.
            norm_weight: [hidden] Gemma RMSNorm weight (scale is 1 + w).
            norm_eps: RMSNorm epsilon.
            mxfp8: MXFP8 output (128x4-swizzled scales) or BF16.
            generation: Mailbox generation; consecutive layers alternate.

        """
        routed = moe_output.routed
        shared = moe_output.shared_output
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert shared is not None and moe_output.routed_scaling_factor == 1.0
        num_tokens, hidden = shared.shape
        shard = residual.shape[0]
        assert hidden == self.hidden_size and residual.shape[1] == hidden
        assert shard <= self.max_shard and shard * (self.tp - 1) < num_tokens
        assert num_tokens <= shard * self.tp and residual.is_contiguous()
        top_k = routed.expert_weights.shape[-1]
        if shard * self.tp > num_tokens:
            # The last rank's padding rows get no contribution from any rank.
            self.mailbox[generation, :, num_tokens - shard * (self.tp - 1) :].zero_()
        _finalize_scatter_kernel[(num_tokens, triton.cdiv(hidden, _FINALIZE_BLOCK))](
            routed.gemm2_permuted,
            routed.expert_weights.contiguous(),
            routed.expanded_idx_to_permuted_idx.contiguous(),
            shared.contiguous(),
            self.mailbox_ptrs,
            shard,
            self.max_shard,
            self.rank,
            generation,
            H=hidden,
            TOP_K=top_k,
            TP=self.tp,
            BLOCK=_FINALIZE_BLOCK,
            num_warps=4,
        )
        # Every rank's rows of this generation have landed.
        self.handle.barrier(channel=generation)
        if mxfp8:
            rows = round_up(shard, SF_ROW_TILE)
            out = torch.empty(
                (shard, hidden), dtype=torch.float8_e4m3fn, device=shared.device
            )
            scale = torch.empty(
                rows * round_up(hidden // MXFP8_BLOCK, 4),
                dtype=torch.uint8,
                device=shared.device,
            )
        else:
            rows = shard
            out = torch.empty((shard, hidden), dtype=shared.dtype, device=shared.device)
            scale = None
        _reduce_norm_kernel[(rows,)](
            self.mailbox,
            residual,
            norm_weight,
            out.view(torch.uint8) if mxfp8 else out,
            scale,
            shard,
            self.max_shard,
            generation,
            norm_eps,
            H=hidden,
            TP=self.tp,
            num_warps=4,
        )
        if not mxfp8:
            return out, residual
        return (
            QuantizedActivation(
                data=out,
                scale=scale,
                orig_dtype=shared.dtype,
                orig_shape=out.shape,
                quant_key=kMxfp8Dynamic,
            ),
            residual,
        )


_sharded_moe_finalize: FinalizeReduceScatterGemmaRMSNorm | None = None


def init_sharded_moe_finalize(
    *, hidden_size: int, max_num_tokens: int, device: torch.device
) -> None:
    """Build the MoE tail for token-sharded batches. Collective over TP."""
    global _sharded_moe_finalize
    _sharded_moe_finalize = FinalizeReduceScatterGemmaRMSNorm(
        hidden_size=hidden_size, max_num_tokens=max_num_tokens, device=device
    )


def finalize_reduce_scatter_gemma_rms_norm(
    moe_output: MoEOutput,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    mxfp8: bool,
    generation: int,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Finalize a deferred whole-batch MoE output into this rank's normalized
    shard rows (MXFP8 or BF16) and new residual shard."""
    assert _sharded_moe_finalize is not None, "init_sharded_moe_finalize not called"
    return _sharded_moe_finalize(
        moe_output,
        residual,
        norm_weight,
        norm_eps,
        mxfp8=mxfp8,
        generation=generation,
    )
