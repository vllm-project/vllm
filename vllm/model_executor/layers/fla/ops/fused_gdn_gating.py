# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton

UNIFIED_BUFFER_SIZE = 1572864
NUM_CORES = 40


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    COL_ITER: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER: tl.constexpr,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    for row_idx in range(0, ROW_ITER):
        batch_off = i_b * ROW_ITER * BLK_BATCHES + row_idx * BLK_BATCHES + tl.arange(
            0, BLK_BATCHES)

        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)

            off = batch_off[:,
                            None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[
                                None, :]
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)

            blk_A_log = tl.load(A_log + head_off, mask=head_mask)
            blk_a = tl.load(a + off, mask=mask)
            blk_b = tl.load(b + off, mask=mask)
            blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(beta * x <= threshold,
                                  (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)

            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

            # compute beta_output = sigmoid(b)
            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
            tl.store(beta_output + off,
                     blk_beta_output.to(beta_output.dtype.element_ty),
                     mask=mask)


def fused_gdn_gating_patch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    BLK_HEADS = 8
    COL_ITER = triton.cdiv(num_heads, BLK_HEADS)

    if batch <= NUM_CORES:
        progs = batch
        BLK_BATCHES = 1
        ROW_ITER = 1
    else:
        progs = NUM_CORES
        FACTOR = 8 * num_heads
        row_per_core = triton.cdiv(batch, NUM_CORES)
        BLK_BATCHES = triton.next_power_of_2(
            triton.cdiv(UNIFIED_BUFFER_SIZE, FACTOR * BLK_HEADS) //
            a.element_size()) // 2
        ROW_ITER = triton.cdiv(row_per_core, BLK_BATCHES)

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1,
                              batch,
                              num_heads,
                              dtype=b.dtype,
                              device=b.device)

    grid = (progs, seq_len)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        batch,
        beta,
        threshold,
        BLK_HEADS=BLK_HEADS,
        COL_ITER=COL_ITER,
        BLK_BATCHES=BLK_BATCHES,
        ROW_ITER=ROW_ITER,
    )
    return g, beta_output
