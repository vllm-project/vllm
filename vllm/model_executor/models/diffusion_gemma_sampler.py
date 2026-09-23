# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass row statistics for the DiffusionGemma denoise sampler.

The step needs, for every canvas position, the argmax of the
temperature-scaled logits, a Gumbel-max sample from them, the entropy of
their softmax, and the softmax itself in the model dtype for the
self-conditioning matmul. As PyTorch ops these are several passes over a
``[rows, vocab]`` fp32 tensor plus a same-sized noise tensor. The Triton
kernel reads each row once with an online max and sum, draws the noise
inline, and writes the probabilities once.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    argmax_ptr,
    sample_ptr,
    entropy_ptr,
    probs_ptr,
    probs_stride,
    seed,
    V,
    CL,
    WRITE_PROBS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row // CL).to(tl.float32)
    inv = 1.0 / tl.maximum(temp, 1e-10)
    noise = (temp > 0).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    rng_base = row.to(tl.int64) * V

    # The running max stays in raw logit units and the temperature is applied
    # to differences from it. Scaling the row first breaks at a zero
    # temperature: the row is multiplied by 1e10, fp32 spacing there is about
    # a thousand, and exp(x - max) becomes 0 or inf for the max element itself
    # whenever the two sides round differently.
    m = float("-inf")  # running max of the raw row
    z = 0.0  # sum exp(d), d = (x - m) * inv
    s = 0.0  # sum d exp(d)
    best = float("-inf")
    best_idx = 0
    best_noisy = float("-inf")
    best_noisy_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, 0)
        new_m = tl.maximum(m, block_max)
        # Rescale the running sums to the new max. Before the first block
        # z is 0, so the shift term is dropped rather than -inf * 0.
        alpha = tl.exp((m - new_m) * inv)
        shift = tl.where(z > 0, (m - new_m) * inv * z, 0.0)
        d = (x - new_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        # A -inf logit (top_k/top_p, or the row's padding) has d = -inf and
        # e = 0, and -inf * 0 is NaN; it contributes nothing to the sum.
        s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
        m = new_m

        block_idx = tl.argmax(x, 0)
        take = block_max > best
        best_idx = tl.where(take, start + block_idx, best_idx)
        best = tl.where(take, block_max, best)

        u = tl.rand(seed, rng_base + offs)
        u = tl.maximum(u, 1e-20)
        gumbel = -tl.log(-tl.log(u))
        noisy = tl.where(mask, x * inv + gumbel * noise, float("-inf"))
        noisy_max = tl.max(noisy, 0)
        noisy_idx = tl.argmax(noisy, 0)
        take_n = noisy_max > best_noisy
        best_noisy_idx = tl.where(take_n, start + noisy_idx, best_noisy_idx)
        best_noisy = tl.where(take_n, noisy_max, best_noisy)

    # With d = (x - m) * inv: p = e^d / z, log Z = log z in d units, and
    # H = -sum p (d - log z) = log z - s / z.
    # tl.store casts to the pointee dtype; the explicit .to() reads as an int
    # method to mypy, which type-checks the kernel body as Python.
    tl.store(argmax_ptr + row, best_idx)
    tl.store(sample_ptr + row, best_noisy_idx)
    tl.store(entropy_ptr + row, tl.log(z) - s / z)

    if WRITE_PROBS:
        out = probs_ptr + row.to(tl.int64) * probs_stride
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            p = tl.exp((x - m) * inv) / z
            tl.store(out + offs, p.to(probs_ptr.dtype.element_ty), mask=mask)


def sample_row_stats(
    logits: torch.Tensor,
    temps: torch.Tensor,
    canvas_len: int,
    seed: int,
    probs_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Argmax, Gumbel-max sample, entropy and (optionally) softmax per row.

    ``logits`` is ``[rows, vocab]``; row ``i`` uses temperature
    ``temps[i // canvas_len]``. A zero temperature means greedy: the sample is
    the argmax and the entropy is that of the unscaled row's limit, matching
    the PyTorch reference, which clamps the temperature at 1e-10.
    """
    rows, vocab = logits.shape
    device = logits.device
    argmax = torch.empty(rows, dtype=torch.int64, device=device)
    sample = torch.empty(rows, dtype=torch.int64, device=device)
    entropy = torch.empty(rows, dtype=torch.float32, device=device)
    if probs_dtype is not None:
        probs = torch.empty(rows, vocab, dtype=probs_dtype, device=device)
        probs_arg, probs_stride = probs, probs.stride(0)
    else:
        probs = None
        probs_arg, probs_stride = entropy, 0
    if rows == 0:
        return argmax, sample, entropy, probs
    _row_stats_kernel[(rows,)](
        logits,
        logits.stride(0),
        temps,
        argmax,
        sample,
        entropy,
        probs_arg,
        probs_stride,
        seed,
        vocab,
        canvas_len,
        WRITE_PROBS=probs is not None,
        BLOCK=4096,
        num_warps=8,
    )
    return argmax, sample, entropy, probs


def sample_row_stats_reference(
    logits: torch.Tensor,
    temps: torch.Tensor,
    canvas_len: int,
    probs_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The PyTorch form of the same statistics, for tests and non-CUDA runs."""
    temp = temps.repeat_interleave(canvas_len).float()[:, None]
    scaled = logits.float() / temp.clamp(min=1e-10)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    noisy = scaled + (-torch.log(-torch.log(u))) * (temp > 0).float()
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    # Masked (-inf) columns: 0 * -inf is NaN, so they are dropped from the sum.
    entropy = -torch.where(probs > 0, probs * log_probs, 0.0).sum(dim=-1)
    out_probs = probs.to(probs_dtype) if probs_dtype is not None else None
    return scaled.argmax(dim=-1), noisy.argmax(dim=-1), entropy, out_probs
