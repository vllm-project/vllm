# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the DiffusionGemma denoise sampler.

The step needs, for every canvas position, the argmax of the
temperature-scaled logits, a Gumbel-max sample from them, the entropy of
their softmax, and the softmax itself in the model dtype for the
self-conditioning matmul. As PyTorch ops these are several passes over a
`[rows, vocab]` fp32 tensor plus a same-sized noise tensor. The Triton
kernel reads each row once with an online max and sum, draws the noise
inline, and writes the probabilities once.

`sample_step` then updates each request's canvas and convergence state
from those statistics, and `softcap_logits` applies the final logit
softcap in one pass. Each has a PyTorch reference for tests and non-CUDA
runs. The dummy runs never take a decode step, so the model registers these
kernels for JIT warmup with every variant serving reaches.

The logprobs of a converging slot are written to a per-slot stash by
`stash_converged_logprobs`, which skips the rows of slots that did not
converge.
"""

import math
from typing import Any, NamedTuple

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, tldevice, triton

# Allowed-set sizes covering Triton's integer specializations (1, divisible by
# 16, other) for constrained steps, which run the sampler over K columns.
_ALLOWED_SET_SIZES = (1, 2, 16)
# Triton's ROCm backend specializes a pointer on whether its storage fits
# 32-bit offsets.
_MAX_32BIT_BYTES = 2**31 - 1


def stability_rows(generation_config: dict[str, Any]) -> int:
    """Rows of the accepted-canvas history.

    In Transformers, `stability_threshold=1` means the current step must
    match the previous step. The history includes the current step, so it
    holds one more row.
    """
    return generation_config["stability_threshold"] + 1


class _WarmupDims(NamedTuple):
    canvas_length: int
    history_rows: int
    vocab: int
    dtype: torch.dtype
    # Constrained steps run at tensor parallel 1 only.
    constrained: bool
    max_tile_rows: int


def _warmup_dims(vllm_config: Any) -> _WarmupDims:
    model_config = vllm_config.model_config
    diffusion_config = vllm_config.diffusion_config
    canvas_length = diffusion_config.canvas_length if diffusion_config else 32
    return _WarmupDims(
        canvas_length=canvas_length,
        history_rows=stability_rows(model_config.try_get_generation_config()),
        vocab=model_config.get_vocab_size(),
        dtype=model_config.dtype,
        constrained=vllm_config.parallel_config.tensor_parallel_size == 1,
        max_tile_rows=vllm_config.scheduler_config.max_num_seqs * canvas_length,
    )


def _tile_rows(max_tile_rows: int, cols: int, *itemsizes: int) -> list[int]:
    """Row counts for a `[rows, cols]` tile reaching each storage-size class
    of tensors with these element sizes."""
    rows = {2}
    for itemsize in itemsizes:
        large = _MAX_32BIT_BYTES // (cols * itemsize) + 1
        if large <= max_tile_rows:
            rows.add(large)
    return sorted(rows)


def _warmup_tensor(dtype: torch.dtype, *shape: int) -> TritonWarmupTensor:
    return TritonWarmupTensor(
        dtype, shape=shape, ptr_range_bytes=math.prod(shape) * dtype.itemsize
    )


def _empty(like: Any, *shape: int, dtype: torch.dtype) -> Any:
    """`torch.empty` on `like`'s device, or only its metadata when
    `like` is a warmup input, so warmup allocates nothing."""
    if isinstance(like, TritonWarmupTensor):
        return _warmup_tensor(dtype, *shape)
    return torch.empty(shape, dtype=dtype, device=like.device)


@triton.jit(do_not_specialize=["seed", "CL"])
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
        # Until a block holds a finite logit the running max stays -inf, and
        # top_k/top_p leave whole blocks masked. x - new_m would then be
        # -inf - -inf = NaN and poison z for the rest of the row, so subtract
        # 0 instead: every d is still -inf and every e 0.
        safe_m = tl.where(new_m > float("-inf"), new_m, 0.0)
        # Rescale the running sums to the new max. While m is -inf, z and s
        # are 0, so alpha and the shift are zeroed rather than computed from
        # -inf - -inf.
        alpha = tl.where(m > float("-inf"), tl.exp((m - safe_m) * inv), 0.0)
        shift = tl.where(z > 0, (m - safe_m) * inv * z, 0.0)
        d = (x - safe_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        # A -inf logit (top_k/top_p, or the row's padding) has d = -inf and
        # e = 0, and -inf * 0 is NaN. The where drops it from the sum.
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
    # tl.store casts to the pointee dtype. mypy type-checks the kernel body
    # as Python and rejects .to() on the int accumulators.
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


def _row_stats_warmup_inputs(vllm_config: Any) -> list[dict[str, Any]]:
    dims = _warmup_dims(vllm_config)
    cases = []
    for cols in (dims.vocab, *(_ALLOWED_SET_SIZES if dims.constrained else ())):
        for probs_dtype in (None, dims.dtype):
            itemsizes = [4] if probs_dtype is None else [4, probs_dtype.itemsize]
            for rows in _tile_rows(dims.max_tile_rows, cols, *itemsizes):
                cases.append(
                    dict(
                        logits=_warmup_tensor(torch.float32, rows, cols),
                        temps=_warmup_tensor(torch.float32, 2),
                        canvas_len=dims.canvas_length,
                        seed=0,
                        probs_dtype=probs_dtype,
                    )
                )
    return cases


@triton_kernel_dispatcher_with_warmup(
    kernel=_row_stats_kernel, warmup_inputs=_row_stats_warmup_inputs
)
def sample_row_stats(
    logits: torch.Tensor,
    temps: torch.Tensor,
    canvas_len: int,
    seed: int,
    probs_dtype: torch.dtype | None,
) -> DispatchSpec:
    """Argmax, Gumbel-max sample, entropy and (optionally) softmax per row.

    `logits` is `[rows, vocab]` and row `i` uses temperature
    `temps[i // canvas_len]`. A zero temperature is greedy: the sample is
    the argmax, and the entropy is the reference's, which clamps the
    temperature at 1e-10.
    """
    rows, vocab = logits.shape
    argmax = _empty(logits, rows, dtype=torch.int64)
    sample = _empty(logits, rows, dtype=torch.int64)
    entropy = _empty(logits, rows, dtype=torch.float32)
    if probs_dtype is not None:
        probs = _empty(logits, rows, vocab, dtype=probs_dtype)
        probs_arg, probs_stride = probs, probs.stride(0)
    else:
        probs = None
        probs_arg, probs_stride = entropy, 0
    outputs = argmax, sample, entropy, probs
    return (
        (rows,),
        dict(
            logits_ptr=logits,
            logits_stride=logits.stride(0),
            temps_ptr=temps,
            argmax_ptr=argmax,
            sample_ptr=sample,
            entropy_ptr=entropy,
            probs_ptr=probs_arg,
            probs_stride=probs_stride,
            seed=seed,
            V=vocab,
            CL=canvas_len,
            WRITE_PROBS=probs is not None,
            BLOCK=4096,
            num_warps=8,
        ),
        outputs,
    )


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


@triton.jit(do_not_specialize=["W"])
def _sample_step_kernel(
    new_tokens_ptr,
    argmax_tokens_ptr,
    token_entropy_ptr,
    random_tokens_ptr,
    decode_slots_ptr,
    decode_idx_ptr,
    valid_canvas_len_ptr,
    canvas_ptr,
    canvas_stride,
    argmax_canvas_ptr,
    argmax_canvas_stride,
    history_ptr,
    history_stride,
    history_stride_1,
    pin_mask_ptr,
    pin_mask_stride,
    seed_canvas_ptr,
    seed_canvas_stride,
    step_ptr,
    is_encoder_phase_ptr,
    confident_ptr,
    history_len_ptr,
    max_steps_ptr,
    read_only_ptr,
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    keep_ptr,
    W,
    confidence_threshold,
    entropy_bound,
    ST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(decode_slots_ptr + i)
    offs = tl.arange(0, BLOCK)
    in_row = offs < W
    row = i.to(tl.int64) * W

    entropy = tl.load(token_entropy_ptr + row + offs, mask=in_row, other=0.0)
    argmax = tl.load(argmax_tokens_ptr + row + offs, mask=in_row, other=0)
    confident = tl.sum(entropy, 0) / W < confidence_threshold
    tl.store(confident_ptr + slot, confident)

    is_commit = tl.load(is_encoder_phase_ptr + slot)
    is_denoise = is_commit == 0
    new_step = tl.where(is_denoise, tl.load(step_ptr + slot) + 1, 0)
    tl.store(step_ptr + slot, new_step)

    # The denoise step writes its argmax into the history ring, then the
    # canvas is stable when every row of the ring matches the first.
    hist_len = tl.load(history_len_ptr + slot)
    write_pos = hist_len % ST
    hist = history_ptr + slot * history_stride + offs
    write = in_row & is_denoise & (write_pos == 0)
    ref = tl.load(hist, mask=in_row, other=0)
    ref = tl.where(write, argmax, ref)
    tl.store(hist, ref, mask=write)
    mismatch = 0
    for h in tl.static_range(1, ST):
        write = in_row & is_denoise & (write_pos == h)
        cur = tl.load(hist + h * history_stride_1, mask=in_row, other=0)
        cur = tl.where(write, argmax, cur)
        tl.store(hist + h * history_stride_1, cur, mask=write)
        mismatch += tl.sum((in_row & (cur != ref)).to(tl.int32), 0)
    new_hist_len = tl.where(is_denoise, hist_len + 1, 0)
    tl.store(history_len_ptr + slot, new_hist_len)

    argmax_canvas = argmax_canvas_ptr + slot * argmax_canvas_stride + offs
    emitted = tl.load(argmax_canvas, mask=in_row, other=0)
    emitted = tl.where(is_denoise, argmax, emitted)
    tl.store(argmax_canvas, emitted, mask=in_row & is_denoise)

    converged = ((mismatch == 0) & confident & (new_hist_len >= ST)) | (
        new_step >= tl.load(max_steps_ptr + slot)
    )
    read_only = tl.load(read_only_ptr + slot) != 0
    # Commit done: denoise next. Denoise converged: commit next, unless a
    # read-only slot emits now and skips the commit forward.
    next_commit = is_denoise & converged
    tl.store(is_encoder_phase_ptr + slot, next_commit & ~read_only)
    # Only a slot that denoises again reads its soft embed.
    tl.store(keep_ptr + i, is_denoise & ~next_commit)

    emit = (is_commit != 0) | (next_commit & read_only)
    out = tl.load(decode_idx_ptr + i)
    tl.store(
        sampled_ptr + out * sampled_stride + offs,
        tl.where(emit, emitted, 0),
        mask=in_row,
    )
    valid_len = tl.load(valid_canvas_len_ptr + i)
    tl.store(num_sampled_ptr + out, tl.where(emit, valid_len, 0))

    # Entropy-bound acceptance: sorted by entropy, a position is kept while
    # the entropy of the positions before it stays within the bound. Each
    # key packs an order-preserving int32 form of the entropy over the
    # position, so ties break by position and the sort carries the position.
    # NaN sorts after every number, as in torch.sort, and padding sorts last.
    bits = entropy.to(tl.int32, bitcast=True)
    bits = tl.where(entropy != entropy, 0x7FC00000, bits)
    key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    key = tl.where(in_row, key, 0x7FFFFFFF)
    packed = tl.sort((key.to(tl.int64) << 32) | offs.to(tl.int64), 0)
    pos = (packed & 0xFFFFFFFF).to(tl.int32)
    key = (packed >> 32).to(tl.int32)
    sorted_entropy = tl.where(key >= 0, key, key ^ 0x7FFFFFFF).to(
        tl.float32, bitcast=True
    )
    sorted_entropy = tl.where(in_row, sorted_entropy, 0.0)
    accept = tl.cumsum(sorted_entropy, 0) - sorted_entropy <= entropy_bound

    # The canvas is written in sorted order. A commit reinitializes it with
    # noise; a converged denoise holds the argmax; otherwise accepted
    # positions take the sample, the rest are renoised, and pinned positions
    # keep their seed.
    sample = tl.load(new_tokens_ptr + row + pos, mask=in_row, other=0)
    noise = tl.load(random_tokens_ptr + row + pos, mask=in_row, other=0)
    sorted_argmax = tl.load(argmax_tokens_ptr + row + pos, mask=in_row, other=0)
    pinned = tl.load(pin_mask_ptr + slot * pin_mask_stride + pos, mask=in_row, other=0)
    seed = tl.load(
        seed_canvas_ptr + slot * seed_canvas_stride + pos, mask=in_row, other=0
    )
    token = tl.where(accept, sample, noise)
    token = tl.where(pinned != 0, seed, token)
    token = tl.where(next_commit, sorted_argmax, token)
    token = tl.where(is_commit != 0, noise, token)
    tl.store(canvas_ptr + slot * canvas_stride + pos, token, mask=in_row)


def _sample_step_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    dims = _warmup_dims(vllm_config)
    canvas_length, rows = dims.canvas_length, dims.history_rows

    tile = (2, canvas_length)
    slots = (2,)
    return dict(
        new_tokens=_warmup_tensor(torch.int64, *tile),
        argmax_tokens=_warmup_tensor(torch.int64, *tile),
        token_entropy=_warmup_tensor(torch.float32, *tile),
        random_tokens=_warmup_tensor(torch.int64, *tile),
        decode_slots=_warmup_tensor(torch.int64, *slots),
        decode_idx=_warmup_tensor(torch.int64, *slots),
        valid_canvas_len=_warmup_tensor(torch.int64, *slots),
        canvas=_warmup_tensor(torch.int64, *tile),
        argmax_canvas=_warmup_tensor(torch.int64, *tile),
        history=_warmup_tensor(torch.int64, 2, rows, canvas_length),
        pin_mask=_warmup_tensor(torch.bool, *tile),
        seed_canvas=_warmup_tensor(torch.int64, *tile),
        step=_warmup_tensor(torch.int32, *slots),
        is_encoder_phase=_warmup_tensor(torch.bool, *slots),
        confident=_warmup_tensor(torch.bool, *slots),
        history_len=_warmup_tensor(torch.int32, *slots),
        max_steps=_warmup_tensor(torch.int32, *slots),
        read_only=_warmup_tensor(torch.bool, *slots),
        sampled=_warmup_tensor(torch.int32, *tile),
        num_sampled=_warmup_tensor(torch.int32, *slots),
        confidence_threshold=0.0,
        entropy_bound=0.0,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_sample_step_kernel, warmup_inputs=_sample_step_warmup_inputs
)
def sample_step(
    *,
    new_tokens: torch.Tensor,
    argmax_tokens: torch.Tensor,
    token_entropy: torch.Tensor,
    random_tokens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_idx: torch.Tensor,
    valid_canvas_len: torch.Tensor,
    canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    history: torch.Tensor,
    pin_mask: torch.Tensor,
    seed_canvas: torch.Tensor,
    step: torch.Tensor,
    is_encoder_phase: torch.Tensor,
    confident: torch.Tensor,
    history_len: torch.Tensor,
    max_steps: torch.Tensor,
    read_only: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    confidence_threshold: float,
    entropy_bound: float,
) -> DispatchSpec:
    """Advance each decode request one step: confidence, entropy-bound
    accept/renoise, the history ring and convergence.

    The contiguous `[n, W]` statistics come from `sample_row_stats` and
    `random_tokens` is the renoise draw. Per-slot state is indexed by
    `decode_slots` and updated in place over its first `W` columns;
    `sampled` and `num_sampled` are written at `decode_idx`. Returns
    which requests denoise again and so read their soft embed next step.
    """
    n, W = new_tokens.shape
    for stats in (new_tokens, argmax_tokens, token_entropy, random_tokens):
        assert stats.shape == (n, W) and stats.stride() == (W, 1)
    for state in (canvas, argmax_canvas, history, pin_mask, seed_canvas, sampled):
        assert state.stride(-1) == 1
    keep = _empty(new_tokens, n, dtype=torch.bool)
    # Sized by the full canvas so narrow tiles share the compiled kernel.
    BLOCK = triton.next_power_of_2(canvas.shape[1])
    # The remaining tensors, strides and scalars bind to the kernel's
    # arguments by name.
    return (
        (n,),
        dict(
            keep_ptr=keep,
            W=W,
            ST=history.shape[1],
            BLOCK=BLOCK,
            num_warps=min(8, max(1, BLOCK // 256)),
        ),
        keep,
    )


def sample_step_reference(
    *,
    new_tokens: torch.Tensor,
    argmax_tokens: torch.Tensor,
    token_entropy: torch.Tensor,
    random_tokens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_idx: torch.Tensor,
    valid_canvas_len: torch.Tensor,
    canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    history: torch.Tensor,
    pin_mask: torch.Tensor,
    seed_canvas: torch.Tensor,
    step: torch.Tensor,
    is_encoder_phase: torch.Tensor,
    confident: torch.Tensor,
    history_len: torch.Tensor,
    max_steps: torch.Tensor,
    read_only: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    confidence_threshold: float,
    entropy_bound: float,
) -> torch.Tensor:
    """The PyTorch form of the same step, for tests and non-CUDA runs."""
    num_decode, W = new_tokens.shape
    ST = history.shape[1]
    canvas = canvas[:, :W]
    argmax_canvas = argmax_canvas[:, :W]
    history = history[:, :, :W]
    pin_mask = pin_mask[:, :W]
    seed_canvas = seed_canvas[:, :W]
    sampled = sampled[:, :W]

    # ---- Confidence ----
    # A canvas truncated near max_model_len is zero-padded up to W by the
    # caller; those padded rows are uniform (max entropy, argmax 0), so they
    # never trigger early convergence and are stable, and only the real
    # ``valid_canvas_len`` tokens are committed (num_sampled below).
    mean_entropy = token_entropy.mean(dim=-1)
    confident[decode_slots] = mean_entropy < confidence_threshold

    # ---- Entropy-bound acceptance mask ----
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1, stable=True)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Post-sample ----
    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit

    # Step update: +1 for denoise, reset to 0 for commit
    cur_step = step[decode_slots]
    step[decode_slots] = torch.where(
        is_denoise, cur_step + 1, cur_step.new_zeros(num_decode)
    )

    # Renoise draws from the whole vocabulary even for a constrained read.
    # The model was trained on random-vocabulary noise. Noise drawn from the
    # allowed set looks like garbled text, and later steps denoise it into
    # more garbled text.
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)
    denoise_canvas = torch.where(
        pin_mask[decode_slots], seed_canvas[decode_slots], denoise_canvas
    )
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    # History: write argmax_tokens for denoise requests at circular position
    hist_len = history_len[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    # Argmax canvas: update for denoise, preserve for commit
    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len[decode_slots] = new_hist_len

    # ---- Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=new_tokens.device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    converged = (stable & confident[decode_slots] & (new_hist_len >= ST)) | (
        step[decode_slots] >= max_steps[decode_slots]
    )
    # Commit done → denoise next (False); denoise converged → commit next (True)
    next_commit = is_denoise & converged
    keep = is_denoise & ~next_commit

    emit = is_commit | (next_commit & read_only[decode_slots])
    sampled[decode_idx] = argmax_canvas[decode_slots].to(sampled.dtype) * emit[:, None]
    num_sampled[decode_idx] = (emit * valid_canvas_len).to(num_sampled.dtype)

    # Overwrite canvas with argmax for newly converged denoise requests
    canvas[decode_slots] = torch.where(
        next_commit.unsqueeze(1), argmax_canvas[decode_slots], canvas[decode_slots]
    )
    is_encoder_phase[decode_slots] = next_commit & ~read_only[decode_slots]
    return keep


@triton.jit(do_not_specialize=["logits_stride", "out_stride", "V"])
def _softcap_kernel(
    logits_ptr,
    logits_stride,
    out_ptr,
    out_stride,
    V,
    cap,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * out_stride + offs, tldevice.tanh(x / cap) * cap, mask=mask)


def _softcap_warmup_inputs(vllm_config: Any) -> list[dict[str, Any]]:
    dims = _warmup_dims(vllm_config)
    # The full-vocabulary logits come in the model dtype, a constrained
    # step's K columns (and an fp32 head's logits) in fp32. The output is fp32.
    return [
        dict(logits=_warmup_tensor(logits_dtype, rows, dims.vocab), cap=1.0)
        for logits_dtype in {dims.dtype, torch.float32}
        for rows in _tile_rows(dims.max_tile_rows, dims.vocab, logits_dtype.itemsize, 4)
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_softcap_kernel, warmup_inputs=_softcap_warmup_inputs
)
def softcap_logits(logits: torch.Tensor, cap: float) -> DispatchSpec:
    """`tanh(logits / cap) * cap` in fp32, as a new `[rows, vocab]` tensor.

    fp32 before tanh for numerical stability (matches HF DiffusionGemma).
    """
    rows, vocab = logits.shape
    out = _empty(logits, rows, vocab, dtype=torch.float32)
    BLOCK = 4096
    return (
        (rows, triton.cdiv(vocab, BLOCK)),
        dict(
            logits_ptr=logits,
            logits_stride=logits.stride(0),
            out_ptr=out,
            out_stride=out.stride(0),
            V=vocab,
            cap=cap,
            BLOCK=BLOCK,
        ),
        out,
    )


# The stash kernels reproduce compute_topk_scores' values (tied top-k ids
# aside) only while each block is summed as _topk_log_softmax_kernel sums it:
# 1024 wide, 4 warps, in block order.
_LP_BLOCK = 1024
_LP_BLOCKS_PER_CHUNK = 16
_NO_KEY = tl.constexpr(-(2**63))


@triton.jit
def _lp_skip(keep_ptr, valid_ptr, j, pos):
    return (tl.load(keep_ptr + j) == 0) | (pos >= tl.load(valid_ptr + j))


@triton.jit
def _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
    skip = _lp_skip(keep_ptr, valid_ptr, j, pos)
    if HAS_IDS:
        # Requests with their own token ids report those instead.
        skip = skip | (tl.load(num_ids_ptr + tl.load(slots_ptr + j)) > 0)
    return skip


@triton.jit
def _lp_load(logits_ptr, logits_stride, src_ptr, temps_ptr, j, pos, W, offs, V, other):
    row = (tl.load(src_ptr + j) * W + pos).to(tl.int64)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < V, other=other)
    # Round-to-nearest fp32 division, as torch divides.
    return tl.div_rn(x.to(tl.float32), tl.load(temps_ptr + j))


@triton.jit
def _lp_key32(x):
    # fp32 -> int32 with the same order.
    bits = x.to(tl.int32, bitcast=True)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@triton.jit
def _lp_max_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    chunk_arg_ptr,
    block_max_ptr,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    STORE_BLOCK_MAX: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    max_val = float("-inf")
    best_idx = 0
    for b in range(BLOCKS_PER_CHUNK):
        block_id = chunk * BLOCKS_PER_CHUNK + b
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr,
            logits_stride,
            src_ptr,
            temps_ptr,
            j,
            pos,
            W,
            offs,
            V,
            float("-inf"),
        )
        block_max, block_idx = tl.max(x, 0, return_indices=True)
        if STORE_BLOCK_MAX:
            tl.store(
                block_max_ptr + r * NUM_BLOCKS + block_id,
                block_max,
                mask=block_id < NUM_BLOCKS,
            )
        take = block_max > max_val
        best_idx = tl.where(take, block_id * BLOCK + block_idx, best_idx)
        max_val = tl.where(take, block_max, max_val)
    tl.store(chunk_max_ptr + r * NUM_CHUNKS + chunk, max_val)
    tl.store(chunk_arg_ptr + r * NUM_CHUNKS + chunk, best_idx)


@triton.jit
def _lp_sum_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    block_sum_ptr,
    block_cnt_ptr,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CHUNKS_POW2)
    maxes = tl.load(
        chunk_max_ptr + r * NUM_CHUNKS + c, mask=c < NUM_CHUNKS, other=float("-inf")
    )
    max_val = tl.max(maxes, 0)
    for b in range(BLOCKS_PER_CHUNK):
        block_id = chunk * BLOCKS_PER_CHUNK + b
        block = block_id * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr, logits_stride, src_ptr, temps_ptr, j, pos, W, block, V, 0.0
        )
        e = tl.exp(x - max_val)
        e = tl.where(block < V, e, 0.0)
        out = r * NUM_BLOCKS + block_id
        in_row = block_id < NUM_BLOCKS
        tl.store(block_sum_ptr + out, tl.sum(e), mask=in_row)
        rank = tl.sum(((x >= max_val) & (block < V)).to(tl.int32))
        tl.store(block_cnt_ptr + out, rank, mask=in_row)


@triton.jit
def _lp_topk_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    slots_ptr,
    num_ids_ptr,
    block_max_ptr,
    cand_ptr,
    K,
    W,
    V,
    NUM_CHUNKS,
    NUM_BLOCKS,
    HAS_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    KP: tl.constexpr,
):
    """The KP largest keys of one chunk of a row, unordered.

    A key orders by value, then by lower index: the fp32 bits mapped to an
    order-preserving int32 in the high word, the inverted index in the low.
    Keys are distinct, so the top-k under them is unique.
    """
    j = tl.program_id(0)
    pos = tl.program_id(1)
    chunk = tl.program_id(2)
    if _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
        return
    r = (j * W + pos).to(tl.int64)
    # K block maxima are K distinct elements at or above the K-th largest
    # block max, so every top-k element is too: the rest of the row is
    # skipped without a reduction over it.
    b = tl.arange(0, BLOCKS_POW2)
    block_max = tl.load(
        block_max_ptr + r * NUM_BLOCKS + b, mask=b < NUM_BLOCKS, other=float("-inf")
    )
    sorted_max = tl.sort(_lp_key32(block_max), descending=True)
    kth = tl.max(tl.where(b == K - 1, sorted_max, -0x7FFFFFFF - 1), 0)
    threshold = kth.to(tl.int64) << 32
    # Distinct placeholders below every real key.
    best = _NO_KEY + 1 + tl.arange(0, KP).to(tl.int64)
    best_min = tl.min(best, 0)
    for blk in range(BLOCKS_PER_CHUNK):
        offs = (chunk * BLOCKS_PER_CHUNK + blk) * BLOCK + tl.arange(0, BLOCK)
        x = _lp_load(
            logits_ptr,
            logits_stride,
            src_ptr,
            temps_ptr,
            j,
            pos,
            W,
            offs,
            V,
            float("-inf"),
        )
        key = (_lp_key32(x).to(tl.int64) << 32) | (0x7FFFFFFF - offs).to(tl.int64)
        key = tl.where((offs < V) & (key >= threshold), key, _NO_KEY)
        key_max = tl.max(key, 0)
        while key_max > best_min:
            best = tl.where(best == best_min, key_max, best)
            best_min = tl.min(best, 0)
            key = tl.where(key == key_max, _NO_KEY, key)
            key_max = tl.max(key, 0)
    out = (r * NUM_CHUNKS + chunk) * KP + tl.arange(0, KP)
    tl.store(cand_ptr + out, best)


@triton.jit
def _lp_topk_merge_kernel(
    keep_ptr,
    valid_ptr,
    slots_ptr,
    num_ids_ptr,
    cand_ptr,
    topk_ptr,
    K,
    W,
    NUM_CAND,
    HAS_IDS: tl.constexpr,
    CAND_POW2: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    if _lp_skip_topk(keep_ptr, valid_ptr, slots_ptr, num_ids_ptr, j, pos, HAS_IDS):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CAND_POW2)
    keys = tl.load(cand_ptr + r * NUM_CAND + c, mask=c < NUM_CAND, other=_NO_KEY)
    keys = tl.sort(keys, descending=True)
    ids = 0x7FFFFFFF - (keys & 0x7FFFFFFF)
    tl.store(topk_ptr + r * K + c, ids.to(tl.int32), mask=c < K)


@triton.jit
def _lp_write_kernel(
    logits_ptr,
    logits_stride,
    src_ptr,
    slots_ptr,
    keep_ptr,
    valid_ptr,
    temps_ptr,
    chunk_max_ptr,
    chunk_arg_ptr,
    block_sum_ptr,
    block_cnt_ptr,
    topk_ptr,
    num_ids_ptr,
    ids_ptr,
    ids_stride,
    out_ids_ptr,
    out_lp_ptr,
    out_rank_ptr,
    num_topk,
    W,
    CL,
    V,
    STASH_W,
    NUM_CHUNKS,
    NUM_BLOCKS,
    HAS_IDS: tl.constexpr,
    LOGITS_MODE: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    SUM_GROUP: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    j = tl.program_id(0)
    pos = tl.program_id(1)
    if _lp_skip(keep_ptr, valid_ptr, j, pos):
        return
    r = (j * W + pos).to(tl.int64)
    c = tl.arange(0, CHUNKS_POW2)
    in_chunks = c < NUM_CHUNKS
    maxes = tl.load(
        chunk_max_ptr + r * NUM_CHUNKS + c, mask=in_chunks, other=float("-inf")
    )
    max_val = tl.max(maxes, 0)
    # The first chunk holding the max holds the first-index argmax.
    first = tl.min(tl.where(maxes == max_val, c, CHUNKS_POW2), 0)
    best_idx = tl.load(chunk_arg_ptr + r * NUM_CHUNKS + first).to(tl.int64)
    blocks = tl.arange(0, BLOCKS_POW2)
    rank = tl.sum(
        tl.load(block_cnt_ptr + r * NUM_BLOCKS + blocks, mask=blocks < NUM_BLOCKS)
    )
    # Add the block sums one by one in block order. A group is loaded at once
    # and each sum picked out by a masked sum, which adds only zeros to it, so
    # the loop does not wait on a load per block.
    lanes = tl.arange(0, SUM_GROUP)
    se = 0.0
    for g in range(0, NUM_BLOCKS, SUM_GROUP):
        sums = tl.load(
            block_sum_ptr + r * NUM_BLOCKS + g + lanes,
            mask=g + lanes < NUM_BLOCKS,
            other=0.0,
        )
        for i in tl.static_range(SUM_GROUP):
            se += tl.sum(tl.where(lanes == i, sums, 0.0))
    lse = tl.log(se)

    row = (tl.load(src_ptr + j) * W + pos).to(tl.int64)
    base = logits_ptr + row * logits_stride
    temp = tl.load(temps_ptr + j)
    slot = tl.load(slots_ptr + j).to(tl.int64)
    out = (slot * CL + pos) * STASH_W
    tl.store(out_rank_ptr + slot * CL + pos, rank)
    num_ids = 0
    if HAS_IDS:
        num_ids = tl.load(num_ids_ptr + slot)
    for c0 in range(0, STASH_W, COL_BLOCK):
        col = c0 + tl.arange(0, COL_BLOCK)
        k = col - 1
        # Column 0 is the argmax; then the request's own token ids, or else
        # the top-k. The rest pads with token 0 at -inf.
        in_topk = (k >= 0) & (k < num_topk) & (num_ids == 0)
        tok = tl.load(topk_ptr + r * num_topk + k, mask=in_topk, other=0)
        tok = tok.to(tl.int64)
        valid = (col == 0) | in_topk
        if HAS_IDS:
            in_ids = (k >= 0) & (k < num_ids)
            from_ids = tl.load(ids_ptr + slot * ids_stride + k, mask=in_ids, other=0)
            tok = tl.where(in_ids, from_ids.to(tl.int64), tok)
            valid = valid | in_ids
        tok = tl.where(col == 0, best_idx, tok)
        x = tl.load(base + tok, mask=valid, other=0.0).to(tl.float32)
        x = tl.div_rn(x, temp)
        if not LOGITS_MODE:
            x = x - max_val - lse
        in_stash = col < STASH_W
        tl.store(out_ids_ptr + out + col, tl.where(valid, tok, 0), mask=in_stash)
        tl.store(
            out_lp_ptr + out + col, tl.where(valid, x, float("-inf")), mask=in_stash
        )


def stash_converged_logprobs(
    logits: torch.Tensor,
    canvas_len: int,
    src: torch.Tensor,
    slots: torch.Tensor,
    keep: torch.Tensor,
    valid: torch.Tensor,
    temps: torch.Tensor,
    num_topk: int,
    num_token_ids: torch.Tensor,
    token_ids: torch.Tensor | None,
    out_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_ranks: torch.Tensor,
    logits_mode: bool,
) -> None:
    """Write the logprobs of the converged requests' rows to the stash.

    Request `j` owns logits rows `src[j] * canvas_len + p` for
    `p < valid[j]`, divided by `temps[j]`. Where `keep[j]`, they go to
    `out_*[slots[j], p]`, laid out as compute_topk_scores lays out a row:
    the argmax, then the request's `token_ids` if it has any, else the
    `num_topk` largest (tied ids by lower id), padded with token 0 at -inf.
    Rows not kept exit at once; the others are split into chunks so a few
    kept rows still fill the GPU.
    """
    m = slots.shape[0]
    vocab = logits.shape[-1]
    assert logits.stride(-1) == 1
    assert temps.dtype == torch.float32
    assert num_topk + 1 <= out_logprobs.shape[-1]
    block = _LP_BLOCK
    num_blocks = triton.cdiv(vocab, block)
    blocks_per_chunk = min(_LP_BLOCKS_PER_CHUNK, num_blocks)
    num_chunks = triton.cdiv(num_blocks, blocks_per_chunk)
    rows = m * canvas_len
    device = logits.device
    chunk_max = torch.empty(rows, num_chunks, dtype=torch.float32, device=device)
    chunk_arg = torch.empty(rows, num_chunks, dtype=torch.int64, device=device)
    block_sum = torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
    block_cnt = torch.empty(rows, num_blocks, dtype=torch.int32, device=device)
    block_max = (
        torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
        if num_topk > 0
        else chunk_max
    )
    has_ids = token_ids is not None
    ids_arg = token_ids if token_ids is not None else num_token_ids
    common = (logits, logits.stride(0), src, keep, valid, temps)
    grid = (m, canvas_len, num_chunks)
    _lp_max_kernel[grid](
        *common,
        chunk_max,
        chunk_arg,
        block_max,
        canvas_len,
        vocab,
        num_chunks,
        num_blocks,
        BLOCK=block,
        BLOCKS_PER_CHUNK=blocks_per_chunk,
        STORE_BLOCK_MAX=num_topk > 0,
        num_warps=4,
    )
    chunks_pow2 = triton.next_power_of_2(num_chunks)
    _lp_sum_kernel[grid](
        *common,
        chunk_max,
        block_sum,
        block_cnt,
        canvas_len,
        vocab,
        num_chunks,
        num_blocks,
        BLOCK=block,
        BLOCKS_PER_CHUNK=blocks_per_chunk,
        CHUNKS_POW2=chunks_pow2,
        num_warps=4,
    )
    topk = chunk_arg
    if num_topk > 0:
        kp = triton.next_power_of_2(num_topk)
        cand = torch.empty(rows, num_chunks, kp, dtype=torch.int64, device=device)
        _lp_topk_kernel[grid](
            *common,
            slots,
            num_token_ids,
            block_max,
            cand,
            num_topk,
            canvas_len,
            vocab,
            num_chunks,
            num_blocks,
            HAS_IDS=has_ids,
            BLOCK=block,
            BLOCKS_PER_CHUNK=blocks_per_chunk,
            BLOCKS_POW2=triton.next_power_of_2(max(num_blocks, num_topk)),
            KP=kp,
            num_warps=4,
        )
        topk = torch.empty(rows, num_topk, dtype=torch.int32, device=device)
        _lp_topk_merge_kernel[(m, canvas_len)](
            keep,
            valid,
            slots,
            num_token_ids,
            cand,
            topk,
            num_topk,
            canvas_len,
            num_chunks * kp,
            HAS_IDS=has_ids,
            CAND_POW2=triton.next_power_of_2(num_chunks * kp),
        )
    stash_width = out_logprobs.shape[-1]
    _lp_write_kernel[(m, canvas_len)](
        logits,
        logits.stride(0),
        src,
        slots,
        keep,
        valid,
        temps,
        chunk_max,
        chunk_arg,
        block_sum,
        block_cnt,
        topk,
        num_token_ids,
        ids_arg,
        ids_arg.stride(0) if has_ids else 0,
        out_ids,
        out_logprobs,
        out_ranks,
        num_topk,
        canvas_len,
        out_logprobs.shape[1],
        vocab,
        stash_width,
        num_chunks,
        num_blocks,
        HAS_IDS=has_ids,
        LOGITS_MODE=logits_mode,
        CHUNKS_POW2=chunks_pow2,
        BLOCKS_POW2=triton.next_power_of_2(num_blocks),
        SUM_GROUP=16,
        COL_BLOCK=min(triton.next_power_of_2(stash_width), 1024),
    )
