# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for HY V4 iHC (independent Hyper-Connections).

Open-source counterpart of the HPC ``fuse_ihc_pre`` / ``fuse_ihc_post`` /
``fuse_ihc_head`` kernels (which need a closed ``.so`` and sm100/sm103). The
eager path in ``vllm/models/hy_v4/nvidia/hc.py`` issues ~20 / 5 / 15 kernels
for pre / post / head; these ops issue 2 / 1 / 2.

pre / head are split into a *stats* launch (row sum-of-squares and the
``hc_fn`` dot products, split over the hidden dim so decode-sized batches
still fill the GPU) and an *apply* launch (finalize the sigmoid gates and
reduce the channels). Decode-sized batches (``T <= CFG["small_t"]``) use a
single launch instead, where the last program of each row finalizes it.
Arithmetic is fp32 throughout, matching the eager code.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _ihc_stats_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    T,
    stride_xt,
    stride_xc,
    stride_ws_t,
    stride_ws_s,
    D_PER_SPLIT,
    D: tl.constexpr,
    HC: tl.constexpr,
    N_OUT: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Partial sum(x^2) and partial x_flat @ w.T for a block of rows and a
    slice of the hidden dim (same slice in every hc channel).

    The dot runs on tensor cores with the fp32 weight split into a bf16
    hi + lo pair (x itself is bf16, so this keeps ~fp32 accuracy while the
    kernel stays purely memory bound).
    """
    pid_t = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    outs = tl.arange(0, N_PAD)
    out_mask = outs < N_OUT

    d_start = split * D_PER_SPLIT
    d_end = tl.minimum(d_start + D_PER_SPLIT, D)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    sq = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_T, N_PAD], dtype=tl.float32)
    for d0 in range(d_start, d_end, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_end
        for c in tl.static_range(HC):
            x = tl.load(
                x_ptr + rows[:, None] * stride_xt + c * stride_xc + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            xf = x.to(tl.float32)
            sq += xf * xf
            w = tl.load(
                w_ptr + outs[:, None] * (HC * D) + c * D + offs_d[None, :],
                mask=out_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            w_hi = w.to(x.dtype)
            w_lo = (w - w_hi.to(tl.float32)).to(x.dtype)
            acc += tl.dot(x, tl.trans(w_hi)) + tl.dot(x, tl.trans(w_lo))

    sumsq = tl.sum(sq, axis=1)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    ws_base = ws_ptr + rows * stride_ws_t + split * stride_ws_s
    tl.store(
        ws_base[:, None] + outs[None, :],
        acc,
        mask=row_mask[:, None] & out_mask[None, :],
    )
    tl.store(ws_base + N_OUT, sumsq, mask=row_mask)


@triton.jit
def _ihc_apply_kernel(
    x_ptr,
    ws_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    T,
    stride_xt,
    stride_xc,
    stride_ws_t,
    stride_ws_s,
    stride_yt,
    stride_pt,
    D_PER_SPLIT,
    SPLIT,
    norm_eps,
    hc_eps,
    magnitude,
    D: tl.constexpr,
    HC: tl.constexpr,
    N_OUT: tl.constexpr,
    HAS_POST: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Finalize gates from the workspace partials and reduce the channels for
    this program's slice of the hidden dim."""
    pid_t = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    outs = tl.arange(0, N_OUT)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Reduce the split partials with one vectorized load per row block.
    s_offs = tl.arange(0, SPLIT_PAD)
    s_mask = s_offs < SPLIT
    ws_rows = ws_ptr + rows[:, None] * stride_ws_t + s_offs[None, :] * stride_ws_s
    rs_mask = row_mask[:, None] & s_mask[None, :]
    acc = tl.sum(
        tl.load(
            ws_rows[:, :, None] + outs[None, None, :],
            mask=rs_mask[:, :, None],
            other=0.0,
        ),
        axis=1,
    )
    sumsq = tl.sum(tl.load(ws_rows + N_OUT, mask=rs_mask, other=0.0), axis=1)

    rstd = tl.rsqrt(sumsq / (HC * D) + norm_eps)
    mixes = acc * rstd[:, None]
    base = tl.load(base_ptr + outs)
    if HAS_POST:
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        is_pre = outs < HC
        scale = tl.where(is_pre, s0, s1)
        mag = tl.where(is_pre, 1.0, magnitude)
    else:
        scale = tl.load(scale_ptr) + tl.zeros([N_OUT], dtype=tl.float32)
        mag = 1.0 + tl.zeros([N_OUT], dtype=tl.float32)
    gates = mag[None, :] * tl.sigmoid(mixes * scale[None, :] + base[None, :]) + hc_eps
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if HAS_POST:  # noqa: SIM102 - constexpr branch, keep it separate from the runtime one
        if split == 0:
            post_mask = row_mask[:, None] & (outs[None, :] >= HC)
            tl.store(
                post_ptr + rows[:, None] * stride_pt + (outs[None, :] - HC),
                gates,
                mask=post_mask,
            )

    d_start = split * D_PER_SPLIT
    d_end = tl.minimum(d_start + D_PER_SPLIT, D)
    for d0 in range(d_start, d_end, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_end
        y = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        for c in tl.static_range(HC):
            g = tl.sum(tl.where(outs[None, :] == c, gates, 0.0), axis=1)
            x = tl.load(
                x_ptr + rows[:, None] * stride_xt + c * stride_xc + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            y += g[:, None] * x
        tl.store(
            y_ptr + rows[:, None] * stride_yt + offs_d[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & d_mask[None, :],
        )


@triton.jit
def _ihc_small_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    cnt_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    stride_xt,
    stride_xc,
    stride_ws_t,
    stride_ws_s,
    stride_yt,
    stride_pt,
    D_PER_SPLIT,
    SPLIT,
    norm_eps,
    hc_eps,
    magnitude,
    D: tl.constexpr,
    HC: tl.constexpr,
    N_OUT: tl.constexpr,
    HAS_POST: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Single-launch pre / head for decode-sized batches.

    One program per (row, hidden slice) computes the partial stats with plain
    fp32 FMAs (no tensor-core padding to 16 rows). The last program to finish a
    row (arrival counter, release/acquire) finalizes the gates and reduces the
    channels for the whole row; the serial tail is one L2-resident row read.
    """
    t = tl.program_id(0)
    split = tl.program_id(1)
    outs = tl.arange(0, N_OUT)
    d_start = split * D_PER_SPLIT
    d_end = tl.minimum(d_start + D_PER_SPLIT, D)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    sq = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([N_OUT, BLOCK_D], dtype=tl.float32)
    for d0 in range(d_start, d_end, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_end
        for c in tl.static_range(HC):
            xf = tl.load(
                x_ptr + t * stride_xt + c * stride_xc + offs_d, mask=d_mask, other=0.0
            ).to(tl.float32)
            sq += xf * xf
            w = tl.load(
                w_ptr + outs[:, None] * (HC * D) + c * D + offs_d[None, :],
                mask=d_mask[None, :],
                other=0.0,
            )
            acc += w * xf[None, :]
    ws_base = ws_ptr + t * stride_ws_t + split * stride_ws_s
    tl.store(ws_base + outs, tl.sum(acc, axis=1))
    tl.store(ws_base + N_OUT, tl.sum(sq, axis=0))

    # Publish this program's partials (all threads' stores, hence the barrier)
    # and find out whether we are the last program of the row.
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + t, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    if arrived == SPLIT - 1:
        # Reset for the next launch; every program of this row has arrived.
        tl.store(cnt_ptr + t, 0)
        s_offs = tl.arange(0, SPLIT_PAD)
        s_mask = s_offs < SPLIT
        ws_rows = ws_ptr + t * stride_ws_t + s_offs * stride_ws_s
        acc_all = tl.sum(
            tl.load(
                ws_rows[:, None] + outs[None, :],
                mask=s_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            ),
            axis=0,
        )
        sumsq = tl.sum(
            tl.load(ws_rows + N_OUT, mask=s_mask, other=0.0, cache_modifier=".cg"),
            axis=0,
        )
        rstd = tl.rsqrt(sumsq / (HC * D) + norm_eps)
        mixes = acc_all * rstd
        base = tl.load(base_ptr + outs)
        if HAS_POST:
            s0 = tl.load(scale_ptr)
            s1 = tl.load(scale_ptr + 1)
            is_pre = outs < HC
            scale = tl.where(is_pre, s0, s1)
            mag = tl.where(is_pre, 1.0, magnitude)
        else:
            scale = tl.load(scale_ptr) + tl.zeros([N_OUT], dtype=tl.float32)
            mag = 1.0 + tl.zeros([N_OUT], dtype=tl.float32)
        gates = mag * tl.sigmoid(mixes * scale + base) + hc_eps
        if HAS_POST:
            tl.store(post_ptr + t * stride_pt + (outs - HC), gates, mask=outs >= HC)
        for d0 in range(0, D, APPLY_BLOCK_D):
            offs = d0 + tl.arange(0, APPLY_BLOCK_D)
            mask = offs < D
            y = tl.zeros([APPLY_BLOCK_D], dtype=tl.float32)
            for c in tl.static_range(HC):
                g = tl.sum(tl.where(outs == c, gates, 0.0), axis=0)
                x = tl.load(
                    x_ptr + t * stride_xt + c * stride_xc + offs, mask=mask, other=0.0
                ).to(tl.float32)
                y += g * x
            tl.store(
                y_ptr + t * stride_yt + offs, y.to(y_ptr.dtype.element_ty), mask=mask
            )


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    y_ptr,
    stride_xt,
    stride_rt,
    stride_rc,
    stride_pt,
    stride_yt,
    stride_yc,
    D: tl.constexpr,
    HC: tl.constexpr,
    CH_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """y[t, c, :] = post[t, c] * x[t, :] + residual[t, c, :] (fp32 math).

    Each program handles one tile of the hidden dim for CH_PER_PROG channels:
    all of them for large batches (x read once), a single one for decode-sized
    batches where extra programs matter more than an L2-resident re-read of x.
    """
    t = tl.program_id(0)
    pid = tl.program_id(1)
    n_tiles = tl.cdiv(D, BLOCK_D)
    tile = pid % n_tiles
    c0 = (pid // n_tiles) * CH_PER_PROG
    offs = tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + t * stride_xt + offs, mask=mask, other=0.0).to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    for i in tl.static_range(CH_PER_PROG):
        c = c0 + i
        g = tl.load(post_ptr + t * stride_pt + c).to(tl.float32)
        r = tl.load(
            res_ptr + t * stride_rt + c * stride_rc + offs, mask=mask, other=0.0
        )
        y = g * x + r.to(tl.float32)
        tl.store(
            y_ptr + t * stride_yt + c * stride_yc + offs,
            y.to(y_ptr.dtype.element_ty),
            mask=mask,
        )


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# Launch heuristics. Overridable for tuning sweeps (see kernels/hy_v4_ihc/tune.py).
CFG = {
    # tl.dot needs M >= 16; rows beyond T are masked. Larger row blocks amortize
    # the weight slice each program streams from L2 (tuned on RTX 5070 Ti).
    "block_t_small": 16,  # T < 128
    "block_t_mid": 32,  # 128 <= T < 1024
    "block_t_large": 64,  # T >= 1024
    "block_d": 64,
    "programs_per_sm": 4,
    "max_split": 32,  # bounds the partials each apply program reduces
    "warps": 4,
    "stages": 1,
    "post_block_d": 1024,
    "post_warps": 4,
    # Single-launch path (_ihc_small_kernel) for T <= small_t rows; see
    # _small_t_limit for the per-GPU crossover against the two-launch path.
    "small_t": 64,
    "small_t_consumer": 32,
    "small_t_consumer_small": 16,
    "small_block_d": 256,
    "small_apply_block_d": 1024,
    "small_warps": 4,
}
SMALL_T_MAX = 64  # arrival counters allocated per device


@functools.cache
def _is_datacenter(device_index: int) -> bool:
    """A100 / H100 / B200-class parts (sm80, sm90, sm100/103); GeForce Blackwell
    is sm120 and Ada / Ampere consumer parts have a non-zero minor version."""
    cap = current_platform.get_device_capability(device_index)
    if cap is None:
        return False
    return cap.major == 10 or (cap.major < 12 and cap.minor == 0)


def _small_t_limit(device_index: int) -> int:
    """Largest row count handled by the single-launch path.

    Its serial per-row tail (one program reduces a whole row) loses to the
    two-launch path earlier on GPUs with less per-SM bandwidth. Measured
    crossover (us, single vs two-launch, hidden 6144): H100 T=64 7.6 vs 10.3,
    A100 T=64 19.4 vs 21.4, RTX 5090 T=32 17.8 vs 21.3 but T=64 28.8 vs 22.0,
    RTX 5070 Ti T=16 6.3 vs 7.3 but T=32 9.6 vs 7.5.
    """
    if _is_datacenter(device_index):
        limit = CFG["small_t"]
    elif _sm_count(device_index) >= 128:
        limit = CFG["small_t_consumer"]
    else:
        limit = CFG["small_t_consumer_small"]
    return min(limit, SMALL_T_MAX)


def _small_split(T: int, D: int, sm_count: int) -> tuple[int, int]:
    """(SPLIT, D_PER_SPLIT) for the single-launch path."""
    BLOCK_D = CFG["small_block_d"]
    want = max(1, (CFG["programs_per_sm"] * sm_count) // T)
    SPLIT = max(1, min(want, triton.cdiv(D, BLOCK_D), CFG["max_split"]))
    D_PER_SPLIT = triton.cdiv(triton.cdiv(D, SPLIT), BLOCK_D) * BLOCK_D
    return triton.cdiv(D, D_PER_SPLIT), D_PER_SPLIT


def _pick_launch(T: int, D: int, device_index: int) -> tuple[int, int, int, int, int]:
    """Return (BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT, num_warps).

    Split the hidden dim across programs so small (decode) batches still
    occupy every SM.
    """
    if T < 128:
        BLOCK_T = CFG["block_t_small"]
    elif T < 1024:
        BLOCK_T = CFG["block_t_mid"]
    else:
        BLOCK_T = CFG["block_t_large"]
    BLOCK_D = CFG["block_d"]
    n_row_blocks = triton.cdiv(T, BLOCK_T)
    want = max(1, (CFG["programs_per_sm"] * _sm_count(device_index)) // n_row_blocks)
    max_split = min(triton.cdiv(D, BLOCK_D), CFG["max_split"])
    SPLIT = max(1, min(want, max_split))
    D_PER_SPLIT = triton.cdiv(triton.cdiv(D, SPLIT), BLOCK_D) * BLOCK_D
    SPLIT = triton.cdiv(D, D_PER_SPLIT)
    return BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT, CFG["warps"]


@functools.cache
def _row_counters(device_index: int) -> torch.Tensor:
    """Per-row arrival counters for _ihc_small_kernel (left zeroed by the
    kernel itself). One buffer per device; launches are stream-ordered."""
    return torch.zeros(
        SMALL_T_MAX, dtype=torch.int32, device=torch.device("cuda", device_index)
    )


def _ihc_reduce_small(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    magnitude: float,
    has_post: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, HC, D = x.shape
    n_out = weight.shape[0]
    device_index = x.device.index or 0
    BLOCK_D = CFG["small_block_d"]
    SPLIT, D_PER_SPLIT = _small_split(T, D, _sm_count(device_index))

    ws = torch.empty((T, SPLIT, n_out + 1), dtype=torch.float32, device=x.device)
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    post = torch.empty((T, HC if has_post else 1), dtype=torch.float32, device=x.device)
    _ihc_small_kernel[(T, SPLIT)](
        x,
        weight,
        ws,
        _row_counters(device_index),
        scale,
        base,
        y,
        post,
        x.stride(0),
        x.stride(1),
        ws.stride(0),
        ws.stride(1),
        y.stride(0),
        post.stride(0),
        D_PER_SPLIT,
        SPLIT,
        norm_eps,
        hc_eps,
        magnitude,
        D=D,
        HC=HC,
        N_OUT=n_out,
        HAS_POST=has_post,
        SPLIT_PAD=triton.next_power_of_2(SPLIT),
        BLOCK_D=BLOCK_D,
        APPLY_BLOCK_D=min(CFG["small_apply_block_d"], triton.next_power_of_2(D)),
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=CFG["small_warps"],
    )
    return y, post


def _ihc_reduce(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    magnitude: float,
    has_post: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3 and x.stride(2) == 1, (
        "x must be [T, hc, d] with unit last stride"
    )
    T, HC, D = x.shape
    n_out = weight.shape[0]
    assert weight.shape == (n_out, HC * D) and weight.is_contiguous()
    assert n_out == (2 * HC if has_post else HC)
    assert (n_out & (n_out - 1)) == 0, "hc_mult must be a power of two"
    assert weight.dtype == torch.float32
    assert x.dtype in (torch.bfloat16, torch.float16), (
        "x must be bf16/fp16 (tensor-core dot)"
    )

    if _small_t_limit(x.device.index or 0) >= T:
        return _ihc_reduce_small(
            x, weight, scale, base, norm_eps, hc_eps, magnitude, has_post
        )

    BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT, warps = _pick_launch(
        T, D, x.device.index or 0
    )
    ws = torch.empty((T, SPLIT, n_out + 1), dtype=torch.float32, device=x.device)
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    post = torch.empty((T, HC if has_post else 1), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(T, BLOCK_T), SPLIT)
    pdl = current_platform.is_arch_support_pdl()
    _ihc_stats_kernel[grid](
        x,
        weight,
        ws,
        T,
        x.stride(0),
        x.stride(1),
        ws.stride(0),
        ws.stride(1),
        D_PER_SPLIT,
        D=D,
        HC=HC,
        N_OUT=n_out,
        N_PAD=max(16, n_out),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        launch_pdl=pdl,
        num_warps=warps,
        num_stages=CFG["stages"],
    )
    _ihc_apply_kernel[grid](
        x,
        ws,
        scale,
        base,
        y,
        post,
        T,
        x.stride(0),
        x.stride(1),
        ws.stride(0),
        ws.stride(1),
        y.stride(0),
        post.stride(0),
        D_PER_SPLIT,
        SPLIT,
        norm_eps,
        hc_eps,
        magnitude,
        D=D,
        HC=HC,
        N_OUT=n_out,
        HAS_POST=has_post,
        SPLIT_PAD=triton.next_power_of_2(SPLIT),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        launch_pdl=pdl,
        num_warps=warps,
    )
    return y, post


def _ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ihc_reduce(x, weight, hc_scale, hc_base, norm_eps, hc_eps, magnitude, True)


def _ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    y, _ = _ihc_reduce(x, weight, hc_scale, hc_base, norm_eps, hc_eps, 1.0, False)
    return y


def _post_launch(T: int, D: int, HC: int, device_index: int) -> tuple[int, int, int]:
    """(BLOCK_D, n_tiles, CH_PER_PROG). Small batches use one program per
    (token, channel, tile) to fill the GPU; large ones read x once per tile."""
    BLOCK_D = min(CFG["post_block_d"], triton.next_power_of_2(D))
    n_tiles = triton.cdiv(D, BLOCK_D)
    ch_per_prog = 1 if T * n_tiles < 2 * _sm_count(device_index) else HC
    return BLOCK_D, n_tiles, ch_per_prog


def _spec_class(v: int) -> int:
    """Triton's default integer specialization class: 1, multiple of 16, other."""
    return 1 if v == 1 else (16 if v % 16 == 0 else 0)


def _launch_key(T: int, D: int, HC: int, device_index: int) -> tuple:
    sm = _sm_count(device_index)
    if _small_t_limit(device_index) >= T:
        SPLIT, D_PER_SPLIT = _small_split(T, D, sm)
        reduce_key: tuple = ("small", _spec_class(SPLIT), _spec_class(D_PER_SPLIT))
    else:
        BLOCK_T, _, SPLIT, D_PER_SPLIT, _ = _pick_launch(T, D, device_index)
        reduce_key = (
            "split",
            BLOCK_T,
            _spec_class(T),
            _spec_class(SPLIT),
            triton.next_power_of_2(SPLIT),
            _spec_class(D_PER_SPLIT),
        )
    return reduce_key, _post_launch(T, D, HC, device_index)[2]


def warmup_token_sizes(
    D: int, HC: int, max_tokens: int, device_index: int
) -> list[int]:
    """One token count per distinct Triton compile key of the three ops.

    Calling ``ihc_pre`` / ``ihc_head`` / ``ihc_post`` for each returned size
    JIT-compiles every kernel variant reachable for ``T <= max_tokens``.
    """
    seen: set[tuple] = set()
    sizes: list[int] = []
    for T in range(1, max_tokens + 1):
        key = _launch_key(T, D, HC, device_index)
        if key not in seen:
            seen.add(key)
            sizes.append(T)
    return sizes


def _ihc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    assert x.dim() == 2 and residual.dim() == 3 and post.dim() == 2
    T, HC, D = residual.shape
    assert x.shape == (T, D) and post.shape == (T, HC)
    assert x.stride(1) == 1 and residual.stride(2) == 1 and post.stride(1) == 1
    y = torch.empty_like(residual)
    BLOCK_D, n_tiles, ch_per_prog = _post_launch(T, D, HC, x.device.index or 0)
    grid = (T, n_tiles * (HC // ch_per_prog))
    _ihc_post_kernel[grid](
        x,
        residual,
        post,
        y,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        post.stride(0),
        y.stride(0),
        y.stride(1),
        D=D,
        HC=HC,
        CH_PER_PROG=ch_per_prog,
        BLOCK_D=BLOCK_D,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=CFG["post_warps"],
    )
    return y


def _ihc_pre_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, HC, D = x.shape
    return x.new_empty((T, D)), x.new_empty((T, HC), dtype=torch.float32)


def _ihc_head_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    T, HC, D = x.shape
    return x.new_empty((T, D))


def _ihc_post_fake(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="hy_v4_ihc_pre",
    op_func=_ihc_pre,
    fake_impl=_ihc_pre_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_head",
    op_func=_ihc_head,
    fake_impl=_ihc_head_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_post",
    op_func=_ihc_post,
    fake_impl=_ihc_post_fake,
)


def ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused HYV4HCPreLayer.forward: returns (reduced [T, d], post gates [T, hc])."""
    return torch.ops.vllm.hy_v4_ihc_pre(
        x, weight, hc_scale, hc_base, norm_eps, hc_eps, magnitude
    )


def ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Fused HYV4HCHeadLayer.forward: returns the merged hidden state [T, d]."""
    return torch.ops.vllm.hy_v4_ihc_head(x, weight, hc_scale, hc_base, norm_eps, hc_eps)


def ihc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """Fused HYV4HCPostLayer.forward: post[t, c] * x[t, :] + residual[t, c, :]."""
    return torch.ops.vllm.hy_v4_ihc_post(x, residual, post)


__all__ = ["ihc_head", "ihc_post", "ihc_pre", "warmup_token_sizes"]
