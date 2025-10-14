# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The fused-5 ssd kernel
# ruff: noqa: E501,SIM102


import torch

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.autotune(
    configs=[
        triton.Config(
            {  # A100 SXM4 80GB and H100 80GB HBM3 same config
                # head dim block for chunk state, state passing, and chunk scan
                "BLOCK_SIZE_HD": 64,
                # dstate and chunk_size blocks for chunk state and state passing
                "BLOCK_SIZE_DS": 128,
                "BLOCK_SIZE_CS": 32,
                # chunk scan config
                "CS_BLOCK_SIZE_CS_outer": 64,
                "CS_BLOCK_SIZE_CS_inner": 32,
                "CS_BLOCK_SIZE_DS": 64,
                "CS_WHOLEBLOCK_DS": 128,  # if dstate <= CS_WHOLEBLOCK_DS, we don't block along dstate
                # BMM config
                "BMM_BLOCK_SIZE_M": 64,
                "BMM_BLOCK_SIZE_N": 64,
                "BMM_BLOCK_SIZE_K": 64,
                # cumsum config
                "CCS_BLOCK_SIZE_H": 16,
            },
            num_stages=1,
            num_warps=4,
            maxnreg=128,
        ),
        # Use this for autotuning, it makes hundreds of configs though
        # You can also try other block size values not in the lists below
        # It's probably best to autotune the BMM and CCS block sizes separate from the other block sizes,
        # since they are for smaller amounts of work and are mostly independent
        # triton.Config({ 'BLOCK_SIZE_HD': BLOCK_SIZE_HD, 'BLOCK_SIZE_DS': BLOCK_SIZE_DS, 'BLOCK_SIZE_CS': BLOCK_SIZE_CS,
        #     'CS_BLOCK_SIZE_CS_outer': CS_BLOCK_SIZE_CS_outer, 'CS_BLOCK_SIZE_CS_inner': CS_BLOCK_SIZE_CS_inner, 'CS_BLOCK_SIZE_DS': CS_BLOCK_SIZE_DS,
        #     'CS_WHOLEBLOCK_DS': CS_WHOLEBLOCK_DS,
        #     'BMM_BLOCK_SIZE_M': BMM_BLOCK_SIZE_M, 'BMM_BLOCK_SIZE_N': BMM_BLOCK_SIZE_N, 'BMM_BLOCK_SIZE_K': BMM_BLOCK_SIZE_K,
        #     'CCS_BLOCK_SIZE_H': CCS_BLOCK_SIZE_H, }, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg) \
        #     for BLOCK_SIZE_HD in            [64] \
        #     for BLOCK_SIZE_DS in            [128] \
        #     for BLOCK_SIZE_CS in            [64, 128, 256] \
        #     for CS_BLOCK_SIZE_CS_outer in   [32, 64, 128, 256] \
        #     for CS_BLOCK_SIZE_CS_inner in   [32, 64, 128, 256] \
        #     for CS_BLOCK_SIZE_DS in         [128] \
        #     for CS_WHOLEBLOCK_DS in         [128] \
        #     for BMM_BLOCK_SIZE_M in         [64] \
        #     for BMM_BLOCK_SIZE_N in         [64] \
        #     for BMM_BLOCK_SIZE_K in         [64] \
        #     for CCS_BLOCK_SIZE_H in         [16] \
        #     for num_stages, num_warps, maxnreg in [(1, 4, 128), (2, 4, 256), (2, 8, 256)]
    ],
    key=["hdim", "dstate", "chunk_size", "IS_CAUSAL"],
)
@triton.heuristics(
    values={
        "NEED_MASK_HD": lambda args: args["hdim"] / args["BLOCK_SIZE_HD"]
        != args["hdim"] // args["BLOCK_SIZE_HD"],
        "NEED_MASK_CS_DS": lambda args: args["dstate"] / args["CS_BLOCK_SIZE_DS"]
        != args["dstate"] // args["CS_BLOCK_SIZE_DS"]
        or args["dstate"] != args["BLOCK_SIZE_DSTATE"],
        "NEED_MASK_CS_CS_inner": lambda args: args["chunk_size"]
        / args["CS_BLOCK_SIZE_CS_inner"]
        != args["chunk_size"] // args["CS_BLOCK_SIZE_CS_inner"],
        "NEED_MASK_CS_CS_outer": lambda args: args["chunk_size"]
        / args["CS_BLOCK_SIZE_CS_outer"]
        != args["chunk_size"] // args["CS_BLOCK_SIZE_CS_outer"],
        "NEED_MASK_1_DS": lambda args: args["dstate"] / args["BLOCK_SIZE_DS"]
        != args["dstate"] // args["BLOCK_SIZE_DS"],
    }
)
@triton.jit
def _fused5_ssd_kernel(
    # Synchronization
    first2_wait_ptr,
    first2_wait_stride_chunk,
    grid_atomic,
    USE_ATOMIC_PID: tl.constexpr,
    sync_atomic,
    stride_sync_head,
    stride_sync_hdim,
    stride_sync_dstate,
    # Tensor dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    nheads: tl.constexpr,
    nchunks,
    ngroups: tl.constexpr,
    # Tensor ptrs
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    states_G_ptr,
    initial_states_ptr,
    cu_chunk_seqlens_ptr,
    cb_ptr,
    out_ptr,
    out_x_ptr,
    C_ptr,
    D_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_orig_ptr,
    # Tensor strides
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_chunk,
    stride_states_G_chunk,
    stride_states_G_head,
    stride_states_G_hdim,
    stride_states_G_dstate,
    stride_initial_states_batch,
    stride_initial_states_head,
    stride_initial_states_hdim,
    stride_initial_states_dstate,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_D_head,
    stride_dt_orig_seqlen,
    stride_dt_orig_head,
    stride_A_head,
    stride_dt_bias_head,
    # dt limits
    dt_min,
    dt_max,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    CB_SCALE_FP32: tl.constexpr,
    CS_ACC_FP32: tl.constexpr,
    CB_COMP_FP32: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_HD: tl.constexpr,
    BLOCK_SIZE_DS: tl.constexpr,
    BLOCK_SIZE_CS: tl.constexpr,
    CS_BLOCK_SIZE_DS: tl.constexpr,
    CS_BLOCK_SIZE_CS_outer: tl.constexpr,
    CS_BLOCK_SIZE_CS_inner: tl.constexpr,
    CS_WHOLEBLOCK_DS: tl.constexpr,
    BMM_BLOCK_SIZE_M: tl.constexpr,
    BMM_BLOCK_SIZE_N: tl.constexpr,
    BMM_BLOCK_SIZE_K: tl.constexpr,
    CCS_BLOCK_SIZE_H: tl.constexpr,
    # pwr2 dim constexprs
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
    # Heuristic mask bools
    NEED_MASK_HD: tl.constexpr,
    NEED_MASK_CS_DS: tl.constexpr,
    NEED_MASK_CS_CS_outer: tl.constexpr,
    NEED_MASK_CS_CS_inner: tl.constexpr,
    NEED_MASK_1_DS: tl.constexpr,
):
    """
    This fused Mamba2 SSD kernel combines the 5 original SSD kernels.
    There are a few important things to keep in mind when using it:
    * This kernel assumes that a warp resident in an SM (already executed some instructions)
    is not permanently starved if other warps are spamming atomic instructions.
    I think this is true for Volta+ (V100 and later).
    * This kernel is extremely sensitive to register pressure. Any modifications should be done carefully.
    The config is extremely sensitive, and exhaustive autotuning can generate hundreds of configs.
    * The config and / or kernel may need slight changes to be optimal for other models, currently tuned on Mamba2-2.7B.
    * This kernel can handle larger seqlen than the original kernels (which get either bad output or illegal memory accesses from int32 overflow).
    If you do get illegal memory access for extreme seqlen, you might need to cast more strides (e.g. any stride dependent on seqlen or nchunks) to int64.
    * This kernel could have very slightly different output due to a different order of operations and casting (fp16 intermediate results instead of fp32),
    * NEED_MASK_*=True is not tested, and this kernel was only tested on an A100 and H100.

    :param first2_wait_ptr: The atomic sync tensor for waiting for bmm and cumsum to be ready
    :param grid_atomic: The atomic counter to get pids from, making sure that previous pids are either concurrent or finished
    :param USE_ATOMIC_PID: If False, we assume that the GPU driver will launch pids in increasing order
    :param sync_atomic: The atomic sync tensor for waiting for state passing chunk local states to be ready
    :param dt_ptr: the new dt tensor generated by cumsum, which is chunked
    :param dA_cumsum_ptr: the dA_cumsum tensor generated by cumsum
    :param states_G_ptr: the global states tensor, including initial states and final states
    :param cb_ptr: space for CB
    :param dt_orig_ptr: the input dt, which is not chunked
    :param BLOCK_SIZE_HD: the head dim (hdim) block for chunk state, state passing, and chunk scan
    :param BLOCK_SIZE_DS: the state dim (dstate) block for chunk state and state passing
    :param BLOCK_SIZE_CS: the chunk size block for chunk state (matmul loop k)
    :param CS_WHOLEBLOCK_DS: threshold above which we use CS_BLOCK_SIZE_DS instead of the full state dim (dstate)
    :param CS_BLOCK_SIZE_DS: the state dim (dstate) block for chunk scan, ignored if dstate <= CS_WHOLEBLOCK_DS
    :param CS_BLOCK_SIZE_CS_outer: the chunk size block for chunk scan along rows of CB
    :param CS_BLOCK_SIZE_CS_inner: the chunk size block for chunk scan inner dim (CB columns, x rows)
    :param BMM_BLOCK_SIZE_M: BMM M dim (chunk size along C)
    :param BMM_BLOCK_SIZE_N: BMM N dim (chunk size along B)
    :param BMM_BLOCK_SIZE_K: BMM K dim (dstate inner dim)
    :param CCS_BLOCK_SIZE_H: the block along heads for cumsum
    :param BLOCK_SIZE_DSTATE: the state dim (dstate) rounded up pwr2
    :param BLOCK_SIZE_CHUNK: the chunk size rounded up pwr2
    """

    if USE_ATOMIC_PID:
        # order does not matter, just need previous threadblocks concurrently running or finished
        pid_og = tl.atomic_add(grid_atomic, 1, sem="relaxed")
    else:
        pid_og = tl.program_id(0)

    ccs_num_pid_h = tl.cdiv(nheads, CCS_BLOCK_SIZE_H)
    num_pids_css = nchunks * ccs_num_pid_h
    num_pid_n = tl.cdiv(chunk_size, BMM_BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(chunk_size, BMM_BLOCK_SIZE_M)
    num_pids_bmm = num_pid_n * num_pid_m * nchunks * ngroups

    ########################################
    # Chunk Cumsum
    ########################################
    if pid_og < num_pids_css:
        pid_ccs = pid_og
        pid_c = (pid_ccs) % nchunks
        pid_h = (pid_ccs // nchunks) % ccs_num_pid_h

        # where this sequence starts and stops in the non-padded combined seqlen
        chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
        chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

        css_dt_ptr = dt_orig_ptr + chunk_seqlen_start * stride_dt_orig_seqlen
        css_dt_out_ptr = dt_ptr + pid_c * stride_dt_chunk
        css_dA_cumsum_ptr = dA_cumsum_ptr + pid_c * stride_dA_cs_chunk

        offs_h = pid_h * CCS_BLOCK_SIZE_H + tl.arange(0, CCS_BLOCK_SIZE_H)
        offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
        dt_ptrs = css_dt_ptr + (
            offs_h[:, None] * stride_dt_orig_head
            + offs_c[None, :] * stride_dt_orig_seqlen
        )
        A_ptrs = A_ptr + offs_h * stride_A_head
        dt_out_ptrs = css_dt_out_ptr + (
            offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_csize
        )
        dA_cs_ptrs = css_dA_cumsum_ptr + (
            offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
        )
        chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

        dt = tl.load(
            dt_ptrs,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
            other=0.0,
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(
                dt_bias_ptr + offs_h * stride_dt_bias_head,
                mask=offs_h < nheads,
                other=0.0,
            ).to(tl.float32)
            dt += dt_bias[:, None]
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        dt = tl.clamp(dt, dt_min, dt_max)
        dt = tl.where(
            (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
        )
        tl.store(
            dt_out_ptrs,
            dt,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
        )
        A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dA = dt * A[:, None]
        dA_cs = tl.cumsum(dA, axis=1)
        tl.store(
            dA_cs_ptrs,
            dA_cs,
            mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
        )

        # mark progress
        tl.atomic_add(
            first2_wait_ptr + pid_c * first2_wait_stride_chunk,
            CCS_BLOCK_SIZE_H,
            sem="release",
        )
    ########################################
    # BMM (CB)
    ########################################
    elif pid_og < num_pids_css + num_pids_bmm:
        pid_bmm = pid_og - num_pids_css
        pid_n = pid_bmm % num_pid_n
        pid_m = (pid_bmm // num_pid_n) % num_pid_m
        pid_c = (pid_bmm // (num_pid_n * num_pid_m)) % nchunks
        pid_h = (pid_bmm // (num_pid_n * num_pid_m * nchunks)) % ngroups

        chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
        chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

        if not IS_CAUSAL or pid_n * BMM_BLOCK_SIZE_N < (pid_m + 1) * BMM_BLOCK_SIZE_M:
            a_ptr = C_ptr + chunk_seqlen_start * stride_C_seqlen + pid_h * stride_C_head
            b_ptr_bmm = (
                b_ptr + chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head
            )

            offs_m = pid_m * BMM_BLOCK_SIZE_M + tl.arange(0, BMM_BLOCK_SIZE_M)
            offs_n = pid_n * BMM_BLOCK_SIZE_N + tl.arange(0, BMM_BLOCK_SIZE_N)
            offs_k = tl.arange(0, BMM_BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_m[:, None] * stride_C_seqlen + offs_k[None, :] * stride_C_dstate
            )
            b_ptrs = b_ptr_bmm + (
                offs_k[:, None] * stride_b_dstate + offs_n[None, :] * stride_b_seqlen
            )
            chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

            acc = tl.zeros(
                (BMM_BLOCK_SIZE_M, BMM_BLOCK_SIZE_N),
                dtype=tl.float32 if CB_COMP_FP32 else cb_ptr.dtype.element_ty,
            )
            for k in range(0, tl.cdiv(dstate, BMM_BLOCK_SIZE_K)):
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit)
                    & (offs_k[None, :] < dstate - k * BMM_BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < dstate - k * BMM_BLOCK_SIZE_K)
                    & (offs_n[None, :] < chunk_size_limit),
                    other=0.0,
                )
                if CB_COMP_FP32:
                    a = a.to(acc.dtype)
                    b = b.to(acc.dtype)
                acc += tl.dot(a, b, out_dtype=acc.dtype)
                a_ptrs += BMM_BLOCK_SIZE_K * stride_C_dstate
                b_ptrs += BMM_BLOCK_SIZE_K * stride_b_dstate

            offs_m = pid_m * BMM_BLOCK_SIZE_M + tl.arange(0, BMM_BLOCK_SIZE_M)
            offs_n = pid_n * BMM_BLOCK_SIZE_N + tl.arange(0, BMM_BLOCK_SIZE_N)

            out_ptr_cb = cb_ptr + pid_c * stride_cb_chunk + pid_h * stride_cb_head
            out_ptrs_cb = out_ptr_cb + (
                stride_cb_csize_m * offs_m[:, None]
                + offs_n[None, :] * stride_cb_csize_k
            )
            tl.store(
                out_ptrs_cb,
                acc,
                mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
            )
        # mark progress
        tl.atomic_add(
            first2_wait_ptr + pid_c * first2_wait_stride_chunk,
            BMM_BLOCK_SIZE_M * BMM_BLOCK_SIZE_N,
            sem="release",
        )

    # is this threadblock for the last 3 fused kernels?
    if pid_og < (num_pids_css + num_pids_bmm):
        return
    pid_fused3 = pid_og - (num_pids_css + num_pids_bmm)

    # pids for heads and chunks are the same for chunk state, state passing, and chunk scan
    # all pids represent domain parallelism except pid_c for state passing (which becomes serialized due to sync)
    num_pid_ds = tl.cdiv(dstate, BLOCK_SIZE_DS)
    num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD)
    pid_h = pid_fused3 % nheads
    pid_hd = (pid_fused3 // (nheads)) % num_pid_hd
    pid_c = (pid_fused3 // (nheads * num_pid_hd)) % nchunks

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    # advance ptrs up front to simplify and slightly reduce register pressure
    # does actually provide a small benefit vs the original separate ptrs per step
    states_G_ptr += pid_h * stride_states_G_head + (pid_c - 1) * stride_states_G_chunk
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += (
        chunk_seqlen_start * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    C_ptr += (
        chunk_seqlen_start * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )
    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    sync_atomic += (
        pid_h * stride_sync_head + pid_hd * stride_sync_hdim
    )  # + pid_ds * stride_sync_dstate

    ########################################
    # Chunk State
    ########################################

    # wait for this chunk
    first2_wait_ptr += pid_c * first2_wait_stride_chunk
    first2_wait_val = tl.atomic_add(first2_wait_ptr, 0, sem="acquire")
    # includes both, cumsum + bmm
    while (
        first2_wait_val
        < nheads + num_pid_n * BMM_BLOCK_SIZE_N * num_pid_m * BMM_BLOCK_SIZE_M * ngroups
    ):
        first2_wait_val = tl.atomic_add(first2_wait_ptr, 0, sem="acquire")

    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)

    for pid_ds in range(0, num_pid_ds, 1):
        # chunk state offsets
        # NOTE: m ->hdim, n -> dstate, k -> chunk_size
        offs_ds = pid_ds * BLOCK_SIZE_DS + tl.arange(0, BLOCK_SIZE_DS)
        offs_cs = tl.arange(0, BLOCK_SIZE_CS)

        # chunk state ptr blocks
        x_ptrs_cs = x_ptr + (
            offs_hd[:, None] * stride_x_hdim + offs_cs[None, :] * stride_x_seqlen
        )
        b_ptrs_cs = b_ptr + (
            offs_ds[None, :] * stride_b_dstate + offs_cs[:, None] * stride_b_seqlen
        )
        dt_ptrs_cs = dt_ptr + offs_cs * stride_dt_csize
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
            tl.float32
        )
        dA_cumsum_ptrs_cs = dA_cumsum_ptr + offs_cs * stride_dA_cs_csize

        # chunk state chunk_size loop
        acc_dtype = (
            tl.float32
            if states_G_ptr.dtype.element_ty == tl.bfloat16
            else states_G_ptr.dtype.element_ty
        )
        acc = tl.zeros((BLOCK_SIZE_HD, BLOCK_SIZE_DS), dtype=acc_dtype)
        for k in range(0, chunk_size_limit, BLOCK_SIZE_CS):
            if (not NEED_MASK_HD) and (not NEED_MASK_1_DS):
                x = tl.load(
                    x_ptrs_cs,
                    mask=(offs_cs[None, :] < chunk_size_limit - k),
                    other=0.0,
                    cache_modifier=".cg",
                )
                b = tl.load(
                    b_ptrs_cs,
                    mask=(offs_cs[:, None] < chunk_size_limit - k),
                    other=0.0,
                    eviction_policy="evict_first",
                )
            else:
                x = tl.load(
                    x_ptrs_cs,
                    mask=(offs_hd[:, None] < hdim)
                    & (offs_cs[None, :] < chunk_size_limit - k),
                    other=0.0,
                    cache_modifier=".cg",
                )
                b = tl.load(
                    b_ptrs_cs,
                    mask=(offs_cs[:, None] < chunk_size_limit - k)
                    & (offs_ds[None, :] < dstate),
                    other=0.0,
                    eviction_policy="evict_first",
                )
            dA_cs_k = tl.load(
                dA_cumsum_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0
            ).to(tl.float32)

            dt_k = tl.load(
                dt_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0
            ).to(tl.float32)

            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
            x *= (scale[None, :]).to(x_ptr.dtype.element_ty)
            acc += tl.dot(x, b, out_dtype=acc.dtype)
            x_ptrs_cs += BLOCK_SIZE_CS * stride_x_seqlen
            b_ptrs_cs += BLOCK_SIZE_CS * stride_b_seqlen
            dt_ptrs_cs += BLOCK_SIZE_CS * stride_dt_csize
            dA_cumsum_ptrs_cs += BLOCK_SIZE_CS * stride_dA_cs_csize

        states = acc

        ########################################
        # State Passing
        ########################################
        state_G_ptrs = (
            states_G_ptr
            + offs_hd[:, None] * stride_states_G_hdim
            + offs_ds[None, :] * stride_states_G_dstate
        )

        main_mask = (
            None
            if ((not NEED_MASK_HD) and (not NEED_MASK_1_DS))
            else (offs_hd < hdim)[:, None] & (offs_ds < dstate)[None, :]
        )

        # offset gets us to the end of each chunk
        dA_cs = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
            tl.float32
        )
        scale = tl.exp(dA_cs)

        seq_idx_prev = tl.load(
            seq_idx_ptr + (pid_c - 1) * stride_seq_idx_chunk, mask=pid_c > 0, other=-1
        )
        seq_idx_new = tl.load(seq_idx_ptr + pid_c * stride_seq_idx_chunk)

        # sync
        # the atomic represents which pid_c is ready
        # therefore, wait for it to reach our pid_c
        sync_val = tl.atomic_add(sync_atomic, 0, sem="acquire")
        while sync_val < pid_c:
            sync_val = tl.atomic_add(sync_atomic, 0, sem="acquire")

        states_prev_ptrs = state_G_ptrs
        # might need to swap with initial state
        if HAS_INITSTATES:
            if seq_idx_prev != seq_idx_new:
                states_prev_ptrs = (
                    initial_states_ptr
                    + pid_h * stride_initial_states_head
                    + seq_idx_new * stride_initial_states_batch
                    + offs_hd[:, None] * stride_initial_states_hdim
                    + offs_ds[None, :] * stride_initial_states_dstate
                )

        if seq_idx_new != seq_idx_prev and not HAS_INITSTATES:
            states_prev = tl.zeros(
                states.shape, dtype=states_prev_ptrs.dtype.element_ty
            )
        else:
            if (not NEED_MASK_HD) and (not NEED_MASK_1_DS):
                states_prev = tl.load(states_prev_ptrs)
            else:
                states_prev = tl.load(states_prev_ptrs, mask=main_mask, other=0.0)

        states_mod = (
            scale * states_prev + states
        )  # NOTE: scale.to(tl.float16) seems to slow it down

        # ptrs
        state_G_ptrs += stride_states_G_chunk  # offset since 0 gets initial states

        if (not NEED_MASK_HD) and (not NEED_MASK_1_DS):
            tl.store(state_G_ptrs, states_mod)
        else:
            tl.store(state_G_ptrs, states_mod, mask=main_mask)
        # let the next one go
        tl.atomic_add(sync_atomic, 1, sem="release")

        sync_atomic += stride_sync_dstate

    ########################################
    # Chunk Scan
    ########################################
    # pids same for all 3 parts
    # all pids represent domain parallelism except pid_c for state passing
    cs_num_pid_cs = tl.cdiv(chunk_size, CS_BLOCK_SIZE_CS_outer)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk

    prev_state_base_ptr = states_G_ptr
    prev_state_stride_hdim = stride_states_G_hdim
    prev_state_stride_dstate = stride_states_G_dstate

    seq_idx_prev = tl.load(
        seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1
    )
    seq_idx_m = tl.load(seq_idx_ptr)  # current seq idx
    if HAS_INITSTATES:  # if new sequence, switch to initial states
        if seq_idx_prev != seq_idx_m:
            # - replace prev_states_ptr with init_states
            prev_state_base_ptr = (
                initial_states_ptr
                + seq_idx_m * stride_initial_states_batch
                + pid_h * stride_initial_states_head
            )
            prev_state_stride_hdim = stride_initial_states_hdim  # override strides
            prev_state_stride_dstate = stride_initial_states_dstate

    for pid_cs in range(0, cs_num_pid_cs, 1):
        offs_cs = pid_cs * CS_BLOCK_SIZE_CS_outer + tl.arange(0, CS_BLOCK_SIZE_CS_outer)
        if not NEED_MASK_CS_CS_outer:
            dA_cs_m = tl.load(dA_cumsum_ptr + offs_cs * stride_dA_cs_csize).to(
                tl.float32
            )
        else:
            dA_cs_m = tl.load(
                dA_cumsum_ptr + offs_cs * stride_dA_cs_csize,
                mask=offs_cs < chunk_size,
                other=0.0,
            ).to(tl.float32)

        acc_dtype = (
            tl.float32
            if out_ptr.dtype.element_ty == tl.bfloat16
            else out_ptr.dtype.element_ty
        )
        acc_dtype = acc_dtype if not CS_ACC_FP32 else tl.float32
        acc = tl.zeros((CS_BLOCK_SIZE_CS_outer, BLOCK_SIZE_HD), dtype=acc_dtype)

        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size CS_WHOLEBLOCK_DS
        offs_k_dstate = tl.arange(
            0,
            BLOCK_SIZE_DSTATE
            if BLOCK_SIZE_DSTATE <= CS_WHOLEBLOCK_DS
            else CS_BLOCK_SIZE_DS,
        )
        C_ptrs = C_ptr + (
            offs_cs[:, None] * stride_C_seqlen
            + offs_k_dstate[None, :] * stride_C_dstate
        )
        prev_states_ptrs = prev_state_base_ptr + (
            offs_hd[None, :] * prev_state_stride_hdim
            + offs_k_dstate[:, None] * prev_state_stride_dstate
        )

        # add previous chunk affect if needed
        if seq_idx_prev == seq_idx_m or HAS_INITSTATES:
            scale_m = tl.exp(dA_cs_m)
            if BLOCK_SIZE_DSTATE <= CS_WHOLEBLOCK_DS:
                if (not NEED_MASK_HD) and (not NEED_MASK_CS_DS):
                    C = tl.load(
                        C_ptrs,
                        mask=(offs_cs[:, None] < chunk_size_limit),
                        other=0.0,
                    )
                    prev_states = tl.load(prev_states_ptrs)
                else:
                    C = tl.load(
                        C_ptrs,
                        mask=(offs_cs[:, None] < chunk_size_limit)
                        & (offs_k_dstate[None, :] < dstate),
                        other=0.0,
                    )
                    prev_states = tl.load(
                        prev_states_ptrs,
                        mask=(offs_k_dstate[:, None] < dstate)
                        & (offs_hd[None, :] < hdim),
                        other=0.0,
                    )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc = tl.dot(C, prev_states, out_dtype=acc.dtype) * (
                    scale_m[:, None]
                ).to(acc.dtype)
            else:
                for k in range(0, dstate, CS_BLOCK_SIZE_DS):
                    if (not NEED_MASK_HD) and (not NEED_MASK_CS_DS):
                        C = tl.load(
                            C_ptrs,
                            mask=(offs_cs[:, None] < chunk_size_limit),
                            other=0.0,
                        )
                        prev_states = tl.load(prev_states_ptrs)
                    else:
                        C = tl.load(
                            C_ptrs,
                            mask=(offs_cs[:, None] < chunk_size_limit)
                            & (offs_k_dstate[None, :] < dstate - k),
                            other=0.0,
                        )
                        prev_states = tl.load(
                            prev_states_ptrs,
                            mask=(offs_k_dstate[:, None] < dstate - k)
                            & (offs_hd[None, :] < hdim),
                            other=0.0,
                        )
                    prev_states = prev_states.to(C_ptr.dtype.element_ty)
                    acc += tl.dot(C, prev_states, out_dtype=acc.dtype)
                    C_ptrs += CS_BLOCK_SIZE_DS
                    prev_states_ptrs += CS_BLOCK_SIZE_DS
                acc *= (scale_m[:, None]).to(acc.dtype)

        offs_k = tl.arange(0, CS_BLOCK_SIZE_CS_inner)
        cb_ptrs = cb_ptr + (
            offs_cs[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
        )
        x_ptrs = x_ptr + (
            offs_k[:, None] * stride_x_seqlen + offs_hd[None, :] * stride_x_hdim
        )
        dt_ptrs = dt_ptr + offs_k * stride_dt_csize
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
        K_MAX = (
            chunk_size_limit
            if not IS_CAUSAL
            else min((pid_cs + 1) * CS_BLOCK_SIZE_CS_outer, chunk_size_limit)
        )
        for k in range(0, K_MAX, CS_BLOCK_SIZE_CS_inner):
            # NOTE: CB, dA, and dt (dt_out) are always allocated in chunks, it's always fine to read beyond seqlen within a chunk
            if (not NEED_MASK_CS_CS_outer) and (not NEED_MASK_CS_CS_inner):
                cb = tl.load(cb_ptrs, eviction_policy="evict_last")
                dA_cs_k = tl.load(dA_cumsum_ptrs).to(tl.float32)
                dt_k = tl.load(dt_ptrs).to(tl.float32 if CB_SCALE_FP32 else acc.dtype)
            else:
                cb = tl.load(
                    cb_ptrs,
                    mask=(offs_cs[:, None] < chunk_size)
                    & (offs_k[None, :] < chunk_size - k),
                    other=0.0,
                    eviction_policy="evict_last",
                )
                dA_cs_k = tl.load(
                    dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0
                ).to(tl.float32)
                dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
                    tl.float32 if CB_SCALE_FP32 else acc.dtype
                )
            # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
            # So we don't need masking wrt seq_idx here.
            if CB_SCALE_FP32:
                cb = (
                    cb.to(tl.float32)
                    * tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
                    * dt_k
                ).to(acc.dtype)
            else:
                cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :]).to(acc.dtype)
                cb *= dt_k

            if not NEED_MASK_HD:
                x = tl.load(
                    x_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - k),
                    other=0.0,
                    eviction_policy="evict_last",
                )
            else:
                x = tl.load(
                    x_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - k)
                    & (offs_hd[None, :] < hdim),
                    other=0.0,
                    eviction_policy="evict_last",
                )
            if IS_CAUSAL:
                mask = offs_cs[:, None] >= k + offs_k[None, :]
                cb = tl.where(mask, cb, 0.0)
            cb = cb.to(x.dtype)
            acc += tl.dot(cb, x, out_dtype=acc.dtype)
            cb_ptrs += CS_BLOCK_SIZE_CS_inner * stride_cb_csize_k
            x_ptrs += CS_BLOCK_SIZE_CS_inner * stride_x_seqlen
            dt_ptrs += CS_BLOCK_SIZE_CS_inner * stride_dt_csize
            dA_cumsum_ptrs += CS_BLOCK_SIZE_CS_inner * stride_dA_cs_csize

        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(
                    D_ptr + pid_h * stride_D_head + offs_hd,
                    mask=offs_hd < hdim,
                    other=0.0,
                )
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head)
            if not NEED_MASK_HD:
                x_residual = tl.load(
                    x_ptr
                    + (
                        offs_cs[:, None] * stride_x_seqlen
                        + offs_hd[None, :] * stride_x_hdim
                    ),
                    mask=(offs_cs[:, None] < chunk_size_limit),
                    other=0.0,
                )
            else:
                x_residual = tl.load(
                    x_ptr
                    + (
                        offs_cs[:, None] * stride_x_seqlen
                        + offs_hd[None, :] * stride_x_hdim
                    ),
                    mask=(offs_cs[:, None] < chunk_size_limit)
                    & (offs_hd[None, :] < hdim),
                    other=0.0,
                )
            acc += x_residual * D

        out_ptrs = out_ptr + (
            stride_out_seqlen * offs_cs[:, None] + offs_hd[None, :] * stride_out_hdim
        )
        tl.store(
            out_ptrs,
            acc,
            mask=(offs_cs[:, None] < chunk_size_limit) & (offs_hd[None, :] < hdim),
            eviction_policy="evict_first",
        )


def _fused5_ssd(
    x,
    dt,
    A,
    B,
    C,
    D,
    out,
    cu_chunk_seqlens,
    seq_idx,
    chunk_size,
    initial_states=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    use_atomic_pid=True,
):
    """
    Runs the Mamba2 SSD with 1 large fused kernel instead of the 5 original kernels,
    should be about 2.5x, 1.5x, or 2x faster for small, medium, or large sizes on an A100 or H100.
    Note:
    * Only tested on an A100 and H100
    * Optimized and tested for Mamba2-2.7B fp16 (headdim=64, dstate=128, etc)
    * Can handle larger seqlen than the original
    * Could have slightly different output, but very close

    :param x: The input x
    :param dt: The delta time dt
    :param A: SSM A (for old state -> state)
    :param B: SSM B (for x -> state)
    :param C: SSM C (for state -> y)
    :param D: SSM D (for x -> y)
    :param out: The preallocated output
    :param chunk_size: The chunk size, i.e. base case / size for the Mamba2 efficient algorithm to use Tensor Cores at. Tested for 128 and 256 only.
    :param initial_states: The initial states to start with, can be used for chunked prefil or starting from a precomputed state.
    :param seq_idx: Can be used for variable seqlen, see https://github.com/state-spaces/mamba/issues/383. Not fully tested.
    :param z: The non-SSM path (not to be confused with residual x) like an MLP gate. Not tested because for Mamba2-2.7B the layernorm handles it.
    :param states_in_fp32: Should the states be in fp32 instead of fp16? Recommended false.
    :param out_dtype: The type for the output (fp16 recommended)
    :param use_atomic_pid: If False, we assume that the GPU driver will launch pids in increasing order. Might not be necessary for particular GPUs and GPU drivers, but doesn't cost much performance.
    :param dt_bias: bias for dt
    :param dt_softplus: should we use the softplus function on dt?
    :param dt_limit: clamp for dt
    """
    # precision settings
    cb_store_fp32 = True
    cb_scale_fp32 = True
    cs_acc_fp32 = False
    cb_comp_fp32 = True

    seqlen, nheads, hdim = x.shape
    assert z is None
    # setup from chunk cumsum
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    assert (
        cu_chunk_seqlens is not None
        and cu_chunk_seqlens.dim() == 1
        and cu_chunk_seqlens.stride(0) == 1
    )
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    # setup from chunk state
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    state_dtype = state_dtype if state_dtype is not None else C.dtype
    # setup from bmm
    CB = torch.empty(
        (nchunks, ngroups, chunk_size, chunk_size),
        device=C.device,
        dtype=torch.float32 if cb_store_fp32 else torch.float16,
    )
    # setup from state passing
    dA_chunk_cumsum = dA_cumsum[:, :, -1]
    assert dA_chunk_cumsum.shape == (nheads, nchunks)
    assert seq_idx is not None and seq_idx.shape == (nchunks,)

    states_G = torch.empty(
        (nchunks, nheads, hdim, dstate), device=x.device, dtype=state_dtype
    )
    # setup from chunk scan
    assert C.shape == (seqlen, ngroups, dstate)
    assert CB.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, hdim) or D.shape == (nheads,)
    # Allocates output.
    # out = torch.empty(seqlen, nheads, hdim, device=x.device, dtype=x.dtype)
    out_x = None

    if initial_states is not None:
        num_varlen_seqs = initial_states.shape[0]
        assert initial_states.shape == (num_varlen_seqs, nheads, hdim, dstate)
        assert initial_states.dtype == states_G.dtype

    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    grid = lambda META: (
        # Chunk Cumsum grid
        nchunks * triton.cdiv(nheads, META["CCS_BLOCK_SIZE_H"])
        +
        # BMM grid
        triton.cdiv(chunk_size, META["BMM_BLOCK_SIZE_N"])
        * triton.cdiv(chunk_size, META["BMM_BLOCK_SIZE_M"])
        * nchunks
        * ngroups
        +
        # fused3 grid
        nchunks * nheads * triton.cdiv(hdim, META["BLOCK_SIZE_HD"]),
    )

    # 32 is for cache lines, dstate is not used here
    states_ready_size = nheads * hdim * dstate
    grid_atomic_size = 1 * 32
    bmm_ready_size = nchunks * 32
    sync_atomic = torch.zeros(
        (states_ready_size + grid_atomic_size + bmm_ready_size,),
        dtype=torch.int32,
        device=x.device,
    )

    nheads_ngroups_ratio = nheads // ngroups
    _fused5_ssd_kernel[grid](
        # Synchronization
        # bmm_wait_ptr, bmm_wait_stride_chunk,
        sync_atomic[
            states_ready_size + grid_atomic_size : states_ready_size
            + grid_atomic_size
            + 1
        ],
        32,
        # grid_atomic, use_atomic_pid
        # sync_atomic, sync_atomic.stride(0), sync_atomic.stride(1), sync_atomic.stride(2), sync_atomic.stride(3),
        sync_atomic[states_ready_size : states_ready_size + 1],
        use_atomic_pid,
        sync_atomic,
        hdim * dstate,
        dstate,
        1,
        # Matrix dimensions
        hdim,
        dstate,
        chunk_size,
        seqlen,
        nheads_ngroups_ratio,
        nheads,
        nchunks,
        ngroups,
        # Tensor ptrs
        x,
        B,
        dt_out,
        dA_cumsum,
        seq_idx,
        states_G,
        initial_states,
        cu_chunk_seqlens,
        CB,
        out,
        out_x,
        C,
        D,
        A,
        dt_bias,
        dt,
        # Tensor strides
        x.stride(0),
        x.stride(1),
        x.stride(2),  # stride_x_seqlen, stride_x_head, stride_x_hdim,
        B.stride(0),
        B.stride(1),
        B.stride(-1),  # stride_b_seqlen, stride_b_head, stride_b_dstate,
        dt_out.stride(1),
        dt_out.stride(0),
        dt_out.stride(2),  # stride_dt_chunk, stride_dt_head, stride_dt_csize,
        dA_cumsum.stride(1),
        dA_cumsum.stride(0),
        dA_cumsum.stride(
            2
        ),  # stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
        seq_idx.stride(0),  # stride_seq_idx_chunk
        states_G.stride(0),
        states_G.stride(1),
        states_G.stride(2),
        states_G.stride(3),
        *initial_states_strides,
        CB.stride(0),
        CB.stride(1),
        CB.stride(2),
        CB.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        D.stride(0) if D is not None else 0,
        dt.stride(0),
        dt.stride(1),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        # dt limits
        dt_limit[0],
        dt_limit[1],
        # Meta-parameters
        IS_CAUSAL=True,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        HAS_INITSTATES=initial_states is not None,
        CB_SCALE_FP32=cb_scale_fp32,
        CS_ACC_FP32=cs_acc_fp32,
        CB_COMP_FP32=cb_comp_fp32,
    )

    return out_x, states_G, dA_cumsum, dt_out
