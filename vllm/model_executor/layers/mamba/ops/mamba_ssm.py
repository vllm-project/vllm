# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import functools
import json
import os
from contextlib import contextmanager
from typing import Any

import regex as re
import torch
from packaging import version
from pathvalidate import sanitize_filename

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

if current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "configs", "selective_state_update"
)


def get_ssm_config_file_name(
    headdim: int, dstate: int, cache_dtype: str, device_name: str
) -> str:
    """Return the JSON filename for the given kernel shape.

    Layout: ``configs/selective_state_update/
    headdim=<H>,dstate=<D>,device_name=<dev>,cache_dtype=<dt>.json``.
    """
    return (
        f"headdim={headdim},dstate={dstate},"
        f"device_name={device_name},cache_dtype={cache_dtype}.json"
    )


def get_ssm_device_name() -> str:
    name = current_platform.get_device_name()
    name = re.sub(r"[\s/-]+", "_", name)
    name = sanitize_filename(name)
    return name


def _canonical_cache_dtype(cache_dtype: str) -> str:
    """Canonical key for config lookup. bf16 and fp16 share the same tuned
    configs because the kernel only sees bit width when accessing state."""
    return "float16" if cache_dtype == "bfloat16" else cache_dtype


@functools.cache
def get_ssm_configs(
    headdim: int, dstate: int, cache_dtype: str
) -> dict[int, Any] | None:
    """
    Return tuned (BLOCK_SIZE_M, num_warps) configs for *selective_state_update*
    keyed by ``effective_batch = batch * nheads``, or ``None`` if no config
    file is found for the (headdim, dstate, cache_dtype, device) combination.

    They can be generated with:
        benchmarks/kernels/benchmark_selective_state_update.py --save-configs
    """
    cache_dtype = _canonical_cache_dtype(cache_dtype)
    device_name = get_ssm_device_name()
    json_file_name = get_ssm_config_file_name(headdim, dstate, cache_dtype, device_name)

    config_file_paths: list[str] = []

    # User-supplied override
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        config_file_paths.append(
            os.path.join(user_defined_config_folder, json_file_name)
        )

    # Bundled default
    config_file_paths.append(os.path.join(_CONFIGS_DIR, json_file_name))

    for path in config_file_paths:
        if os.path.exists(path):
            with open(path) as f:
                logger.info_once(
                    "Using SSM config from %s for selective_state_update.",
                    path,
                    scope="global",
                )
                raw = json.load(f)
                if isinstance(raw, dict):
                    # triton_version included in the config file only for reference
                    raw.pop("triton_version", None)
                    return {int(k): v for k, v in raw.items() if k.isdigit()}

    logger.warning_once(
        "Using default Mamba SSU config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
    )
    return None


def _get_default_ssm_launch_config(
    dstate: int,
    is_blackwell: bool,
) -> tuple[int, int]:
    """Hard-coded fallback heuristic used when no tuned config is available."""
    BLOCK_SIZE_M, num_warps = 4, 8
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    else:
        if is_blackwell:
            BLOCK_SIZE_M, num_warps = 32, 8
        elif dstate <= 128:
            BLOCK_SIZE_M, num_warps = 4, 4
    return BLOCK_SIZE_M, num_warps


@functools.cache
def _try_get_optimal_ssm_config_cached(
    headdim: int,
    dstate: int,
    batch: int,
    nheads: int,
    cache_dtype: str,
    is_blackwell: bool,
) -> tuple[int, int]:
    """Cached resolution. See :func:`try_get_optimal_ssm_config`."""
    effective_batch = batch * nheads
    configs = get_ssm_configs(headdim, dstate, cache_dtype)
    if configs:
        # Pick the closest effective_batch in the tuned grid (MoE strategy).
        closest = min(configs.keys(), key=lambda x: abs(x - effective_batch))
        cfg = configs[closest]
        return cfg["BLOCK_SIZE_M"], cfg["num_warps"]

    return _get_default_ssm_launch_config(dstate, is_blackwell)


# Override hook for benchmarks/tests, see `override_ssm_config`.
_ssm_config_override: tuple[int, int] | None = None


@contextmanager
def override_ssm_config(config: tuple[int, int]):
    """Pin ``try_get_optimal_ssm_config`` to ``config`` for the duration of
    the context. Used by the tuning benchmark to time specific configs."""
    global _ssm_config_override
    prev = _ssm_config_override
    _ssm_config_override = config
    try:
        yield
    finally:
        _ssm_config_override = prev


def try_get_optimal_ssm_config(
    headdim: int,
    dstate: int,
    batch: int,
    nheads: int,
    cache_dtype: str,
    is_blackwell: bool,
) -> tuple[int, int]:
    """Return (BLOCK_SIZE_M, num_warps) for the given kernel shape.

    Tuning is keyed on ``effective_batch = batch * nheads`` (the kernel grid
    scales with the product), so configs transfer across (model, TP) combos
    sharing ``(headdim, dstate, cache_dtype)``.
    """
    if _ssm_config_override is not None:
        return _ssm_config_override
    return _try_get_optimal_ssm_config_cached(
        headdim, dstate, batch, nheads, cache_dtype, is_blackwell
    )


if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt
else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.jit
def convert_rs_fp16x2(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    y = tl.inline_asm_elementwise(
        asm="""{
cvt.rs.f16x2.f32 $0, $2, $1, $3;
}""",
        constraints="=r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )
    return y


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    rand_seed_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    dst_state_batch_indices_ptr,
    null_block_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    # Matrix dimensions
    N,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    stride_dst_state_indices_batch,
    stride_dst_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos

        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0

        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(
                dst_state_batch_indices_ptr
                + init_token_idx * stride_dst_state_indices_T
            ).to(tl.int64)
            dst_state_ptr = state_ptr + (
                dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head
            )

        state_batch_indices_ptr += (
            pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T
        )
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = (
            state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        )
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    if not IS_SPEC_DECODING:
        dst_state_ptrs = dst_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != null_block_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = fast_exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = fast_exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        if IS_SPEC_DECODING:
            dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
            token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
            if token_dst_idx != null_block_id:
                token_dst_ptrs = (
                    state_ptr_base
                    + token_dst_idx * stride_state_batch
                    + pid_h * stride_state_head
                    + offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                tl.store(
                    token_dst_ptrs, state.to(token_dst_ptrs.dtype.element_ty), mask=mask
                )

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch

    if not IS_SPEC_DECODING:
        if USE_RS_ROUNDING:
            # Load random seed
            rand_seed = tl.load(rand_seed_ptr)
            # Generate random offsets for each element in state
            if HAS_STATE_BATCH_INDICES:
                rand_offsets = (
                    state_batch_idx * stride_state_batch + pid_h * stride_state_head
                )
            else:
                rand_offsets = pid_b * stride_state_batch + pid_h * stride_state_head
            rand_offsets += (
                offs_m[:, None] * stride_state_dim
                + offs_n[None, :] * stride_state_dstate
            )
            # Generate random 32-bits for each element in state
            if PHILOX_ROUNDS > 0:
                rand = tl.randint(rand_seed, rand_offsets, PHILOX_ROUNDS)
            else:
                rand = tl.randint(rand_seed, rand_offsets)
            # Convert state to fp16 with RS rounding
            state = convert_rs_fp16x2(state, rand)
            tl.static_assert(state.dtype == tl.float16, "state must be fp16")
            tl.static_assert(
                dst_state_ptrs.dtype.element_ty == tl.float16,
                "dst_state_ptrs must be fp16",
            )
        else:
            state = state.to(dst_state_ptrs.dtype.element_ty)
        tl.store(dst_state_ptrs, state, mask=mask)


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    z=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    null_block_id=NULL_BLOCK_ID,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell=False,
    enable_stochastic_rounding=False,
    cache_philox_rounds=0,
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
        null_block_id: int
            if state_batch_indices is passed, lets the kernel identify
            padded entries that will not be processed,
            for example: state_batch_indices = [null_block_id, 1, 20,
            null_block_id] in this case, the kernel will not process
            entries at indices 0 and 3
        out: Preallocated ssm output tensor. Assume same shape as x.
             In-place updated.
        num_accepted_tokens: (batch,)
            number of accepted tokens from previous verification step,
            tells the kernel which initial state to use
        cu_seqlens: (batch,)
            length per sequence, for variable length in speculative decoding cases
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)
    if dst_state_batch_indices is not None and dst_state_batch_indices.dim() == 1:
        dst_state_batch_indices = dst_state_batch_indices.unsqueeze(1)
    if num_accepted_tokens is not None:
        assert state_batch_indices is not None and state_batch_indices.dim() == 2
        assert dst_state_batch_indices is None or dst_state_batch_indices.dim() == 2

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        # Only used to verify the shape of
        # state_batch_indices and dst_state_batch_indices
        max_seqlen = (
            state_batch_indices.size(-1) if state_batch_indices is not None else 1
        )
    else:
        N = batch
        max_seqlen = 1

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape[0] >= N
        assert state_batch_indices.shape[1] >= max_seqlen
    if dst_state_batch_indices is not None:
        assert dst_state_batch_indices.shape[0] >= N
        assert dst_state_batch_indices.shape[1] >= max_seqlen
    else:
        # revert to the default behavior of in-place state updates
        dst_state_batch_indices = state_batch_indices
    assert out.shape == x.shape
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.shape == (N,)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), N, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    state_batch_indices_strides = (
        (state_batch_indices.stride(0), state_batch_indices.stride(1))
        if state_batch_indices is not None
        else (0, 0)
    )
    dst_state_batch_indices_strides = (
        (dst_state_batch_indices.stride(0), dst_state_batch_indices.stride(1))
        if dst_state_batch_indices is not None
        else (0, 0)
    )
    # We don't want autotune since it will overwrite the state.
    # Load from JSON config if available, otherwise fall back to heuristic.
    cache_dtype = str(state.dtype).removeprefix("torch.")
    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, N, nheads, cache_dtype, is_blackwell
    )

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )
    rand_seed = (
        torch.randint(0, 2**32, (1,), device=state.device)
        if enable_stochastic_rounding
        else None
    )

    with torch.accelerator.device_index(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            rand_seed,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            dst_state_batch_indices,
            null_block_id,
            num_accepted_tokens,
            cu_seqlens,
            N,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0),
            D.stride(1),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            state_batch_indices_strides[0],
            state_batch_indices_strides[1],
            dst_state_batch_indices_strides[0],
            dst_state_batch_indices_strides[1],
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
            USE_RS_ROUNDING=enable_stochastic_rounding,
            PHILOX_ROUNDS=cache_philox_rounds,
        )


def selective_scan_fn(
    u,
    ssm_states,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    null_block_id=NULL_BLOCK_ID,
    block_size=1024,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    cu_chunk_seqlen=None,
    last_chunk_indices=None,
) -> torch.Tensor:
    """
    u: (dim, total_length) for varlen or (batch, dim, seqlen)
        applies changes in place.
    ssm_states: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        applies changes in place.
    delta: (dim, total_length) for varlen or (batch, dim, seqlen)
    A: (dim, dstate)
    B: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    C: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    D: (dim,)
    z: (dim, total_length) for varlen or (batch, dim, seqlen)
    dt_bias: (dim,) or (dim)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended with 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch) int32
        A tensor with each cell is a correspondent
        input and output ssm_state indices
      - Without APC: (batch,) - single state index per batch item
      - With APC: (batch, max_positions) - cache block indices for read/write
        Each non-zero value indicates a cache block to load from and/or write to.
    has_initial_state: (batch) bool
        A tensor populated with ones and zeros,
        indicate if the ssm_state at the corresponding index should be
        used as initial state. Not providing argument assumes
        there's no initial state
    null_block_id: int
        if cache_indices is passed, lets the kernel identify padding entries
        that will not be processed,
        for example: cache_indices = [null_block_id, 1 ,20 ,null_block_id]
        in this case, the kernel will not process entries at indices 0 and 3
    block_size: int
        The block size to align the cached states to
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first
        cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block
        to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block
        containing the initial state is located.
    returns
        output: (dim, total_length) for varlen or (batch, dim, seqlen)
                supports inplace replacement
    """
    if u.stride(-1) != 1:
        u = u.contiguous()
    if delta.stride(-1) != 1:
        delta = delta.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3 and query_start_loc is None:
        B = B.unsqueeze(1)
    if B.dim() == 2 and query_start_loc is not None:
        B = B.unsqueeze(0)
    if C.dim() == 3 and query_start_loc is None:
        C = C.unsqueeze(1)
    if C.dim() == 2 and query_start_loc is not None:
        C = C.unsqueeze(0)

    if current_platform.is_xpu():
        xpu_ops.selective_scan_fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            query_start_loc,
            cache_indices,
            has_initial_state,
            ssm_states,
            null_block_id,
            block_size,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            cu_chunk_seqlen,
            last_chunk_indices,
        )
    else:
        ops.selective_scan_fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            query_start_loc,
            cache_indices,
            has_initial_state,
            ssm_states,
            null_block_id,
            block_size,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            cu_chunk_seqlen,
            last_chunk_indices,
        )

    if z is None:
        return delta  # output written inplace to delta
    else:
        return z  # output written inplace to z
