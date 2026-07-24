# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for the FlashInfer CuTeDSL GDN "ucache" spec-decode kernel.

Replaces the Triton ``gdn_replayssm_spec_decode`` verify+flush pair with the
single fused kernel ``gated_delta_rule_mtp_ucache_flush`` (ReplaySSM Alg. 8):
per-request device-side routing runs the verify path for rows with
``hist_len < flush_min`` and additionally folds the ring into the
checkpoint (and restarts the ring at slots ``[0, T)``) for rows at or past the
threshold. One kernel per layer per step; CUDA-graph capturable.

Ring page layout on this backend (allocated via ``ring_slots=16``):
  page[1] = checkpoint  [blocks, HV, V, K] fp16 default (GDN_UCACHE_STATE_DTYPE)
  page[2] = u_cache     [blocks, HV, 16, V] fp16 default (GDN_UCACHE_RING_DTYPE)
  page[3] = k_cache     [blocks, HK, 16, K] fp16 default (GDN_UCACHE_RING_DTYPE)
  page[4] = g_cache     [blocks, HV, 16]    f32   (abs cumulative log-decay)

Cursor model: ONE block-keyed persistent ``hist_len`` buffer, committed
eagerly in the metadata builder (outside any captured region) by
``commit_gdn_ucache_hist`` with the previous step's acceptance:
``new = (old >= flush_min ? 0 : old) + accepted`` — the ring restart of a
flush step is folded into the commit, so ``hist_len`` is strictly read-only
inside the captured forward (the kernel wrapper is called with
``restart_hist_on_flush=False``; see the kernel-repo flag). The builder
gathers the block-keyed values into fixed-address request-keyed buffers that
the captured kernel reads.
"""

import importlib.util
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# The kernel's hardcoded ring depth; replayssm_buffer_len must equal this.
UCACHE_W_RING = 16

_KMOD: Any = None

# Reserved null page id (mirrors vllm/v1/attention/backends/utils.py).
NULL_BLOCK_ID = 0

# Padded-row sentinel: the kernel retires the whole CTA at entry for rows
# with state index < 0 (pad-skip), so padded rows cost ~nothing instead of a
# full T-step verify. Requires kernel commit 455b0f6+ (_exit_cta_if_neg);
# rows with index 0 still run the legal P=0 verify against null page 0.
UCACHE_PAD_ROW_ID = -1


def ucache_flush_min(max_spec_len: int) -> int:
    """Lazy flush threshold: flush when [P, P+T) would overflow W_RING.

    Identical cadence to the Triton cursors' early-flush predicate
    ``(write_pos + 2*max_spec_len) > logical_L`` for buffer_len == W_RING.
    """
    return UCACHE_W_RING - max_spec_len + 1


def load_ucache_kernel_module(strided_qkv: bool = True):
    """Load the flush-kernel module exactly once.

    ``SGLANG_GDN_WY_STRIDED_QKV`` is read by the module at import time, so it
    must be set *before* the import executes. ``VLLM_GDN_UCACHE_MODULE`` (an
    absolute path to gdn_decode_bf16_wy_ucache_flush.py) loads the file
    directly via importlib — the module is import-self-contained, so this
    avoids putting a whole flashinfer fork on sys.path and shadowing the
    installed flashinfer package.
    """
    global _KMOD
    if _KMOD is not None:
        return _KMOD
    os.environ.setdefault(
        "SGLANG_GDN_WY_STRIDED_QKV", "1" if strided_qkv else "0"
    )
    # Default kernel dtypes: fp16 SSM-state checkpoint + fp16 u/k rings with
    # bf16 IO (inputs). Read by the module at import time; setdefault keeps
    # explicit user overrides (set both to bf16 for bf16 state/rings). The
    # vLLM-side pool allocation defaults match (see
    # gated_delta_net_replayssm_spec_state_dtype).
    os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", "fp16")
    os.environ.setdefault("GDN_UCACHE_RING_DTYPE", "fp16")
    path = os.environ.get("VLLM_GDN_UCACHE_MODULE")
    if path:
        spec = importlib.util.spec_from_file_location(
            "gdn_ucache_flush_kernel", path
        )
        assert spec is not None and spec.loader is not None, (
            f"cannot load ucache kernel module from {path!r}"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        from flashinfer.gdn_kernels import (  # type: ignore[import-not-found]
            gdn_decode_bf16_wy_ucache_flush as mod,
        )
    assert mod.W_RING == UCACHE_W_RING, (
        f"ucache kernel W_RING={mod.W_RING} != expected {UCACHE_W_RING}"
    )
    logger.info_once(
        "GDN spec backend flashinfer_ucache: loaded kernel module from %s "
        "(strided_qkv=%s)",
        path or "flashinfer.gdn_kernels (PYTHONPATH)",
        strided_qkv,
    )
    _KMOD = mod
    return mod


# A_log/dt_bias must reach the kernel wrappers as STABLE tensor objects:
# their fp32->bf16 cast cache is keyed by object identity, and calling
# .detach() per call creates a fresh object every time -> cache miss ->
# one cast kernel per tensor per layer per step (measured ~2 x 1.9us x 36
# layers/step). Detach ONCE per parameter here; the parameter object owns
# the entry lifetime.
_DETACHED: dict = {}


def _detached(t: torch.Tensor) -> torch.Tensor:
    d = _DETACHED.get(id(t))
    if d is None:
        d = t.detach()
        _DETACHED[id(t)] = d
    return d


_PAD_BUFS: dict = {}


def _pad_scratch(key, shape, dtype, device, fill=0):
    buf = _PAD_BUFS.get(key)
    if buf is None or buf.shape != torch.Size(shape):
        buf = torch.full(shape, fill, dtype=dtype, device=device)
        _PAD_BUFS[key] = buf
    return buf


def gdn_ucache_spec_verify(
    *,
    mixed_qkv_spec: torch.Tensor,  # [total_spec, 2*HK*K + HV*V] packed q|k|v
    a: torch.Tensor,  # [num_tokens, HV] (chunk view; rows [0:total_spec] used)
    b: torch.Tensor,  # [num_tokens, HV]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,  # [blocks, HV, V, K] bf16 checkpoint pool
    u_cache: torch.Tensor,  # [blocks, HV, 16, V] bf16
    k_cache: torch.Tensor,  # [blocks, HK, 16, K] bf16
    g_cache: torch.Tensor,  # [blocks, HV, 16] f32
    hist_len: torch.Tensor,  # [B] int32, request-keyed (gathered by builder)
    state_indices: torch.Tensor,  # [B] int32 physical block per request
    num_spec_decodes: int,
    max_spec_len: int,
    num_k_heads: int,  # per-rank HK
    head_k_dim: int,
    head_v_dim: int,
    scale: float,
    strided_qkv: bool = True,
    pad_to: int | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused verify+flush kernel; returns [total_spec, HV, V] view.

    ``pad_to``: pad the request batch to this fixed size with null-page rows.
    The strided-mode JIT cache key includes B, so without padding every new
    batch size seen on an eager (mixed prefill+decode ramp) step costs a
    full CuTeDSL compile (~60s) during batch-admission ramps. One bucket =>
    one cubin. Pad rows: hist=0, state index = -1 (pad-skip: their CTAs
    retire at kernel entry; outputs discarded).
    """
    mod = load_ucache_kernel_module(strided_qkv)
    B, T = num_spec_decodes, max_spec_len
    total_spec = mixed_qkv_spec.shape[0]
    assert total_spec == B * T, (
        f"flashinfer_ucache spec backend requires uniform {T}-token verify "
        f"rows; got total_spec={total_spec}, num_spec_decodes={B}. "
        "Non-uniform spec batches cannot fall back per-step (incompatible "
        "persistent ring formats)."
    )
    HK, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = (mixed_qkv_spec.shape[1] - 2 * HK * K) // V

    # Whole-fp16 kernel mode (GDN_UCACHE_IO_DTYPE=fp16): cast the PACKED qkv
    # slice ONCE, before slicing — the q/k/v sub-views then share one token
    # stride, keeping the wrapper's strided static-descriptor mode engaged
    # (REQUIRED: block-strided vLLM state pools reject the contiguous-dynamic
    # path). bf16 -> fp16 is exact in range. Output is cast back below.
    _io = getattr(mod, "IO_TORCH", torch.bfloat16)
    _out_dtype = mixed_qkv_spec.dtype
    if _io is not mixed_qkv_spec.dtype:
        mixed_qkv_spec = mixed_qkv_spec.to(_io)

    # Caller-provided destination ([real_total, HV, V], the layer's
    # core_attn_out slice): kernels STG directly into it (wrapper
    # alias) and the layer skips its slice-assign (the 2.7us
    # DtoD per layer). Only usable when the padded batch exactly covers it
    # and dtypes match (fp16-IO mode casts, so it falls back there).
    _dest = output
    if (
        _dest is not None
        and (_io is not _out_dtype or _dest.dtype is not _out_dtype
             or not _dest.is_contiguous())
    ):
        _dest = None

    real_total = total_spec
    if pad_to is not None and B < pad_to:
        dev = mixed_qkv_spec.device
        qkv_dim = mixed_qkv_spec.shape[1]
        def _claimable(t, rows, cols):
            need = (t.storage_offset() + (rows - 1) * t.stride(0)
                    + (cols - 1) * t.stride(1) + 1)
            return (t.untyped_storage().nbytes() // t.element_size()) >= need

        if (
            os.environ.get("VLLM_GDN_UCACHE_ZEROCOPY_PAD") == "1"
            and _claimable(mixed_qkv_spec, pad_to * T, qkv_dim)
            and _claimable(a, pad_to * T, HV)
            and _claimable(b, pad_to * T, HV)
        ):
            # ZERO-COPY padding: claim bucket-shaped views over the REAL
            # (possibly strided) tensors via as_strided. Rows beyond
            # total_spec are phantom — their addresses land in descriptors
            # but are NEVER dereferenced: the pad sentinel (-1) in the
            # bucket-length index buffers retires those CTAs at kernel
            # entry before any load/TMA/cp.async. Kills the per-layer
            # direct_copy (packed qkv) + 2 DtoD (a/b) in drain-bucket
            # steps. Env-gated: out-of-bounds-by-contract addressing.
            mixed_qkv_spec = mixed_qkv_spec.as_strided(
                (pad_to * T, qkv_dim), mixed_qkv_spec.stride()
            )
            a = a.as_strided((pad_to * T, HV), a.stride())
            b = b.as_strided((pad_to * T, HV), b.stride())
        else:
            packed = _pad_scratch(
                ("qkv", dev), (pad_to * T, qkv_dim), mixed_qkv_spec.dtype, dev
            )
            packed[:total_spec].copy_(mixed_qkv_spec)
            a_buf = _pad_scratch(("a", dev), (pad_to * T, HV), a.dtype, dev)
            b_buf = _pad_scratch(("b", dev), (pad_to * T, HV), b.dtype, dev)
            a_buf[:total_spec].copy_(a[:total_spec])
            b_buf[:total_spec].copy_(b[:total_spec])
            mixed_qkv_spec, a, b = packed, a_buf, b_buf
        # hist/idx tensors arriving PRE-PADDED at bucket length (the
        # builder fills pad rows each step) are used as-is — no per-layer
        # copies/fills. Shorter tensors take the legacy staging path.
        if hist_len.shape[0] >= pad_to:
            hist_len = hist_len[:pad_to]
        else:
            hist_buf = _pad_scratch(("hist", dev), (pad_to,), torch.int32, dev)
            hist_buf[:B].copy_(hist_len)
            hist_buf[B:].fill_(0)
            hist_len = hist_buf
        if state_indices.shape[0] >= pad_to:
            state_indices = state_indices[:pad_to]
        else:
            idx_buf = _pad_scratch(
                ("idx", dev), (pad_to,), state_indices.dtype, dev,
                fill=UCACHE_PAD_ROW_ID
            )
            idx_buf[:B].copy_(state_indices)
            idx_buf[B:].fill_(UCACHE_PAD_ROW_ID)
            state_indices = idx_buf
        B = pad_to
        total_spec = B * T

    qkv = mixed_qkv_spec.view(B, T, -1)
    # Last-dim slices of one packed row share token stride -> the wrapper's
    # opt-in strided path reads them zero-copy (SGLANG_GDN_WY_STRIDED_QKV=1).
    q = qkv[..., : HK * K].unflatten(-1, (HK, K))
    k = qkv[..., HK * K : 2 * HK * K].unflatten(-1, (HK, K))
    v = qkv[..., 2 * HK * K :].unflatten(-1, (HV, V))
    # a/b are chunk() views with token stride 2*HV; rows [0:total_spec] is the
    # exact window the Triton spec kernel reads (parity). reshape is a pure
    # VIEW here (regular token stride), and the wrappers' strided-a/b mode
    # (ab_t_stride, kernel repo) reads it directly -- the two per-layer
    # .contiguous() copies (~3us x 36 layers/step) are gone. Wrappers fall
    # back to staging automatically if the stride pattern is irregular.
    a_spec = a[:total_spec].reshape(B, T, HV)
    b_spec = b[:total_spec].reshape(B, T, HV)

    if _io is not a_spec.dtype:
        a_spec = a_spec.to(_io)
        b_spec = b_spec.to(_io)

    _fused_out = None
    if _dest is not None and _dest.shape[0] == B * T:
        _fused_out = _dest.view(B, T, HV, V)
    out = mod.gated_delta_rule_mtp_ucache_flush(
        # nn.Parameters carry requires_grad=True, which DLPack refuses to
        # export; detach() shares storage so the wrapper's identity-keyed
        # bf16 cast cache still hits.
        A_log=_detached(A_log),
        a=a_spec,
        dt_bias=_detached(dt_bias),
        q=q,
        k=k,
        v=v,
        b=b_spec,
        initial_state_source=ssm_state,
        initial_state_indices=state_indices,
        k_cache=k_cache,
        u_cache=u_cache,
        g_cache=g_cache,
        hist_len=hist_len,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        # caller-destination alias when available, else zero-copy view
        output=_fused_out,
        flush_min=ucache_flush_min(T),
        restart_hist_on_flush=False,  # builder-owned restart (see module doc)
    )
    if out.dtype is not _out_dtype:
        out = out.to(_out_dtype)
    return out.reshape(B * T, HV, V)[:real_total]


@triton.jit
def _commit_gdn_ucache_hist_kernel(
    hist_ptr,  # [num_gpu_blocks] int32, block-keyed
    num_accepted_ptr,  # [n_rows] int32 (previous step's acceptance)
    sbi_ptr,  # base ptr of spec_state_indices_tensor[:, 0]
    first_decode_ptr,  # [n_rows] int8 (dummy when HAS_RESET == False)
    n_rows,
    sbi_stride,
    FLUSH_MIN: tl.constexpr,
    HAS_RESET: tl.constexpr,
    NULL_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    m = offs < n_rows
    blk = tl.load(sbi_ptr + offs * sbi_stride, mask=m, other=NULL_BLOCK).to(
        tl.int64
    )
    valid = m & (blk > NULL_BLOCK)
    old = tl.load(hist_ptr + blk, mask=valid, other=0).to(tl.int32)
    acc = tl.load(num_accepted_ptr + offs, mask=valid, other=0).to(tl.int32)
    # A row whose previous verify launched with hist >= FLUSH_MIN flushed
    # in-kernel (every layer): its ring restarted at [0, T), so committed
    # history restarts at 0 before adding the accepted count.
    new = tl.where(old >= FLUSH_MIN, 0, old) + acc
    if HAS_RESET:
        fd = tl.load(first_decode_ptr + offs, mask=valid, other=0).to(tl.int32)
        new = tl.where(fd != 0, 0, new)  # prefill->decode / block-recycle
    tl.store(hist_ptr + blk, new, mask=valid)


def commit_gdn_ucache_hist(
    hist_len: torch.Tensor,  # [num_gpu_blocks] int32, block-keyed
    num_accepted_tokens: torch.Tensor,  # [n_rows] int32
    state_indices: torch.Tensor,  # [n_rows] view of block ids (may be strided)
    first_decode: torch.Tensor | None,  # [n_rows] int8 or None
    *,
    flush_min: int,
) -> None:
    """Eager (outside-capture) hist commit; mirrors commit_gdn_replayssm_spec.

    Invariant afterwards: hist <= (flush_min - 1) + T == W_RING, the kernel's
    legal [0, W_RING] range.
    """
    n_rows = state_indices.shape[0]
    if n_rows == 0:
        return
    BLOCK = max(triton.next_power_of_2(n_rows), 16)
    _commit_gdn_ucache_hist_kernel[(1,)](
        hist_len,
        num_accepted_tokens,
        state_indices,
        first_decode if first_decode is not None else hist_len,
        n_rows,
        state_indices.stride(0),
        FLUSH_MIN=flush_min,
        HAS_RESET=first_decode is not None,
        NULL_BLOCK=NULL_BLOCK_ID,
        BLOCK=BLOCK,
    )
