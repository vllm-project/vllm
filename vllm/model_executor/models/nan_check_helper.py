# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph-safe NaN/Inf detection for DeepSeek v2/v3.

Per-layer checks write NaN and Inf counts to GPU tensors
(no .item(), no sync, no graph break).
compute_logits runs outside torch.compile and reads the counts.
Only tracks REAL token NaN (non-padded). Padding NaN is ignored entirely.
Writes to stderr + Lustre file.
"""

import datetime
import os
import sys

import torch

_nan_reported = False
_inf_reported = False
_log_fh = None
_per_layer_checks_enabled = os.environ.get("VLLM_NAN_CHECK", "1") == "1"

# Count tensors: shape (num_layers, 4)
# column 0 = input (before layernorm), column 1 = pre_attn (after layernorm),
# column 2 = attn, column 3 = moe
_nan_counts: torch.Tensor | None = None
_inf_counts: torch.Tensor | None = None

# Attention detail tensors: shape (num_layers, 23)
# Outer MLA wrapper (mla.py):
#   0=fused_qkv_a_proj_output (full), 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
# Inner MLAAttention (mla_attention.py):
#   6=after_kv_cache_update, 7=after_W_UK_bmm, 8=after_fwd_mqa, 9=after_v_up
#   10=after_fwd_mha, 11=kv_cache, 12=mqa_q_pre_fwd, 13=lse_post_fwd_mqa
#   14=mha_q, 15=mha_kv_c_normed, 16=mha_k_pe
#   17=kv_c_normed_decode_real (bf16, seq_lens-filtered)
#   18=kv_cache_fp8_nan (FP8 NaN via uint8 bit pattern check)
#   19=after_kv_b_proj_prefill (new tokens), 20=after_kv_b_proj_context_chunk
#   21=kv_c_pre_norm (before RMSNorm), 22=k_pe_pre_rope (before RoPE)
_attn_detail: torch.Tensor | None = None
_inf_attn_detail: torch.Tensor | None = None

# Real-only fwd_mqa NaN flag per layer (1 = real NaN detected).
# Used by stash_if_nan to gate on real NaN only.
_fwd_mqa_real_nan: torch.Tensor | None = None

# Pre-allocated layer index scalars on GPU — avoids CPU→GPU transfer
# during CUDA graph capture (torch.tensor(int, device=cuda) is illegal).
_layer_idx_gpu: torch.Tensor | None = None

# Per-layer max abs of hidden_states input (column 0). Shape (num_layers,) float32.
_hidden_maxabs: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _nan_counts, _inf_counts, _attn_detail, _inf_attn_detail
    global _fwd_mqa_real_nan, _layer_idx_gpu, _hidden_maxabs
    _ensure_kv_write_counts(num_layers, device)
    if _nan_counts is None or _nan_counts.shape[0] < num_layers:
        _nan_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _inf_counts is None or _inf_counts.shape[0] < num_layers:
        _inf_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _attn_detail is None or _attn_detail.shape[0] < num_layers:
        _attn_detail = torch.zeros(num_layers, 23, dtype=torch.int64, device=device)
    if _inf_attn_detail is None or _inf_attn_detail.shape[0] < num_layers:
        _inf_attn_detail = torch.zeros(num_layers, 23, dtype=torch.int64, device=device)
    if _fwd_mqa_real_nan is None or _fwd_mqa_real_nan.shape[0] < num_layers:
        _fwd_mqa_real_nan = torch.zeros(num_layers, dtype=torch.int64, device=device)
    if _layer_idx_gpu is None or _layer_idx_gpu.shape[0] < num_layers:
        _layer_idx_gpu = torch.arange(num_layers, dtype=torch.int64, device=device)
    if _hidden_maxabs is None or _hidden_maxabs.shape[0] < num_layers:
        _hidden_maxabs = torch.zeros(num_layers, dtype=torch.float32, device=device)


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    )


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    Only counts NaN/Inf in real tokens ([:_last_num_actual_toks]).
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    global _nan_counts, _inf_counts
    if not _per_layer_checks_enabled:
        return
    if _nan_counts is None:
        return
    if _is_fp8(tensor.dtype):
        return
    n = _last_num_actual_toks
    if n > 0 and n < tensor.shape[0]:
        tensor = tensor[:n]
    _nan_counts[layer_idx, stage_col] = tensor.isnan().sum()
    _inf_counts[layer_idx, stage_col] = tensor.isinf().sum()
    if stage_col == 0 and _hidden_maxabs is not None:
        _hidden_maxabs[layer_idx] = tensor.abs().max()


def mark_attn(
    tensor: torch.Tensor,
    stage_col: int,
    layer_idx: int,
    *,
    seq_lens: torch.Tensor | None = None,
    skip_filter: bool = False,
) -> None:
    """Called inside MLA attention forward for detailed tracking.
    Only counts NaN/Inf in real tokens.
    - seq_lens: pass for decode [B,...] tensors — masks with seq_lens > 0
    - skip_filter: pass True for kv_cache (different shape, no filtering)
    - otherwise: auto-slices by _last_num_actual_toks for [N,...] tensors
    """
    global _attn_detail, _inf_attn_detail
    if not _per_layer_checks_enabled:
        return
    if _attn_detail is None:
        return
    if _is_fp8(tensor.dtype):
        return
    if skip_filter:
        nan_count = tensor.isnan().sum()
        inf_count = tensor.isinf().sum()
    elif seq_lens is not None:
        # Broadcast [B] mask to tensor shape: [B] → [B, 1, 1, ...]
        real_mask = (seq_lens > 0).view(-1, *([1] * (tensor.dim() - 1)))
        nan_count = (tensor.isnan() & real_mask).sum()
        inf_count = (tensor.isinf() & real_mask).sum()
    else:
        n = _last_num_actual_toks
        if n > 0 and n < tensor.shape[0]:
            tensor = tensor[:n]
        nan_count = tensor.isnan().sum()
        inf_count = tensor.isinf().sum()
    _attn_detail[layer_idx, stage_col] = nan_count
    _inf_attn_detail[layer_idx, stage_col] = inf_count


def mark_fp8_nan(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Count FP8 NaN values by checking uint8 bit patterns.

    FP8 e4m3fn NaN: 0x7F (positive) and 0xFF (negative).
    Only runs on FP8 tensors; no-op for other dtypes.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    if not _per_layer_checks_enabled:
        return
    if _attn_detail is None:
        return
    if not _is_fp8(tensor.dtype):
        return
    raw = tensor.view(torch.uint8)
    # e4m3fn NaN: 0x7F or 0xFF (S111_1111)
    nan_count = ((raw == 0x7F) | (raw == 0xFF)).sum()
    _attn_detail[layer_idx, stage_col] = nan_count


_kv_write_nan_counts: torch.Tensor | None = None

# Per-layer int32 flags written by concat_and_cache_mla_kernel via atomicOr.
# Shape: (num_layers, 2) — [layer, 0] = bit flags, [layer, 1] = min token_idx.
# Bit layout: bit0=FP8 NaN in kv_c, bit1=FP8 NaN in k_pe,
#             bit2=Inf in kv_c source, bit3=Inf in k_pe source,
#             bit4=NaN in kv_c source, bit5=NaN in k_pe source.
# token_idx initialized to INT_MAX; kernel uses atomicMin to capture first hit.
_kv_kernel_nan_flags: torch.Tensor | None = None


def _ensure_kv_write_counts(num_layers: int, device: torch.device) -> None:
    global _kv_write_nan_counts, _kv_kernel_nan_flags
    if _kv_write_nan_counts is None or _kv_write_nan_counts.shape[0] < num_layers:
        _kv_write_nan_counts = torch.zeros(
            num_layers, dtype=torch.int64, device=device)
    if _kv_kernel_nan_flags is None or _kv_kernel_nan_flags.shape[0] < num_layers:
        _kv_kernel_nan_flags = torch.zeros(
            num_layers, 2, dtype=torch.int32, device=device)
        # Column 1 = min token_idx, init to INT_MAX
        _kv_kernel_nan_flags[:, 1] = 0x7FFFFFFF


def get_kernel_nan_flag(layer_idx: int) -> "torch.Tensor | None":
    """Return a 2-element int32 view for the kernel's nan_flag.

    Element [0] = bit flags (atomicOr), [1] = min token_idx (atomicMin).
    Returns None if checks are disabled or flags not yet allocated.
    """
    if not _per_layer_checks_enabled:
        return None
    if _kv_kernel_nan_flags is None:
        return None
    return _kv_kernel_nan_flags[layer_idx]


def mark_kv_cache_write(
    kv_cache: torch.Tensor,
    slots: torch.Tensor | None,
    layer_idx: int,
) -> None:
    """Check freshly-written KV cache slots for NaN after cache update.

    Only checks the slots that were just written (not the full cache).
    GPU-only ops — no .item(), no sync, no graph break.
    Counts are read by report_if_nan outside the compiled region.
    """
    global _kv_write_nan_counts
    if not _per_layer_checks_enabled:
        return
    if slots is None:
        return
    if _kv_write_nan_counts is None:
        return
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    written = flat[slots]
    if _is_fp8(written.dtype):
        raw = written.view(torch.uint8)
        n = ((raw == 0x7F) | (raw == 0xFF)).sum()
    else:
        n = torch.isnan(written).sum()
    _kv_write_nan_counts[layer_idx] += n


_kv_cache_checked = False


def check_kv_caches(layers) -> None:
    """Check page 0 (null block) of each layer's KV cache for NaN.

    Called outside compiled/CUDA graph region (from forward()).
    Only reports once.
    """
    global _kv_cache_checked
    if not _per_layer_checks_enabled or _kv_cache_checked:
        return
    for layer in layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        kv = getattr(attn, 'kv_cache', None)
        if kv is None or not isinstance(kv, torch.Tensor) or kv.numel() == 0:
            continue
        page0 = kv[0]  # [num_heads, head_dim] or [block_size, head_dim]
        if _is_fp8(page0.dtype):
            raw = page0.view(torch.uint8)
            n = ((raw == 0x7F) | (raw == 0xFF)).sum().item()
        else:
            n = torch.isnan(page0).sum().item()
        if n > 0:
            _kv_cache_checked = True
            layer_idx = getattr(layer, 'layer_idx', '?')
            f = _get_log()
            msg = (f"[KV_CACHE_PAGE0_NAN] layer={layer_idx} "
                   f"nan_count={n} page0_shape={list(page0.shape)} "
                   f"dtype={kv.dtype}\n")
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)
            return


def mark_kv_stale_fp8_nan(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    layer_idx: int,
) -> None:
    """Check stale (unused) slots in each sequence's last KV cache block
    for FP8 NaN bit patterns.

    Only scans the padding region of the last block per real sequence.
    All ops stay on GPU — no .item(), no sync, no graph break.

    kv_cache: [num_blocks, block_size, head_size] (fp8)
    block_table: [B, max_num_blocks_per_seq] (int32)
    seq_lens: [B] (int32)
    """
    if _attn_detail is None:
        return
    if not _is_fp8(kv_cache.dtype):
        return

    # Only check real sequences (seq_lens > 0)
    real_mask = seq_lens > 0
    real_seq_lens = seq_lens[real_mask]
    real_block_table = block_table[real_mask]

    if real_seq_lens.numel() == 0:
        _attn_detail[layer_idx, 18] = 0
        return

    # Last logical block index per sequence
    last_logical = (real_seq_lens - 1) // block_size
    # Physical block index
    last_physical = real_block_table.gather(
        1, last_logical.unsqueeze(1).to(torch.int64)
    ).squeeze(1)
    # Number of used slots in last block
    used = real_seq_lens % block_size
    # Handle exact multiples: used=0 means full block, no stale slots
    full_block_mask = used == 0

    # Gather last blocks: [num_real_seqs, block_size, head_size]
    last_blocks = kv_cache[last_physical.long()]

    # Build a mask for stale slots: [num_real_seqs, block_size]
    slot_idx = torch.arange(block_size, device=kv_cache.device).unsqueeze(0)
    # stale = slot >= used (but not for full blocks)
    stale_mask = (slot_idx >= used.unsqueeze(1)) & ~full_block_mask.unsqueeze(1)

    # Check FP8 NaN in stale slots only
    raw = last_blocks.view(last_blocks.shape[0], block_size, -1)
    raw_u8 = raw.view(torch.uint8)
    is_nan = (raw_u8 == 0x7F) | (raw_u8 == 0xFF)
    # Mask to stale slots: [num_real_seqs, block_size, head_size_bytes]
    stale_3d = stale_mask.unsqueeze(2).expand_as(is_nan)
    nan_count = (is_nan & stale_3d).sum()

    _attn_detail[layer_idx, 18] = nan_count


def mark_fwd_mqa_real(
    attn_out: torch.Tensor, layer_idx: int, seq_lens: torch.Tensor
) -> None:
    """Record whether fwd_mqa produced NaN for any REAL token.

    Called right after mark_attn(attn_out, 8, layer_idx).
    Uses seq_lens > 0 to mask out padding tokens.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    if not _per_layer_checks_enabled:
        return
    if _fwd_mqa_real_nan is None:
        return
    if _is_fp8(attn_out.dtype):
        return
    # real_mask: [B] bool, attn_out: [B, H, D]
    real_mask = seq_lens > 0
    real_nan = (attn_out.isnan() & real_mask.view(-1, 1, 1)).any().to(torch.int64)
    _fwd_mqa_real_nan[layer_idx] = real_nan


_saved_batch_info: dict | None = None
_last_num_actual_toks: int = 0


def report_batch_info(
    layer_idx: int,
    num_actual_toks: int,
    padded_size: int,
    num_decode_tokens: int,
    num_mha_tokens: int,
) -> None:
    """Capture batch sizing info (logged later only when NaN detected)."""
    global _saved_batch_info, _last_num_actual_toks
    _last_num_actual_toks = num_actual_toks
    _saved_batch_info = {
        "layer_idx": layer_idx,
        "num_actual_toks": num_actual_toks,
        "padded_size": padded_size,
        "num_decode_tokens": num_decode_tokens,
        "num_mha_tokens": num_mha_tokens,
    }


def _emit_batch_info(tag: str) -> None:
    if _saved_batch_info is None:
        return
    b = _saved_batch_info
    f = _get_log()
    msg = (
        f"[BATCH_{tag}] layer={b['layer_idx']} "
        f"num_actual_toks={b['num_actual_toks']} "
        f"padded_size={b['padded_size']} "
        f"num_decode_tokens={b['num_decode_tokens']} "
        f"num_mha_tokens={b['num_mha_tokens']}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)


_saved_scales: dict | None = None


def report_scales(
    layer_idx: int,
    scale: float,
    q_scale: float | None,
    k_scale: float | None,
    bmm1_scale: float | None,
    bmm2_scale: float | None,
) -> None:
    """Capture scale factors (logged later only when NaN is detected)."""
    global _saved_scales
    _saved_scales = {
        "layer_idx": layer_idx,
        "scale": scale,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
    }


def _emit_scales(tag: str) -> None:
    if _saved_scales is None:
        return
    s = _saved_scales
    f = _get_log()
    msg = (
        f"[SCALES_{tag}] layer={s['layer_idx']} "
        f"scale={s['scale']} q_scale={s['q_scale']} k_scale={s['k_scale']} "
        f"bmm1_scale={s['bmm1_scale']} bmm2_scale={s['bmm2_scale']}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)


def _get_log():
    global _log_fh
    if _log_fh is None:
        log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
        os.makedirs(log_dir, exist_ok=True)
        hostname = os.environ.get("HOSTNAME", "unknown")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{log_dir}/{hostname}_gpu{gpu}_{ts}.log"
        _log_fh = open(path, "a")  # noqa: SIM115
        _log_fh.write(f"=== NaN/Inf check started {datetime.datetime.now()} ===\n")
        _log_fh.flush()
    return _log_fh



def _zero_all():
    if _nan_counts is not None:
        _nan_counts.zero_()
    if _inf_counts is not None:
        _inf_counts.zero_()
    if _attn_detail is not None:
        _attn_detail.zero_()
    if _inf_attn_detail is not None:
        _inf_attn_detail.zero_()
    if _fwd_mqa_real_nan is not None:
        _fwd_mqa_real_nan.zero_()
    if _hidden_maxabs is not None:
        _hidden_maxabs.zero_()
    if _kv_write_nan_counts is not None:
        _kv_write_nan_counts.zero_()
    if _kv_kernel_nan_flags is not None:
        _kv_kernel_nan_flags[:, 0] = 0
        _kv_kernel_nan_flags[:, 1] = 0x7FFFFFFF


def _emit_report(
    tag: str,
    hidden_states: torch.Tensor,
    layer_counts: torch.Tensor,
    attn_counts: torch.Tensor | None,
    total_count: int,
    num_actual_toks: int,
) -> None:
    """Emit a [NAN_FIRST] or [INF_FIRST] report block (real tokens only)."""
    h = hidden_states.shape[-1]
    f = _get_log()

    msg = (
        f"[{tag}] at_compute_logits (ALL COUNTS ARE REAL-TOKEN ONLY): "
        f"count={total_count} ({total_count // h} real rows) "
        f"num_actual_toks={num_actual_toks} "
        f"shape={list(hidden_states.shape)} dtype={hidden_states.dtype}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)

    for layer_idx in range(layer_counts.shape[0]):
        input_c = layer_counts[layer_idx, 0].item()
        pre_c = layer_counts[layer_idx, 1].item()
        attn_c = layer_counts[layer_idx, 2].item()
        moe_c = layer_counts[layer_idx, 3].item()
        if input_c + pre_c + attn_c + moe_c == 0:
            continue

        msg = (
            f"[{tag}] layer={layer_idx} "
            f"input={input_c} post_ln={pre_c} attn={attn_c} moe={moe_c}\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)

        if attn_counts is not None and attn_c > 0:
            ad = attn_counts[layer_idx]
            msg = (
                f"[{tag}] layer={layer_idx} attn_detail: "
                f"qkv_proj={ad[0].item()} q_norm={ad[1].item()} "
                f"kv_norm={ad[2].item()} rope={ad[3].item()} "
                f"mla_attn={ad[4].item()} o_proj={ad[5].item()}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)

            msg = (
                f"[{tag}] layer={layer_idx} mla_inner: "
                f"kv_cache_upd={ad[6].item()} W_UK_bmm={ad[7].item()} "
                f"fwd_mqa={ad[8].item()} v_up_proj={ad[9].item()} "
                f"fwd_mha={ad[10].item()} kv_cache_bf16={ad[11].item()} "
                f"mqa_q_pre={ad[12].item()} lse={ad[13].item()} "
                f"mha_q={ad[14].item()} mha_kv_c={ad[15].item()} "
                f"mha_k_pe={ad[16].item()} "
                f"kv_c_normed_real={ad[17].item()} "
                f"kv_cache_fp8_nan={ad[18].item()} "
                f"kv_b_proj_prefill={ad[19].item()} "
                f"kv_b_proj_ctx_chunk={ad[20].item()}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)


# ---------------------------------------------------------------------------
# Single shared stash buffer for NaN repro dump.
# One buffer set, sized on first use.  Gates on REAL NaN only
# (using seq_lens > 0 mask on fwd_mqa output).
# Writes ONLY at the first layer that produces real NaN.
# All ops are GPU-side — no .item(), no sync, no graph break.
# ---------------------------------------------------------------------------
_stash_bufs: dict[int, list[torch.Tensor]] = {}  # keyed by batch_size
_stash_captured: dict[int, torch.Tensor] = {}  # keyed by batch_size
_stash_layer_idx: dict[int, torch.Tensor] = {}  # keyed by batch_size
_stashed_metadata: dict = {}  # persistent refs (block_table, seq_lens, etc.)
_stashed_kv_per_layer: dict[int, dict[int, torch.Tensor]] = {}  # [bkey][layer_idx]

# ---------------------------------------------------------------------------
# Prefill (MHA) stash — no CUDA graphs, so just store refs directly.
# ---------------------------------------------------------------------------
_prefill_stash: dict | None = None  # set once at first NaN layer


def stash_if_nan(
    layer_idx: int,
    q_input,
    q_nope_post_bmm,
    q_pe,
    mqa_q,
    kv_cache,
    block_table,
    seq_lens,
    num_actual_toks: int,
    max_seq_len: int = 0,
    qk_nope_head_dim: int = 0,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    block_size: int = 0,
) -> None:
    """Called AFTER mark_attn and mark_fwd_mqa_real for fwd_mqa.
    Writes to one shared buffer only at the first layer where fwd_mqa
    produced NaN for a REAL token (seq_len > 0).

    Uses masked copy_() (GPU-side, in-place, no graph break).
    Buffer is keyed by batch_size since each CUDA graph batch size
    is compiled separately (dynamic=False).
    """
    if not _per_layer_checks_enabled:
        return
    if _nan_reported:
        return
    B = q_input.shape[0]
    bkey = B

    # Allocate buffers on first call for this batch size
    bufs = _stash_bufs.get(bkey)
    if bufs is None:
        bufs = [
            torch.zeros_like(q_input),
            torch.zeros_like(q_nope_post_bmm),
            torch.zeros_like(q_pe),
        ]
        # mqa_q buffer (FP8 post-quant)
        if isinstance(mqa_q, tuple):
            bufs.extend(torch.zeros_like(t) for t in mqa_q)
        else:
            bufs.append(torch.zeros_like(mqa_q))
        _stash_bufs[bkey] = bufs
        _stash_captured[bkey] = torch.zeros(1, dtype=torch.int64, device=q_input.device)
        _stash_layer_idx[bkey] = torch.full(
            (1,), -1, dtype=torch.int64, device=q_input.device
        )

    captured = _stash_captured[bkey]

    # Gate on REAL NaN only (set by mark_fwd_mqa_real just before us)
    has_real_nan = _fwd_mqa_real_nan[layer_idx] > 0
    first_nan = has_real_nan & (captured == 0)

    # Conditional in-place copy: write only at first layer with real NaN
    bufs[0].copy_(torch.where(first_nan, q_input, bufs[0]))
    bufs[1].copy_(torch.where(first_nan, q_nope_post_bmm, bufs[1]))
    bufs[2].copy_(torch.where(first_nan, q_pe, bufs[2]))

    # mqa_q (FP8) — torch.where doesn't support FP8, use view-as-uint8
    if isinstance(mqa_q, tuple):
        for i, t in enumerate(mqa_q):
            idx = 3 + i
            bufs[idx].view(torch.uint8).copy_(
                torch.where(first_nan, t.view(torch.uint8), bufs[idx].view(torch.uint8))
            )
    else:
        bufs[3].view(torch.uint8).copy_(
            torch.where(first_nan, mqa_q.view(torch.uint8), bufs[3].view(torch.uint8))
        )

    # Record which layer was captured (in-place)
    _stash_layer_idx[bkey].copy_(
        torch.where(first_nan, _layer_idx_gpu[layer_idx], _stash_layer_idx[bkey])
    )

    # Block subsequent layers from writing (in-place)
    _stash_captured[bkey].copy_(
        torch.where(has_real_nan, torch.ones_like(captured), captured)
    )

    # Store per-layer kv_cache ref (just a pointer, no copy).
    if bkey not in _stashed_kv_per_layer:
        _stashed_kv_per_layer[bkey] = {}
    _stashed_kv_per_layer[bkey][layer_idx] = kv_cache

    # Store persistent refs (overwritten each layer — cheap, just pointers).
    _stashed_metadata[bkey] = {
        "block_table": block_table,
        "seq_lens": seq_lens,
        "num_actual_toks": num_actual_toks,
        "mqa_q_is_tuple": isinstance(mqa_q, tuple),
        "mqa_q_count": len(mqa_q) if isinstance(mqa_q, tuple) else 1,
        "max_seq_len": max_seq_len,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "block_size": block_size,
    }


def stash_if_nan_prefill(
    layer_idx: int,
    mha_output: torch.Tensor,
    kv_cache: torch.Tensor,
    num_actual_toks: int,
) -> None:
    """Prefill-path stash. No CUDA graphs, so just store refs at first NaN layer.
    Only stores kv_cache ref + layer_idx. Cheap — no copies, no GPU ops.
    """
    global _prefill_stash
    if _nan_reported or _prefill_stash is not None:
        return
    if not mha_output[:num_actual_toks].isnan().any():
        return
    _prefill_stash = {
        "layer_idx": layer_idx,
        "kv_cache": kv_cache,
        "num_actual_toks": num_actual_toks,
    }


def _dump_repro(
    hidden_states: torch.Tensor,
    nan_cpu: torch.Tensor,
    attn_nan_cpu: torch.Tensor | None,
) -> None:
    """Save stashed attention inputs to disk for NaN reproduction."""
    B = hidden_states.shape[0]
    f = _get_log()

    # Find stash buffer matching this batch size
    bufs = _stash_bufs.get(B)
    captured = _stash_captured.get(B)
    if bufs is None or captured is None or captured.item() == 0:
        msg = (
            f"[NAN_REPRO] MISSED DUMP — no stash captured for B={B} "
            f"(available: {list(_stash_bufs.keys())}, "
            f"captured: {[(k, v.item()) for k, v in _stash_captured.items()]})\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
        return

    stash_layer = _stash_layer_idx[B].item()
    meta = _stashed_metadata.get(B, {})

    log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
    hostname = os.environ.get("HOSTNAME", "unknown")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_dir}/{hostname}_gpu{gpu}_{ts}_repro_layer{stash_layer}.pt"

    save_dict = {
        "origin_layer": stash_layer,
        "stash_layer": stash_layer,
        "hidden_states": hidden_states.cpu(),
        "nan_counts": nan_cpu,
        "attn_nan_counts": attn_nan_cpu,
    }
    if _saved_batch_info is not None:
        save_dict["batch_info"] = _saved_batch_info
    if _saved_scales is not None:
        save_dict["scales"] = _saved_scales

    # Pre-quant bf16 tensors from the stash buffer
    prequant_names = ["q_input", "q_nope_post_bmm", "q_pe"]
    for i, name in enumerate(prequant_names):
        save_dict[name] = bufs[i].cpu()

    # FP8 mqa_q from the stash buffer
    nq = meta.get("mqa_q_count", 1)
    if meta.get("mqa_q_is_tuple", False):
        save_dict["mqa_q"] = tuple(bufs[3 + i].cpu() for i in range(nq))
    else:
        save_dict["mqa_q"] = bufs[3].cpu()

    # Save kv_cache from the actual NaN layer (not the last layer)
    kv_per_layer = _stashed_kv_per_layer.get(B, {})
    kv_at_nan_layer = kv_per_layer.get(stash_layer)
    if kv_at_nan_layer is not None:
        save_dict["kv_cache"] = kv_at_nan_layer.cpu()
    else:
        msg = (
            f"[NAN_REPRO] WARNING: no kv_cache for layer {stash_layer} "
            f"(available: {list(kv_per_layer.keys())})\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)

    # Persistent tensor refs from metadata
    for k in ("block_table", "seq_lens"):
        v = meta.get(k)
        if v is not None and isinstance(v, torch.Tensor):
            save_dict[k] = v.cpu()
    for k in (
        "num_actual_toks",
        "mqa_q_is_tuple",
        "mqa_q_count",
        "max_seq_len",
        "qk_nope_head_dim",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "block_size",
    ):
        if k in meta:
            save_dict[k] = meta[k]

    # Include prefill stash if available
    if _prefill_stash is not None:
        ps = _prefill_stash
        save_dict["prefill_stash_layer"] = ps["layer_idx"]
        save_dict["prefill_kv_cache"] = ps["kv_cache"].cpu()
        save_dict["prefill_num_actual_toks"] = ps["num_actual_toks"]

    try:
        torch.save(save_dict, save_path)
        msg = f"[NAN_REPRO] saved to {save_path} (stash_layer={stash_layer})\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
    except Exception as e:
        msg = f"[NAN_REPRO] FAILED to save: {e}\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)


_kv_poison_reported = False


def _emit_kv_poison(
    hidden_states: torch.Tensor,
    nan_cpu: torch.Tensor,
    attn_nan_cpu: torch.Tensor | None,
    attn_inf_cpu: torch.Tensor | None,
    num_actual_toks: int,
) -> None:
    """Emit [KV_POISON_FIRST] when kv_c_normed_real first has NaN or Inf.

    This fires independently of hidden_states NaN — catches the
    initial write of NaN/Inf to the KV cache. Inf matters because
    FP8 e4m3fn has no Inf representation and bf16 Inf may become
    FP8 NaN.
    """
    f = _get_log()

    first_nan_layer = -1
    first_inf_layer = -1
    if attn_nan_cpu is not None:
        poison_layers = (attn_nan_cpu[:, 17] > 0).nonzero(as_tuple=True)[0]
        first_nan_layer = poison_layers[0].item() if len(poison_layers) > 0 else -1
    if attn_inf_cpu is not None:
        poison_layers = (attn_inf_cpu[:, 17] > 0).nonzero(as_tuple=True)[0]
        first_inf_layer = poison_layers[0].item() if len(poison_layers) > 0 else -1

    msg = (
        f"[KV_POISON_FIRST] kv_c_normed has NaN/Inf for real tokens! "
        f"first_nan_layer={first_nan_layer} "
        f"first_inf_layer={first_inf_layer} "
        f"num_actual_toks={num_actual_toks} "
        f"shape={list(hidden_states.shape)}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)

    # Emit full detail for all layers with nonzero kv_c_normed_real NaN or Inf
    num_layers = max(
        attn_nan_cpu.shape[0] if attn_nan_cpu is not None else 0,
        attn_inf_cpu.shape[0] if attn_inf_cpu is not None else 0,
    )
    for layer_idx in range(num_layers):
        ad = attn_nan_cpu[layer_idx] if attn_nan_cpu is not None else None
        ai = attn_inf_cpu[layer_idx] if attn_inf_cpu is not None else None
        kv_c_nan = ad[17].item() if ad is not None else 0
        kv_c_inf = ai[17].item() if ai is not None else 0
        kv_fp8 = ad[18].item() if ad is not None else 0
        fwd_mqa_nan = ad[8].item() if ad is not None else 0
        fwd_mqa_inf = ai[8].item() if ai is not None else 0
        if kv_c_nan + kv_c_inf + kv_fp8 + fwd_mqa_nan + fwd_mqa_inf == 0:
            continue
        nc = nan_cpu[layer_idx] if nan_cpu is not None else None
        msg = (
            f"[KV_POISON_FIRST] layer={layer_idx} "
            f"kv_c_normed_nan={kv_c_nan} "
            f"kv_c_normed_inf={kv_c_inf} "
            f"kv_cache_fp8_nan={kv_fp8} "
            f"fwd_mqa_nan={fwd_mqa_nan} "
            f"fwd_mqa_inf={fwd_mqa_inf} "
            f"W_UK_bmm={ad[7].item() if ad is not None else 0} "
            f"mqa_q_pre={ad[12].item() if ad is not None else 0} "
            f"input={nc[0].item() if nc is not None else '?'} "
            f"kv_norm_nan={ad[2].item() if ad is not None else 0} "
            f"kv_norm_inf={ai[2].item() if ai is not None else 0}\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)

    _emit_scales("KV_POISON")
    _emit_batch_info("KV_POISON")


def report_if_nan(hidden_states: torch.Tensor) -> None:
    """Called from compute_logits (OUTSIDE torch.compile / cudagraph).
    Only reports REAL token NaN/Inf. Padding is ignored.

    Also checks kv_c_normed_real (col 17) independently — this is
    seq_lens-filtered and reliable during CUDA graph replay, unlike
    hidden_states[:_last_num_actual_toks] which only checks 1 token.
    """
    if _nan_counts is None:
        _zero_all()
        return

    n = _last_num_actual_toks
    total = hidden_states.shape[0]

    if n > 0 and n < total:
        real = hidden_states[:n]
    else:
        real = hidden_states
        n = total

    real_has_nan = real.isnan().any().item()
    real_has_inf = real.isinf().any().item()

    # Check kv_c_normed_real (col 17) for NaN OR Inf — seq_lens-filtered,
    # reliable during graph replay. Catches the initial poisoning event
    # even when hidden_states[:1] looks clean.
    # Inf matters because FP8 e4m3fn has no Inf representation — bf16 Inf
    # may become FP8 NaN if __NV_SATFINITE has a hardware bug on Blackwell.
    kv_poison = (
        _attn_detail is not None
        and (
            _attn_detail[:, 17].any().item()
            or (_inf_attn_detail is not None and _inf_attn_detail[:, 17].any().item())
        )
    )

    # Check if any KV cache writes produced NaN
    kv_write_nan = (
        _kv_write_nan_counts is not None
        and _kv_write_nan_counts.any().item()
    )

    # Check kernel-side NaN/Inf flags (set by concat_and_cache_mla_kernel)
    kv_kernel_hit = (
        _kv_kernel_nan_flags is not None
        and _kv_kernel_nan_flags[:, 0].any().item()
    )

    if not (real_has_nan or real_has_inf or kv_poison or kv_write_nan
            or kv_kernel_hit):
        _zero_all()
        return

    # Copy counts to CPU before zeroing
    nan_cpu = _nan_counts.cpu()
    inf_cpu = _inf_counts.cpu()
    attn_nan_cpu = _attn_detail.cpu() if _attn_detail is not None else None
    attn_inf_cpu = _inf_attn_detail.cpu() if _inf_attn_detail is not None else None
    kv_write_cpu = _kv_write_nan_counts.cpu() if _kv_write_nan_counts is not None else None
    kv_kernel_cpu = _kv_kernel_nan_flags.cpu() if _kv_kernel_nan_flags is not None else None
    maxabs_cpu = _hidden_maxabs.cpu() if _hidden_maxabs is not None else None
    _zero_all()

    if kv_write_nan:
        f = _get_log()
        for layer_idx in range(kv_write_cpu.shape[0]):
            c = kv_write_cpu[layer_idx].item()
            if c > 0:
                msg = (f"[KV_CACHE_WRITE_NAN] layer={layer_idx} "
                       f"nan_count={c}\n")
                f.write(msg)
                f.flush()
                print(msg, file=sys.stderr, end="", flush=True)

    if kv_kernel_hit:
        _BIT_NAMES = {
            0: "fp8_nan_kv_c",
            1: "fp8_nan_k_pe",
            2: "inf_src_kv_c",
            3: "inf_src_k_pe",
            4: "nan_src_kv_c",
            5: "nan_src_k_pe",
        }
        f = _get_log()
        for layer_idx in range(kv_kernel_cpu.shape[0]):
            bits = kv_kernel_cpu[layer_idx, 0].item()
            if bits == 0:
                continue
            tok_idx = kv_kernel_cpu[layer_idx, 1].item()
            if tok_idx == 0x7FFFFFFF:
                tok_idx = -1  # no token captured
            flags = [name for bit, name in _BIT_NAMES.items()
                     if bits & (1 << bit)]
            is_padding = (
                "PADDING" if tok_idx >= 0 and tok_idx >= n
                else "REAL" if tok_idx >= 0
                else "?"
            )
            # Include projection stage NaN/Inf counts from attn detail
            ad = attn_nan_cpu[layer_idx] if attn_nan_cpu is not None else None
            ai = attn_inf_cpu[layer_idx] if attn_inf_cpu is not None else None
            kvc_pre_nan = ad[21].item() if ad is not None else 0
            kvc_pre_inf = ai[21].item() if ai is not None else 0
            kvc_post_nan = ad[2].item() if ad is not None else 0
            kvc_post_inf = ai[2].item() if ai is not None else 0
            kpe_pre_nan = ad[22].item() if ad is not None else 0
            kpe_pre_inf = ai[22].item() if ai is not None else 0
            hs_maxabs = maxabs_cpu[layer_idx].item() if maxabs_cpu is not None else 0.0
            fused_qkv_nan = ad[0].item() if ad is not None else 0
            fused_qkv_inf = ai[0].item() if ai is not None else 0
            msg = (f"[KV_KERNEL_NAN] layer={layer_idx} "
                   f"bits=0x{bits:02x} flags={','.join(flags)} "
                   f"first_tok={tok_idx} num_actual={n} "
                   f"tok_type={is_padding} "
                   f"hs_maxabs={hs_maxabs:.4g} "
                   f"fused_qkv_nan={fused_qkv_nan} "
                   f"fused_qkv_inf={fused_qkv_inf} "
                   f"kvc_pre_norm_nan={kvc_pre_nan} "
                   f"kvc_pre_norm_inf={kvc_pre_inf} "
                   f"kvc_post_norm_nan={kvc_post_nan} "
                   f"kvc_post_norm_inf={kvc_post_inf} "
                   f"kpe_pre_rope_nan={kpe_pre_nan} "
                   f"kpe_pre_rope_inf={kpe_pre_inf}\n")
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)

    if kv_poison:
        _emit_kv_poison(hidden_states, nan_cpu, attn_nan_cpu, attn_inf_cpu, n)

    if real_has_nan:
        rc = real.isnan().sum().item()
        _emit_report(
            "NAN_FIRST", hidden_states, nan_cpu, attn_nan_cpu, rc, num_actual_toks=n
        )
        _emit_scales("NAN")
        _emit_batch_info("NAN")
        _dump_repro(hidden_states, nan_cpu, attn_nan_cpu)

    if real_has_inf:
        rc = real.isinf().sum().item()
        _emit_report(
            "INF_FIRST", hidden_states, inf_cpu, attn_inf_cpu, rc, num_actual_toks=n
        )
        _emit_scales("INF")
        _emit_batch_info("INF")
