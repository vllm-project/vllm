# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W0 probe — FlashInfer direct-XQA capture/replay with device-side ragged
updates.

Validates the central premise of the SM120/121 XQA adaptive-mismatch PR:
a CUDA-graph-captured XQA kernel specialized for a fixed max draft width
(q_seq_len=8) must consume in-place-updated persistent device buffers —
q_cu_seq_lens (ragged offsets) and the packed draft mask — across different
ragged layouts, without host synchronization.

Standalone FlashInfer only (no vLLM import): if this fails, the issue is a
FlashInfer-library reproducer.

Geometry:
  - 4 request slots: 3 real + 1 zero-length padded request
  - 8 physical query/output rows, 6 active
  - capture  [2,2,2,0]  q_cu=[0,2,4,6,6]
  - replay1  [3,2,1,0]  q_cu=[0,3,5,6,6]
  - replay2  [1,3,2,0]  q_cu=[0,1,4,6,6]  (2nd layout)

Configs:
  A: bf16 KV, no sliding window        — clean SDPA oracle, JIT compiles fresh
  B: fp8-e4m3 KV, window_left=39       — production flags (page16/hd128/ratio4/
                                         SWA/spec8), warm JIT cache; window is
                                         non-binding (all seq lens <= 24)

Acceptance: capture+replay succeed; replay follows updated device
offsets+mask; active rows match eager XQA (bitwise) and SDPA (dtype tol);
addresses stable; no D2H during replay.

SM120/121-gated: skips on non-SM120/121 hardware (CI has no SM120 GPUs).
"""

import math
import shutil

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and (
        torch.cuda.get_device_capability()[0] == 12
    )),
    reason="Requires SM120/121 (RTX 5090 / B200-class) hardware",
)

try:
    import flashinfer
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

# FlashInfer JIT compilation spawns a subprocess that needs `ninja` in the
# system PATH (not just the venv PATH).
HAS_NINJA = shutil.which("ninja") is not None

pytestmark = pytestmark and pytest.mark.skipif(
    not HAS_FLASHINFER, reason="flashinfer not installed"
) and pytest.mark.skipif(
    not HAS_NINJA, reason="ninja not in PATH (required for FlashInfer JIT)"
)

DEV = torch.device("cuda:0")
PAGE = 16
PAGES_PER_SEQ = 2
NPAGES = 8
HDIM = 128
QH = 32
KVH = 8
GRP = QH // KVH
MAXQ = 8  # fixed configured bound: num_speculative_steps + num_bonus_tokens
NREQ = 4
NPHYS = 8  # physical query/output rows (2 graph-padding rows)
CTX = [10, 16, 22, 8]  # per-request context before the draft block
SM_SCALE = 1.0 / math.sqrt(HDIM)
WORKSPACE_BYTES = 32 * 1024 * 1024

GEOMS = [
    ("capture", [2, 2, 2, 0]),
    ("replay1", [3, 2, 1, 0]),
    ("replay2", [1, 3, 2, 0]),
]


def cum_of(lens):
    c, out = 0, [0]
    for L in lens:
        c += L
        out.append(c)
    return out


def seq_lens_of(lens):
    return [CTX[i] + L for i, L in enumerate(lens)]


def pack_causal_mask(lens, max_q, nphys):
    """Pack a causal ragged draft mask (uint16, 32-bit-aligned little-endian)."""
    num_packed = (max_q + 31) // 32
    m = torch.zeros(nphys, num_packed * 2, dtype=torch.uint16)
    r = 0
    for L in lens:
        for j in range(L):
            for w in range(num_packed):
                bits = 0
                for c in range(w * 32, min(j + 1, (w + 1) * 32)):
                    bits |= 1 << (c - w * 32)
                m[r, 2 * w] = bits & 0xFFFF
                m[r, 2 * w + 1] = (bits >> 16) & 0xFFFF
            r += 1
    assert r <= nphys
    return m


def eager_xqa(kv_dtype, window_left, q, qcu_list, S_list, mask_cpu, k, v, bt):
    """Reference run on freshly-allocated clones (outside any graph)."""
    qcu = torch.tensor(qcu_list, dtype=torch.int32, device=DEV)
    sl = torch.tensor(S_list, dtype=torch.uint32, device=DEV)
    mask = mask_cpu.to(DEV)
    out = torch.empty_like(q)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEV)
    xqa_batch_decode_with_kv_cache(
        query=q.clone(),
        kv_cache=(k.clone(), v.clone()),
        workspace_buffer=ws,
        block_tables=bt.clone(),
        seq_lens=sl,
        max_seq_len=PAGES_PER_SEQ * PAGE,
        bmm1_scale=SM_SCALE,
        bmm2_scale=1.0,
        window_left=window_left,
        out=out,
        kv_layout="NHD",
        q_len_per_req=MAXQ,
        mask=mask,
        q_cu_seq_lens=qcu,
    )
    return out


def sdpa_ref(q, k, v, bt, S_list, lens, scale):
    """Per-request full-causal SDPA (draft block at the end of the sequence)."""
    outs = {}
    cum = cum_of(lens)
    for i, L in enumerate(lens):
        if L == 0:
            continue
        S = S_list[i]
        pages = bt[i].to(torch.long)
        Ki = k[pages].reshape(-1, KVH, HDIM)[:S].to(torch.bfloat16)
        Vi = v[pages].reshape(-1, KVH, HDIM)[:S].to(torch.bfloat16)
        Ki = Ki.repeat_interleave(GRP, dim=1).unsqueeze(0).transpose(1, 2)
        Vi = Vi.repeat_interleave(GRP, dim=1).unsqueeze(0).transpose(1, 2)
        Qi = q[cum[i] : cum[i + 1]].unsqueeze(0).transpose(1, 2)
        causal = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=q.device)
        )[S - L:]
        o = F.scaled_dot_product_attention(
            Qi, Ki, Vi, attn_mask=causal, scale=scale
        )
        outs[i] = o.transpose(1, 2).squeeze(0)
    return outs


def active_diff(a, b, lens):
    """Max abs diff over active rows only, plus per-request maxes."""
    cum = cum_of(lens)
    per_req, mx = [], 0.0
    for i, L in enumerate(lens):
        if L == 0:
            per_req.append(None)
            continue
        d = (
            (a[cum[i] : cum[i + 1]].float() - b[cum[i] : cum[i + 1]].float())
            .abs()
            .max()
            .item()
        )
        per_req.append(round(d, 6))
        mx = max(mx, d)
    return mx, per_req


def padding_report(out, lens):
    cum = cum_of(lens)
    tail = out[cum[-1] :]
    if torch.isnan(tail).all().item():
        return "untouched (still NaN sentinel)"
    return (
        f"WRITTEN by kernel (non-NaN content present), "
        f"max|val|={tail.float().abs().max().item():.4f}"
    )


def _run_config(kv_dtype, window_left):
    """Run one config; return the result dict."""
    # --- persistent buffers (captured by address) ---
    q = torch.randn(NPHYS, QH, HDIM, dtype=torch.bfloat16, device=DEV)
    out = torch.empty_like(q)
    qcu = torch.zeros(NREQ + 1, dtype=torch.int32, device=DEV)
    mask = torch.zeros(
        NPHYS, ((MAXQ + 31) // 32) * 2, dtype=torch.uint16, device=DEV
    )
    sl = torch.zeros(NREQ, dtype=torch.uint32, device=DEV)
    bt = torch.arange(
        NREQ * PAGES_PER_SEQ, dtype=torch.int32, device=DEV
    ).reshape(NREQ, PAGES_PER_SEQ)
    k = (torch.randn(NPAGES, PAGE, KVH, HDIM, device=DEV) * 0.5).to(kv_dtype)
    v = (torch.randn(NPAGES, PAGE, KVH, HDIM, device=DEV) * 0.5).to(kv_dtype)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEV)
    tensors = {
        "q": q, "out": out, "q_cu_seq_lens": qcu, "mask": mask,
        "seq_lens": sl, "block_tables": bt, "k": k, "v": v, "ws": ws,
    }
    addrs0 = {n: t.data_ptr() for n, t in tensors.items()}

    q_payloads = {}
    for name, lens in GEOMS:
        q_payloads[name] = (
            torch.randn(NPHYS, QH, HDIM, device=DEV) * 0.5
        ).to(torch.bfloat16)

    def load_geometry(name, lens):
        qcu.copy_(torch.tensor(cum_of(lens), dtype=torch.int32))
        sl.copy_(torch.tensor(seq_lens_of(lens), dtype=torch.uint32))
        mask.copy_(pack_causal_mask(lens, MAXQ, NPHYS))
        q.copy_(q_payloads[name])
        out.fill_(float("nan"))

    def graph_call():
        xqa_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k, v),
            workspace_buffer=ws,
            block_tables=bt,
            seq_lens=sl,
            max_seq_len=PAGES_PER_SEQ * PAGE,
            bmm1_scale=SM_SCALE,
            bmm2_scale=1.0,
            window_left=window_left,
            out=out,
            kv_layout="NHD",
            q_len_per_req=MAXQ,
            mask=mask,
            q_cu_seq_lens=qcu,
        )

    cfg = {"kv_dtype": str(kv_dtype), "window_left": window_left}

    # --- warm-up + eager references (also triggers any JIT build) ---
    load_geometry("capture", GEOMS[0][1])
    eager_outs, sdpa_outs = {}, {}
    for name, lens in GEOMS:
        load_geometry(name, lens)
        eager_outs[name] = eager_xqa(
            kv_dtype, window_left, q, cum_of(lens), seq_lens_of(lens),
            pack_causal_mask(lens, MAXQ, NPHYS), k, v, bt,
        )
        sdpa_outs[name] = {
            "1/sqrt(d)": sdpa_ref(
                q, k, v, bt, seq_lens_of(lens), lens, SM_SCALE
            ),
            "1.0": sdpa_ref(q, k, v, bt, seq_lens_of(lens), lens, 1.0),
        }
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            graph_call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # --- capture ---
    load_geometry("capture", GEOMS[0][1])
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_call()
    g.replay()
    torch.cuda.synchronize()
    cfg["capture"] = "ok"
    addrs1 = {n: t.data_ptr() for n, t in tensors.items()}
    cap = {}
    cap["addr_stable_after_capture"] = addrs0 == addrs1
    d_eager, _ = active_diff(out, eager_outs["capture"], GEOMS[0][1])
    cap["vs_eager_maxdiff"] = round(d_eager, 8)
    cap["padding_rows"] = padding_report(out, GEOMS[0][1])
    cfg["capture_stage"] = cap

    # --- replays on updated buffers ---
    for name, lens in GEOMS[1:]:
        load_geometry(name, lens)
        g.replay()
        torch.cuda.synchronize()
        r = {}
        d_eager, per_req = active_diff(out, eager_outs[name], lens)
        r["vs_eager_maxdiff"] = round(d_eager, 8)
        r["vs_eager_per_req"] = per_req
        best_conv, best_d = None, float("inf")
        for conv, refs in sdpa_outs[name].items():
            mx = 0.0
            cum = cum_of(lens)
            for i, L in enumerate(lens):
                if L == 0:
                    continue
                mx = max(
                    mx,
                    (out[cum[i] : cum[i + 1]].float() - refs[i].float())
                    .abs()
                    .max()
                    .item(),
                )
            r[f"vs_sdpa{conv}_maxdiff"] = round(mx, 6)
            if mx < best_d:
                best_conv, best_d = conv, mx
        r["sdpa_matching_convention"] = best_conv
        r["padding_rows"] = padding_report(out, lens)
        cfg[name] = r

    addrs2 = {n: t.data_ptr() for n, t in tensors.items()}
    cfg["addr_stable_after_replays"] = addrs0 == addrs1 == addrs2

    # --- no-D2H / no-sync during bare replay: sync debug mode ---
    try:
        torch.cuda.set_sync_debug_mode(2)
        g.replay()
        torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        cfg["sync_debug_replay"] = "pass (no torch-visible sync)"
    except Exception:
        torch.cuda.set_sync_debug_mode(0)
        cfg["sync_debug_replay"] = "FAIL"

    # --- no-D2H / no-sync during replay: profiler event scan ---
    def scan_profile(replays):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(replays):
                g.replay()
        bad, kernels = [], []
        for evt in prof.key_averages():
            key = evt.key
            if any(
                t in key
                for t in (
                    "Memcpy DtoH", "cudaStreamSynchronize",
                    "cudaDeviceSynchronize", "cudaEventSynchronize",
                    "cudaMemcpy", "cudaStreamWaitEvent",
                )
            ):
                bad.append(f"{key} x{evt.count}")
            if "xqa" in key.lower() or "mha" in key.lower():
                kernels.append(key)
        return bad, kernels

    try:
        bad0, _ = scan_profile(0)
        bad5, kernels = scan_profile(5)
        new_bad = [b for b in bad5 if b not in bad0]
        torch.cuda.synchronize()
        cfg["profiler_replay"] = {
            "sync_or_d2h_events": new_bad,
            "baseline_profiler_events": bad0,
            "kernel_names": kernels,
        }
    except Exception as e:
        cfg["profiler_replay"] = f"FAILED: {e!r}"

    return cfg


def _check_gates(cfg, sdpa_tol):
    """Check the acceptance gates for one config."""
    prof_res = cfg.get("profiler_replay", {})
    no_bad_prof_events = isinstance(prof_res, dict) and (
        prof_res.get("sync_or_d2h_events") == []
    )
    return (
        cfg.get("capture") == "ok"
        and cfg.get("capture_stage", {}).get("addr_stable_after_capture")
        and cfg.get("addr_stable_after_replays")
        and cfg.get("capture_stage", {}).get("vs_eager_maxdiff", 1) <= 1e-6
        and all(
            cfg.get(r, {}).get("vs_eager_maxdiff", 1) <= 1e-6
            for r in ("replay1", "replay2")
        )
        and all(
            cfg.get(r, {}).get(
                f"vs_sdpa{cfg[r]['sdpa_matching_convention']}_maxdiff", 1
            )
            <= sdpa_tol
            for r in ("replay1", "replay2")
        )
        and "pass" in str(cfg.get("sync_debug_replay", ""))
        and no_bad_prof_events
    )


def test_w0_xqa_probe_bf16_kv():
    """W0 probe, config A: bf16 KV, no sliding window."""
    torch.manual_seed(0)
    cfg = _run_config(torch.bfloat16, -1)
    assert _check_gates(cfg, 3e-2), f"Config A gates failed: {cfg}"


def test_w0_xqa_probe_fp8_kv_swa():
    """W0 probe, config B: fp8-e4m3 KV, window_left=39 (production flags)."""
    torch.manual_seed(0)
    cfg = _run_config(torch.float8_e4m3fn, 39)
    assert _check_gates(cfg, 2e-1), f"Config B gates failed: {cfg}"
