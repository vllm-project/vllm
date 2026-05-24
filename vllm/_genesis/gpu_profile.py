# SPDX-License-Identifier: Apache-2.0
"""GPU spec database + per-patch recommendation engine for Genesis.

Determines patch enablement recommendations based on the actual GPU
detected at boot, since memory-bandwidth-bound vs compute-bound regimes
have OPPOSITE optimization preferences.

Why this exists
---------------
P40 (TQ k8v4 GQA grouping kernel) was empirically NOT SIGNIFICANT on
2× RTX A5000 (768 GB/s HBM, 4 MB L2) despite PR #40792's +27% claim
on Hopper. Root cause: A5000 saturates HBM ceiling regardless of how
efficiently we use it; the grouping benefit can only manifest when
HBM is NOT the bottleneck (compute-bound regime).

Different cards in the Genesis user community have different ceilings:

| GPU                          | HBM/GDDR  | Bandwidth GB/s | L2     | SM   | Regime        |
|------------------------------|-----------|----------------|--------|------|---------------|
| A5000                        | GDDR6     | 768            |  4 MB  | 64   | bandwidth     |
| RTX 3090                     | GDDR6X    | 936            |  6 MB  | 82   | bandwidth     |
| RTX 4090                     | GDDR6X    | 1008           | 72 MB  | 128  | mixed/compute |
| RTX 5090                     | GDDR7     | 1792           | 88 MB  | 170  | compute       |
| RTX A6000                    | GDDR6     | 768            |  6 MB  | 84   | bandwidth     |
| L40 / L40S                   | GDDR6     | 864            | 96 MB  | 142  | mixed/compute |
| RTX PRO 4000 Blackwell 24G   | GDDR7     | 672            | 24 MB  | 70   | mixed         |
| RTX PRO 4500 Blackwell 32G   | GDDR7     | 896            | 32 MB  | 84   | mixed         |
| RTX PRO 5000 Blackwell 48G   | GDDR7     | 1344           | 64 MB  | 110  | compute       |
| RTX PRO 6000 Blackwell 96G   | GDDR7     | 1792           | 88 MB  | 188  | compute       |
| RTX PRO 6000 Blackwell Max-Q | GDDR7     | 1792           | 88 MB  | 188  | compute       |
| A100                         | HBM2e     | 2039           | 40 MB  | 108  | compute       |
| H100                         | HBM3      | 3350           | 50 MB  | 132  | compute       |
| B200                         | HBM3e     | 8000           | 80 MB+ | 208  | compute       |

The L2 column is decisive for KV-cache-locality optimizations (P40, P67):
- < 8 MB:  KV blocks always evicted between query heads → grouping cannot
  amortize K loads → P40 is no-op (A5000, 3090, A6000)
- 24-50 MB: partial KV resident → P40 modest gain
- 72 MB+:   most KV blocks resident → P40 substantial gain (4090, L40,
  H100, B200)

Design notes
------------
- We hard-code specs from NVIDIA datasheets (most stable spec axis).
- We do NOT auto-enable patches without env opt-in. We only PRINT
  recommendations at boot. User must still pass GENESIS_ENABLE_PXX=1.
- Override via env: GENESIS_AUTO_ENABLE_RECOMMENDED=1 enables auto-on
  for patches with `auto_enable_on_recommend = True` in their wiring.

Adding a new GPU
----------------
Append to GPU_SPECS dict. Match the substring(s) NVIDIA reports via
`torch.cuda.get_device_properties().name`.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.gpu_profile")


# Static spec database. Numbers from NVIDIA public datasheets.
# Match keys are lowercase substrings of `device.name`.
GPU_SPECS: dict[str, dict] = {
    # Ampere consumer
    "rtx 3060": {"bandwidth_gb_s": 360, "l2_mb": 3, "sm": 28, "cc": (8, 6),
                 "regime": "bandwidth", "name_canonical": "RTX 3060"},
    "rtx 3070": {"bandwidth_gb_s": 448, "l2_mb": 4, "sm": 46, "cc": (8, 6),
                 "regime": "bandwidth", "name_canonical": "RTX 3070"},
    "rtx 3080": {"bandwidth_gb_s": 760, "l2_mb": 5, "sm": 68, "cc": (8, 6),
                 "regime": "bandwidth", "name_canonical": "RTX 3080"},
    "rtx 3090": {"bandwidth_gb_s": 936, "l2_mb": 6, "sm": 82, "cc": (8, 6),
                 "regime": "bandwidth", "name_canonical": "RTX 3090"},
    # Ampere workstation/server
    "rtx a4000": {"bandwidth_gb_s": 448, "l2_mb": 4, "sm": 48, "cc": (8, 6),
                  "regime": "bandwidth", "name_canonical": "RTX A4000"},
    "rtx a5000": {"bandwidth_gb_s": 768, "l2_mb": 4, "sm": 64, "cc": (8, 6),
                  "regime": "bandwidth", "name_canonical": "RTX A5000"},
    "rtx a6000": {"bandwidth_gb_s": 768, "l2_mb": 6, "sm": 84, "cc": (8, 6),
                  "regime": "bandwidth", "name_canonical": "RTX A6000"},
    "a100": {"bandwidth_gb_s": 2039, "l2_mb": 40, "sm": 108, "cc": (8, 0),
             "regime": "compute", "name_canonical": "A100 (SXM 80GB)"},
    # Ada (RTX 40-series + L40)
    "rtx 4060": {"bandwidth_gb_s": 272, "l2_mb": 24, "sm": 24, "cc": (8, 9),
                 "regime": "mixed", "name_canonical": "RTX 4060"},
    "rtx 4060 ti": {"bandwidth_gb_s": 288, "l2_mb": 32, "sm": 34, "cc": (8, 9),
                    "regime": "mixed", "name_canonical": "RTX 4060 Ti"},
    "rtx 4070": {"bandwidth_gb_s": 504, "l2_mb": 36, "sm": 46, "cc": (8, 9),
                 "regime": "mixed", "name_canonical": "RTX 4070"},
    "rtx 4070 super": {"bandwidth_gb_s": 504, "l2_mb": 48, "sm": 56, "cc": (8, 9),
                       "regime": "mixed", "name_canonical": "RTX 4070 SUPER"},
    "rtx 4070 ti": {"bandwidth_gb_s": 672, "l2_mb": 48, "sm": 60, "cc": (8, 9),
                    "regime": "mixed", "name_canonical": "RTX 4070 Ti"},
    "rtx 4070 ti super": {"bandwidth_gb_s": 672, "l2_mb": 48, "sm": 66, "cc": (8, 9),
                          "regime": "mixed", "name_canonical": "RTX 4070 Ti SUPER"},
    "rtx 4080": {"bandwidth_gb_s": 716, "l2_mb": 64, "sm": 76, "cc": (8, 9),
                 "regime": "mixed", "name_canonical": "RTX 4080"},
    "rtx 4080 super": {"bandwidth_gb_s": 736, "l2_mb": 64, "sm": 80, "cc": (8, 9),
                       "regime": "mixed", "name_canonical": "RTX 4080 SUPER"},
    "rtx 4090": {"bandwidth_gb_s": 1008, "l2_mb": 72, "sm": 128, "cc": (8, 9),
                 "regime": "mixed", "name_canonical": "RTX 4090"},
    "l40": {"bandwidth_gb_s": 864, "l2_mb": 96, "sm": 142, "cc": (8, 9),
            "regime": "mixed", "name_canonical": "L40 / L40S"},
    "rtx 6000 ada": {"bandwidth_gb_s": 960, "l2_mb": 96, "sm": 142, "cc": (8, 9),
                     "regime": "mixed", "name_canonical": "RTX 6000 Ada"},
    # Hopper
    "h100": {"bandwidth_gb_s": 3350, "l2_mb": 50, "sm": 132, "cc": (9, 0),
             "regime": "compute", "name_canonical": "H100"},
    "h20": {"bandwidth_gb_s": 4000, "l2_mb": 60, "sm": 78, "cc": (9, 0),
            "regime": "compute", "name_canonical": "H20"},
    "h200": {"bandwidth_gb_s": 4800, "l2_mb": 50, "sm": 132, "cc": (9, 0),
             "regime": "compute", "name_canonical": "H200"},
    # Blackwell consumer (specs as of 2026 announcements; Issue #20 — sm_120)
    "rtx 5060": {"bandwidth_gb_s": 448, "l2_mb": 32, "sm": 36, "cc": (12, 0),
                 "regime": "mixed", "name_canonical": "RTX 5060"},
    "rtx 5060 ti": {"bandwidth_gb_s": 448, "l2_mb": 32, "sm": 36, "cc": (12, 0),
                    "regime": "mixed", "name_canonical": "RTX 5060 Ti"},
    "rtx 5070": {"bandwidth_gb_s": 672, "l2_mb": 48, "sm": 50, "cc": (12, 0),
                 "regime": "mixed", "name_canonical": "RTX 5070"},
    "rtx 5070 ti": {"bandwidth_gb_s": 896, "l2_mb": 64, "sm": 70, "cc": (12, 0),
                    "regime": "mixed", "name_canonical": "RTX 5070 Ti"},
    "rtx 5080": {"bandwidth_gb_s": 960, "l2_mb": 64, "sm": 84, "cc": (12, 0),
                 "regime": "mixed", "name_canonical": "RTX 5080"},
    "rtx 5090": {"bandwidth_gb_s": 1792, "l2_mb": 88, "sm": 170, "cc": (12, 0),
                 "regime": "compute", "name_canonical": "RTX 5090"},
    "rtx pro 6000 blackwell": {"bandwidth_gb_s": 1792, "l2_mb": 88, "sm": 188, "cc": (12, 0),
                               "regime": "compute", "name_canonical": "RTX PRO 6000 Blackwell"},
    # Blackwell PRO workstation lineup (announced 2025, datasheet specs).
    # The "Max-Q" variants share the same silicon at lower TDP — same regime,
    # bandwidth typically unchanged (memory speed is not gated by power limit).
    "rtx pro 4000 blackwell": {"bandwidth_gb_s": 672, "l2_mb": 24, "sm": 70, "cc": (12, 0),
                               "regime": "mixed", "name_canonical": "RTX PRO 4000 Blackwell (24 GB)"},
    "rtx pro 4500 blackwell": {"bandwidth_gb_s": 896, "l2_mb": 32, "sm": 84, "cc": (12, 0),
                               "regime": "mixed", "name_canonical": "RTX PRO 4500 Blackwell (32 GB)"},
    "rtx pro 5000 blackwell": {"bandwidth_gb_s": 1344, "l2_mb": 64, "sm": 110, "cc": (12, 0),
                               "regime": "compute", "name_canonical": "RTX PRO 5000 Blackwell (48 GB)"},
    # Note: Sander mentioned a "5000 72GB" SKU. NVIDIA's published Blackwell PRO
    # 5000 lineup as of 2026 datasheet review is 48 GB only — if a 72 GB SKU
    # ships later, append here with the correct memory bus / bandwidth.
    "rtx pro 6000 blackwell max-q": {"bandwidth_gb_s": 1792, "l2_mb": 88, "sm": 188, "cc": (12, 0),
                                     "regime": "compute", "name_canonical": "RTX PRO 6000 Blackwell Max-Q (96 GB)"},
    "b200": {"bandwidth_gb_s": 8000, "l2_mb": 80, "sm": 208, "cc": (10, 0),
             "regime": "compute", "name_canonical": "B200"},
}


# Per-patch recommendation rules.
# Conditions evaluated against the detected GPU spec dict.
# Rules express WHEN this patch is expected to deliver measurable gains.
PATCH_RECOMMENDATIONS: dict[str, dict] = {
    "P40": {
        "title": "TQ k8v4 GQA grouping kernel (vllm#40792)",
        "env": "GENESIS_ENABLE_P40",
        "predicate": lambda gpu: (
            gpu["regime"] in ("compute", "mixed") and gpu["l2_mb"] >= 24
        ),
        "evidence": "Author measured +27% on H100. Empirically NS on A5000 "
                    "(p=0.28). Cache-locality benefit needs L2 >= 24 MB.",
        "expected_gain": "+5-15% (mixed regime), +15-30% (compute regime)",
        "expected_cost": "negligible (load-time wrap)",
    },
    "P67": {
        "title": "Multi-query verify kernel for spec-decode K+1",
        "env": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "predicate": lambda gpu: True,  # Always recommend; useful on all
        "evidence": "+32% TPS on 35B-A3B-FP8 spec-decode K=3 verify (Genesis "
                    "internal benchmark, all GPU classes tested).",
        "expected_gain": "+25-35%",
        "expected_cost": "low",
    },
    "P82": {
        "title": "SGLang-style acceptance threshold OR-clause",
        "env": "GENESIS_ENABLE_P82",
        "predicate": lambda gpu: True,
        "evidence": "Cross-rig confirmed: +12% on A5000 FP8, +10.5% on 3090 INT4.",
        "expected_gain": "+8-12%",
        "expected_cost": "none (sampling layer only)",
    },
    "PN8": {
        "title": "MTP/draft online-quant propagation (vllm#40849)",
        "env": "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT",
        "predicate": lambda gpu: True,  # VRAM savings useful everywhere
        "evidence": "Verified ~1 GiB VRAM saved per GPU on 35B-A3B-FP8 + MTP K=3. "
                    "Use freed VRAM for higher gpu-mem-util or longer ctx.",
        "expected_gain": "0% TPS, but ~1-2 GiB total VRAM headroom",
        "expected_cost": "none",
    },
    "P83+P84+P85": {
        "title": "Prefix-cache cake-and-eat patches (vllm#38182)",
        "env": "GENESIS_ENABLE_P83=1 GENESIS_ENABLE_P84=1 GENESIS_ENABLE_P85=1 GENESIS_P84_HASH_BLOCK_SIZE=16",
        "predicate": lambda gpu: False,  # Empirically failed on current vllm pin
        "evidence": "Tested 4-arm A/B 2026-04-29: -29% TPS regression even with full "
                    "stack including root-cause P84. Possible vllm cache machinery "
                    "fixed-overhead per-step. WAIT for v0.20.2 pin bump.",
        "expected_gain": "(currently negative)",
        "expected_cost": "high — full PROD swap needed to retest",
    },
    # PN63 — fp8_e5m2 advisory for Blackwell consumer (sm 12.0)
    # Source: noonghunna club-3090#51 (apnar's RTX 5090 bench 2026-05-04)
    # measured fp8_e4m3 + 96K ctx LOSES 2-6% TPS vs fp8_e5m2 + 48K. NOT a
    # configuration patch — this is operator advisory only. Suggests staying
    # with fp8_e5m2 until vLLM Blackwell e4m3 path matures (currently the
    # FlashInfer + e4m3 codepath was newly added and underTuned for sm_120).
    "PN63_kv_e5m2_blackwell_consumer": {
        "title": "fp8_e5m2 KV-dtype advisory for consumer Blackwell (PN63)",
        "env": "kv-cache-dtype fp8_e5m2 (CLI flag, not env)",
        "predicate": lambda gpu: (
            gpu.get("cc") == (12, 0)  # consumer Blackwell only
        ),
        "evidence": "club-3090#51 (apnar 2026-05-04): fp8_e4m3 + 96K ctx "
                    "regressed 2-6% TPS vs fp8_e5m2 + 48K on RTX 5090. vLLM "
                    "Blackwell e4m3 codepath is newly added and undertuned. "
                    "Advisory only — operator passes via --kv-cache-dtype.",
        "expected_gain": "+2-6% TPS staying with e5m2 until pin update",
        "expected_cost": "none (CLI flag preserves current behavior)",
    },
    # P100 — FlashInfer FULL CG for spec-decode, recommended on consumer Blackwell
    # Source: apnar club-3090#51 — PIECEWISE downgrade observed on RTX 5090
    # because is_blackwell() returned False (Issue #20, since fixed but not yet
    # released) AND P100 was not auto-recommended. With Issue #20 fix landed,
    # the gate now passes; this rule surfaces P100 to the operator on sm_120.
    "P100_blackwell_consumer_recommend": {
        "title": "P100 FlashInfer FULL CG for spec-decode (recommend on Blackwell consumer)",
        "env": "GENESIS_ENABLE_P100=1 (when using FlashInfer + spec-decode)",
        "predicate": lambda gpu: (
            gpu.get("cc") == (12, 0)  # consumer Blackwell only
        ),
        "evidence": "club-3090#51 boot log: `CUDAGraphMode.FULL_AND_PIECEWISE "
                    "is not supported with spec-decode for FlashInferBackend; "
                    "setting cudagraph_mode=PIECEWISE`. P100 backports vllm#41127 "
                    "to route uniform query_len>1 batches through prefill "
                    "wrapper in cudagraph mode. Bit-identical, +5-10% TPS expected.",
        "expected_gain": "+5-10% TPS (when FlashInfer is the chosen backend)",
        "expected_cost": "low (single env flag, idempotent text-patch)",
    },
}


def detect_current_gpu() -> Optional[dict]:
    """Inspect torch.cuda for the current GPU and look up its spec.

    Returns spec dict augmented with detected fields, or None if no CUDA.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    try:
        props = torch.cuda.get_device_properties(0)
    except Exception as e:
        log.debug("get_device_properties failed: %s", e)
        return None

    name = (props.name or "").lower()
    cc = (props.major, props.minor)

    spec_match = None
    for key, spec in GPU_SPECS.items():
        if key in name:
            spec_match = dict(spec)
            spec_match["name_detected"] = props.name
            spec_match["match_key"] = key
            break

    if spec_match is None:
        # Unknown GPU — return minimal info so caller can degrade gracefully
        return {
            "name_canonical": "unknown",
            "name_detected": props.name,
            "match_key": None,
            "bandwidth_gb_s": None,
            "l2_mb": None,
            "sm": props.multi_processor_count,
            "cc": cc,
            "regime": "unknown",
        }

    return spec_match


def recommend_patches() -> list[dict]:
    """Return list of recommendation entries for the current GPU."""
    gpu = detect_current_gpu()
    if gpu is None:
        return []

    out = []
    for patch_id, rec in PATCH_RECOMMENDATIONS.items():
        try:
            recommended = bool(rec["predicate"](gpu))
        except (KeyError, TypeError) as e:
            # Spec missing field referenced by predicate — degrade
            log.debug("predicate eval failed for %s: %s", patch_id, e)
            recommended = False

        env_set = os.environ.get(rec["env"].split()[0].split("=")[0])
        currently_on = env_set == "1"

        out.append({
            "patch_id": patch_id,
            "title": rec["title"],
            "env": rec["env"],
            "recommended": recommended,
            "currently_on": currently_on,
            "evidence": rec["evidence"],
            "expected_gain": rec["expected_gain"],
            "expected_cost": rec["expected_cost"],
            "gpu_regime": gpu["regime"],
        })

    return out


def print_recommendations(stream=None) -> str:
    """Render recommendations as human-readable text. Returns the text and
    optionally writes to stream."""
    gpu = detect_current_gpu()
    if gpu is None:
        msg = "[Genesis GPU profile] no CUDA device — skipping recommendations"
        if stream:
            print(msg, file=stream)
        return msg

    lines = []
    lines.append("=" * 78)
    lines.append(f"[Genesis GPU profile] detected: {gpu['name_detected']}")
    lines.append(f"  canonical: {gpu['name_canonical']}  cc: {gpu['cc']}  "
                 f"SM: {gpu['sm']}  L2: {gpu['l2_mb']} MB  "
                 f"BW: {gpu['bandwidth_gb_s']} GB/s  regime: {gpu['regime']}")
    lines.append("")
    lines.append("Per-patch recommendations:")
    lines.append("-" * 78)

    recs = recommend_patches()
    for r in recs:
        symbol = ("ON" if r["currently_on"] else
                  "REC" if r["recommended"] else "OFF")
        lines.append(f"  [{symbol:3s}] {r['patch_id']:18s} {r['title']}")
        lines.append(f"          gain: {r['expected_gain']}")
        if r["recommended"] and not r["currently_on"]:
            lines.append(f"          → recommend: export {r['env']}=1")
        elif not r["recommended"] and r["currently_on"]:
            lines.append(f"          ⚠️  enabled but NOT recommended on this GPU "
                         f"(may regress or no-op)")
        elif not r["recommended"] and not r["currently_on"]:
            lines.append(f"          (correctly skipped on this regime)")
        lines.append(f"          why: {r['evidence']}")

    lines.append("=" * 78)
    text = "\n".join(lines)
    if stream:
        print(text, file=stream)
    return text


if __name__ == "__main__":
    import sys
    print_recommendations(stream=sys.stdout)
