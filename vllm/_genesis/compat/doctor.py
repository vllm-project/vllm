# SPDX-License-Identifier: Apache-2.0
"""Genesis doctor — single-command unified diagnostic.

Usage:
  python3 -m vllm._genesis.compat.doctor
  python3 -m vllm._genesis.compat.doctor --json
  python3 -m vllm._genesis.compat.doctor --explain PN14

Sections:
  1. Hardware            — GPUs, compute capabilities
  2. Software            — vllm / torch / triton / cuda / driver / python
  3. Model               — currently-loaded model + detected profile
  4. Patches that APPLY  — full registry walk, why each is on / off
  5. Lifecycle audit     — experimental / deprecated / research breakdown
  6. Recommendations     — actionable suggestions

Output is human-readable by default (color-free, copy-paste friendly)
or JSON via `--json` for machine consumers (CI, dashboards).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

log = logging.getLogger("genesis.compat.doctor")


# ─── Sections ─────────────────────────────────────────────────────────────


def _section_hardware() -> dict[str, Any]:
    """Detect GPUs + compute capabilities. Wraps gpu_profile if available."""
    out: dict[str, Any] = {"gpus": [], "errors": []}
    try:
        import torch
        if not torch.cuda.is_available():
            out["errors"].append("torch.cuda.is_available() == False")
            return out
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cc = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            out["gpus"].append({
                "index": i, "name": name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "compute_capability_tuple": list(cc),
                "vram_total_gb": round(props.total_memory / 1e9, 2),
                "multi_processor_count": props.multi_processor_count,
            })
    except Exception as e:
        out["errors"].append(f"torch GPU probe: {e}")

    # Fold in our gpu_profile classification if available (datasheet bw/L2/etc)
    try:
        from vllm._genesis.compat.gpu_profile import (
            detect_gpu_class as _classify,
        )
        if out["gpus"]:
            try:
                out["gpu_class"] = _classify()
            except Exception as e:
                log.debug("gpu_profile.detect_gpu_class failed: %s", e,
                          exc_info=True)
    except Exception as e:
        log.debug("torch CUDA section probe failed: %s", e, exc_info=True)
    return out


def _section_environment() -> dict[str, Any]:
    """Detect host environment quirks that affect Genesis behavior.

    Currently surfaces:
      * WSL2 (Windows Subsystem for Linux 2) host — display overhead +
        DirectX shim eats VRAM, narrows borderline-OOM headroom; some
        kernels (notably P104 L2 persistence) misbehave under WSL paging.
      * Blackwell-class GPU on WSL2 — R6000 Pro 96GB on WSL2 is an
        atypical combo; Sander's planned upgrade target. Warn that NVFP4
        + PN38 FP8 paths assume bare-metal Linux/Windows, not WSL.
      * PCIe lane width (per-GPU) via nvidia-smi when available; warns
        when any GPU is wired below x16 (cuts P2P/host bandwidth and
        affects TQ continuation-prefill perf in TP=2 setups).

    All probes are best-effort and silently no-op when their data source
    is missing (Mac, container without nvidia-smi, etc.).
    """
    import os
    import shutil
    import subprocess

    out: dict[str, Any] = {
        "is_wsl": False,
        "wsl_version": None,
        "pcie_lanes": [],
        "errors": [],
    }

    # WSL2 detection — /proc/version contains "microsoft" or "WSL"
    proc_version_path = "/proc/version"
    if os.path.exists(proc_version_path):
        try:
            with open(proc_version_path, encoding="utf-8") as f:
                content = f.read().lower()
            if "microsoft" in content or "wsl" in content:
                out["is_wsl"] = True
                if "wsl2" in content:
                    out["wsl_version"] = "WSL2"
                elif "wsl" in content:
                    out["wsl_version"] = "WSL1"
        except Exception as e:
            out["errors"].append(f"/proc/version probe: {e}")

    # PCIe lane width per GPU — nvidia-smi --query-gpu=pcie.link.width.current
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        out["pcie_lanes"].append({
                            "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                            "name": parts[1],
                            "gen_current": parts[2],
                            "width_current": parts[3],
                            "gen_max": parts[4],
                            "width_max": parts[5],
                        })
        except Exception as e:
            out["errors"].append(f"nvidia-smi pcie probe: {e}")

    return out


def _section_software() -> dict[str, Any]:
    """Versions of vllm / torch / triton / cuda / driver / python."""
    from vllm._genesis.compat.version_check import detect_versions
    p = detect_versions()
    return {
        "vllm": p.vllm, "vllm_commit": p.vllm_commit,
        "torch": p.torch, "triton": p.triton,
        "cuda_runtime": p.cuda_runtime, "nvidia_driver": p.nvidia_driver,
        "python": p.python,
        "compute_capabilities": [list(c) for c in p.compute_capabilities],
        "errors": list(p.errors),
    }


def _section_model_profile() -> dict[str, Any]:
    """Resolve the model profile via model_detect.get_model_profile."""
    out: dict[str, Any] = {"resolved": False, "errors": []}
    try:
        from vllm._genesis.compat.model_detect import get_model_profile
        profile = get_model_profile()
        out.update(profile)
    except Exception as e:
        out["errors"].append(f"model_detect: {e}")
    return out


def _section_patches() -> dict[str, Any]:
    """Walk every patch in PATCH_REGISTRY and decide apply/skip with reason."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY, should_apply

    decisions = []
    apply_count = 0
    skip_count = 0
    for pid in PATCH_REGISTRY:
        try:
            decision, reason = should_apply(pid)
        except Exception as e:
            decision, reason = False, f"should_apply raised: {e}"
        meta = PATCH_REGISTRY.get(pid, {})
        decisions.append({
            "patch_id": pid,
            "title": meta.get("title", pid),
            "category": meta.get("category", "uncategorized"),
            "decision": "APPLY" if decision else "SKIP",
            "reason": reason,
            "env_flag": meta.get("env_flag", ""),
            "default_on": meta.get("default_on", False),
        })
        if decision:
            apply_count += 1
        else:
            skip_count += 1
    return {
        "total": len(decisions),
        "apply": apply_count,
        "skip": skip_count,
        "decisions": decisions,
    }


def _section_lifecycle() -> dict[str, Any]:
    """Run the lifecycle audit on the registry."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    from vllm._genesis.compat.lifecycle import audit_registry

    entries = audit_registry(PATCH_REGISTRY)
    by_state: dict[str, list[dict]] = {}
    for e in entries:
        by_state.setdefault(e.state, []).append({
            "patch_id": e.patch_id, "note": e.note, "severity": e.severity,
        })
    return {
        "by_state": by_state,
        "total": len(entries),
    }


def _section_validator() -> dict[str, Any]:
    """Run the A3/D2 validator on the live registry + apply set."""
    try:
        from vllm._genesis.dispatcher import (
            validate_registry, validate_apply_plan, get_apply_matrix,
        )
        static = validate_registry()
        applied = {d["patch_id"] for d in get_apply_matrix() if d["applied"]}
        plan = validate_apply_plan(applied) if applied else []
        return {
            "static_issues": [
                {"severity": i.severity, "patch_id": i.patch_id, "message": i.message}
                for i in static
            ],
            "plan_issues": [
                {"severity": i.severity, "patch_id": i.patch_id, "message": i.message}
                for i in plan
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def _section_recommendations(report: dict[str, Any]) -> list[str]:
    """Heuristic operator-actionable suggestions."""
    rec: list[str] = []

    # Validator errors → actionable
    val = report.get("validator", {})
    for issue in val.get("static_issues", []):
        rec.append(
            f"[{issue['severity']}] PATCH_REGISTRY: {issue['patch_id']} — "
            f"{issue['message']}"
        )
    for issue in val.get("plan_issues", []):
        rec.append(
            f"[{issue['severity']}] APPLY plan: {issue['patch_id']} — "
            f"{issue['message']}"
        )

    # Software errors → block deployment
    sw_errs = report.get("software", {}).get("errors", [])
    for e in sw_errs:
        rec.append(f"[ERROR] software detection: {e}")

    # Hardware errors → at least warn
    hw_errs = report.get("hardware", {}).get("errors", [])
    for e in hw_errs:
        rec.append(f"[WARN] hardware detection: {e}")

    # If model not resolved
    model = report.get("model_profile", {})
    if not model.get("resolved"):
        rec.append(
            "[INFO] no model loaded yet (run this command on a live vllm "
            "container for model-aware analysis)"
        )

    # Environment quirks (P1.11 / P2.11)
    env = report.get("environment", {})
    hw = report.get("hardware", {})
    if env.get("is_wsl"):
        rec.append(
            f"[WARN] {env.get('wsl_version','WSL')} detected — extra display "
            "overhead eats VRAM (~200-400 MiB on 24GB cards). On borderline "
            "configs (Cliff 2 single-card 24GB, club-3090 setups) expect "
            "tighter mem-utilization headroom. Verified noonghunna fix: "
            "set --gpu-memory-utilization to 0.85 (default 0.90 may trip)."
        )
        # Blackwell + WSL2 = atypical combo
        for g in hw.get("gpus", []):
            cc = g.get("compute_capability_tuple", [])
            if cc and len(cc) >= 1 and cc[0] >= 12:
                rec.append(
                    f"[WARN] Blackwell-class GPU '{g.get('name','?')}' "
                    f"detected on {env.get('wsl_version','WSL')}. PN38 NVFP4 "
                    "drafter path + Genesis sm_120 kernel autotune "
                    "assume bare-metal Linux/Windows. WSL paging may "
                    "reduce TPS by 5-10%; report results to "
                    "Genesis_internal_docs/wsl_blackwell_observations.md "
                    "if you have ground-truth bare-metal numbers."
                )
                break

    # PCIe lane warnings — flag any GPU wired below max width
    for lane in env.get("pcie_lanes", []):
        try:
            current = int(lane.get("width_current", "0").lstrip("x"))
            maximum = int(lane.get("width_max", "0").lstrip("x"))
        except (ValueError, AttributeError):
            continue
        if current > 0 and maximum > 0 and current < maximum:
            rec.append(
                f"[WARN] GPU {lane.get('index','?')} ({lane.get('name','?')}) "
                f"is wired x{current} but supports x{maximum} (gen {lane.get('gen_current','?')}/"
                f"{lane.get('gen_max','?')}). On TP=2 with TQ continuation-prefill "
                "this caps host↔device bandwidth and can cost 3-8% TPS. "
                "Check motherboard slot allocation (often x8/x8 vs x16/x16 BIOS "
                "setting) or PCIe riser cable integrity."
            )

    # Show at least one recommendation if everything is clean
    if not rec:
        rec.append("[OK] no issues detected. System is healthy.")

    return rec


# ─── Output formatters ────────────────────────────────────────────────────


def _format_text(report: dict[str, Any]) -> list[str]:
    L: list[str] = []
    L.append("=" * 72)
    L.append("Genesis doctor — system diagnostic")
    L.append("=" * 72)

    # Hardware
    L.append("")
    L.append("[1/6] Hardware")
    hw = report.get("hardware", {})
    if hw.get("gpus"):
        for g in hw["gpus"]:
            L.append(
                f"  GPU {g['index']}: {g['name']:<30} sm_{g['compute_capability'].replace('.', '')}  "
                f"VRAM {g['vram_total_gb']:.1f} GB"
            )
    else:
        L.append("  (no GPUs detected)")
    for e in hw.get("errors", []):
        L.append(f"  ⚠ {e}")

    # Environment quirks (WSL2, PCIe lanes) — only show when interesting
    env = report.get("environment", {})
    if env.get("is_wsl") or env.get("pcie_lanes"):
        L.append("")
        L.append("[1b] Host environment")
        if env.get("is_wsl"):
            L.append(f"  WSL:           {env.get('wsl_version','WSL')} (display "
                     "overhead +200-400 MiB; tighten gpu-mem-util)")
        for lane in env.get("pcie_lanes", []):
            L.append(
                f"  PCIe GPU {lane.get('index','?')}:    "
                f"gen {lane.get('gen_current','?')} x{lane.get('width_current','?')} "
                f"(max gen {lane.get('gen_max','?')} x{lane.get('width_max','?')})"
            )
        for e in env.get("errors", []):
            L.append(f"  ⚠ {e}")

    # Software
    L.append("")
    L.append("[2/6] Software")
    sw = report.get("software", {})
    L.append(f"  vllm:          {sw.get('vllm') or '(not installed)'}")
    if sw.get("vllm_commit"):
        L.append(f"    commit:      {sw['vllm_commit']}")
    L.append(f"  torch:         {sw.get('torch') or '(not installed)'}")
    L.append(f"  triton:        {sw.get('triton') or '(not installed)'}")
    L.append(f"  cuda runtime:  {sw.get('cuda_runtime') or '(none)'}")
    L.append(f"  nvidia driver: {sw.get('nvidia_driver') or '(unavailable)'}")
    L.append(f"  python:        {sw.get('python')}")

    # Model
    L.append("")
    L.append("[3/6] Model profile")
    mp = report.get("model_profile", {})
    if mp.get("resolved"):
        L.append(f"  model_class:   {mp.get('model_class', '?')}")
        L.append(f"  is_hybrid:     {mp.get('hybrid', mp.get('is_hybrid', '?'))}")
        L.append(f"  is_moe:        {mp.get('moe', mp.get('is_moe', '?'))}")
        L.append(f"  is_turboquant: {mp.get('turboquant', mp.get('is_turboquant', '?'))}")
        L.append(f"  quant_format:  {mp.get('quant_format', '?')}")
    else:
        L.append("  (model not loaded — run this on a live vllm container)")
        for e in mp.get("errors", []):
            L.append(f"  ⚠ {e}")

    # Patches
    L.append("")
    L.append("[4/6] Patch registry decisions")
    p = report.get("patches", {})
    L.append(f"  total: {p.get('total', 0)}, "
             f"APPLY: {p.get('apply', 0)}, SKIP: {p.get('skip', 0)}")
    apply_decisions = [d for d in p.get("decisions", []) if d["decision"] == "APPLY"]
    skip_decisions = [d for d in p.get("decisions", []) if d["decision"] == "SKIP"]
    if apply_decisions:
        L.append(f"  Applied ({len(apply_decisions)}):")
        for d in apply_decisions:
            L.append(f"    ✓ {d['patch_id']:<8} {d['title'][:55]}")
    if skip_decisions:
        # Hide the long list of opt-in skips by default; show count
        opt_in = [d for d in skip_decisions if "opt-in" in d.get("reason", "")]
        other_skips = [d for d in skip_decisions if "opt-in" not in d.get("reason", "")]
        if other_skips:
            L.append(f"  Skipped (non-opt-in, {len(other_skips)}):")
            for d in other_skips[:10]:  # cap at 10 to keep output readable
                L.append(f"    • {d['patch_id']:<8} {d['title'][:50]} — {d['reason'][:60]}")
            if len(other_skips) > 10:
                L.append(f"    ... and {len(other_skips) - 10} more")
        if opt_in:
            L.append(f"  Skipped (opt-in only, not engaged): {len(opt_in)}")

    # Lifecycle
    L.append("")
    L.append("[5/6] Lifecycle audit")
    lc = report.get("lifecycle", {})
    for state, ents in lc.get("by_state", {}).items():
        L.append(f"  {state}: {len(ents)}")
    if lc.get("total", 0) == 0:
        L.append("  (registry empty)")

    # Validator
    L.append("")
    L.append("[6/6] Validator")
    val = report.get("validator", {})
    if "error" in val:
        L.append(f"  ⚠ validator error: {val['error']}")
    else:
        si = val.get("static_issues", [])
        pi = val.get("plan_issues", [])
        if not si and not pi:
            L.append("  ✓ clean — no validator issues")
        else:
            for i in si:
                L.append(f"  [{i['severity']}] STATIC {i['patch_id']}: {i['message']}")
            for i in pi:
                L.append(f"  [{i['severity']}] PLAN   {i['patch_id']}: {i['message']}")

    # Recommendations
    L.append("")
    L.append("=" * 72)
    L.append("Recommendations")
    L.append("=" * 72)
    for r in report.get("recommendations", []):
        L.append(f"  {r}")
    L.append("=" * 72)
    return L


# ─── Driver ──────────────────────────────────────────────────────────────


def _section_preflight() -> dict[str, Any]:
    """PN60 + club#34 + club#43 doctor rules.

    Audit P2 fix 2026-05-05 (genesis_deep_cross_audit): PN60 was
    `default_on=True` in registry with credit "Doctor extension; runs at
    preflight" but `collect_report()` never called `run_all_preflight_checks()`.
    Operator running `genesis doctor` got no preflight signal.

    Now: doctor invokes preflight checks against the live container's logs
    (best-effort) and reports any WARN/ERROR findings under a dedicated
    `preflight` section. Operator-supplied `--quantization` and `--model`
    args are not available here (doctor takes no model context), so PN60
    quant validator only fires when the model_profile is resolved.
    """
    findings: list[dict[str, Any]] = []
    try:
        from vllm._genesis.compat.preflight_checks import (
            check_grammar_rejection_pattern,
            check_quant_arg,
            check_spec_decode_token_loop,
            fetch_container_logs,
        )
    except Exception as e:
        return {"status": "preflight module unavailable", "error": str(e),
                "findings": findings}

    # PN60 quant validator — only when we can locate config.json on disk.
    try:
        from vllm._genesis.model_detect import get_model_profile
        profile = get_model_profile()
        model_dir = profile.get("model_dir") or profile.get("model_path")
        cli_quant = os.environ.get("GENESIS_DOCTOR_CLI_QUANT", None)
        if model_dir and cli_quant:
            r = check_quant_arg(cli_quant, model_dir)
            findings.append({"name": r.name, "severity": r.severity,
                             "message": r.message,
                             "remediation": r.remediation})
    except Exception:
        pass

    # club#34 + club#43 — log-driven, fire only if container logs available.
    log_text = fetch_container_logs(container_name="vllm-server-mtp-test")
    if log_text:
        for r in (check_grammar_rejection_pattern(log_text),
                  check_spec_decode_token_loop(log_text)):
            findings.append({"name": r.name, "severity": r.severity,
                             "message": r.message,
                             "remediation": r.remediation})
    return {"status": "ok", "findings": findings}


def collect_report() -> dict[str, Any]:
    """Run all sections and return the unified report."""
    report: dict[str, Any] = {}
    report["hardware"] = _section_hardware()
    report["environment"] = _section_environment()
    report["software"] = _section_software()
    report["model_profile"] = _section_model_profile()
    report["patches"] = _section_patches()
    report["lifecycle"] = _section_lifecycle()
    report["validator"] = _section_validator()
    report["preflight"] = _section_preflight()
    report["recommendations"] = _section_recommendations(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.doctor",
        description="Genesis unified diagnostic — hardware + software + model "
                    "+ patches + validator + lifecycle.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output the full report as JSON (for CI / dashboards)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress section headers; print only critical issues")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    report = collect_report()

    if args.json:
        # Convert any non-JSON-serializable types
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.quiet:
        # Only print recommendations
        for r in report.get("recommendations", []):
            print(r)
        # Exit non-zero if there are ERROR-level recommendations
        for r in report.get("recommendations", []):
            if r.startswith("[ERROR]"):
                return 1
        return 0

    for line in _format_text(report):
        print(line)
    # Exit non-zero if validator found errors
    val = report.get("validator", {})
    has_errors = any(
        i["severity"] == "ERROR"
        for i in (val.get("static_issues", []) + val.get("plan_issues", []))
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
