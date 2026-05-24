# SPDX-License-Identifier: Apache-2.0
"""Genesis post-install smoke test — `genesis verify`.

Replaces "did setup.sh / install.sh actually work?" guesswork with a
deterministic checklist. Models the same pattern rustup uses for
post-install verification (rustup-init prints `1. ... 2. ... `).

Levels:
  --quick    Fast checks, no GPU/model required (~3 sec).  Default.
  --boot     Quick + minimal vLLM boot probe (requires GPU + model).
             Needs at least: a model on disk, ~15 GB VRAM free.
             ~60 sec.
  --full     Boot + tool-call probe + 1K context probe (~3 min).

Quick checks (always run, no external deps):
  C1   genesis package importable
  C2   PATCH_REGISTRY loadable + entries valid
  C3   apply_all module importable
  C4   compat/cli dispatcher routes preset/doctor/verify
  C5   gpu_profile detects a GPU (warn if no CUDA)
  C6   vllm importable (warn if not)
  C7   vllm pin matches Genesis expectation (warn on drift)
  C8   genesis_vllm_plugin entry point registered (warn if missing)
  C9   At least one preset matches detected GPU (info-only)

Boot checks (--boot, requires CUDA + model):
  B1   apply_all() returns success on this vllm tree
  B2   chunk.py self-install hook present (if P103 enabled)
  B3   vLLM imports without error after Genesis applied

Full checks (--full):
  F1   ~10s real boot of a tiny model (e.g. TinyLlama)
  F2   single tool-call probe via OpenAI-compatible /v1/chat/completions

Exit codes:
  0    All applicable checks passed (warnings allowed)
  1    At least one check FAILED (system not functional)
  2    Bad CLI args

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── Output formatting (mirror install.sh style) ────────────────────────


_TTY = sys.stdout.isatty()
_COLORS = {
    "reset": "\033[0m" if _TTY else "",
    "bold": "\033[1m" if _TTY else "",
    "red": "\033[31m" if _TTY else "",
    "green": "\033[32m" if _TTY else "",
    "yellow": "\033[33m" if _TTY else "",
    "blue": "\033[34m" if _TTY else "",
    "gray": "\033[90m" if _TTY else "",
}


def _c(color: str, s: str) -> str:
    return f"{_COLORS[color]}{s}{_COLORS['reset']}"


# ─── Check-result dataclass ─────────────────────────────────────────────


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    detail: str = ""
    duration_ms: int = 0
    hint: Optional[str] = None  # follow-up suggestion for WARN/FAIL


@dataclass
class VerifyReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.checks.append(r)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.status == PASS)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == FAIL)

    @property
    def n_skip(self) -> int:
        return sum(1 for c in self.checks if c.status == SKIP)

    @property
    def overall_pass(self) -> bool:
        """True if no FAIL — WARN allowed."""
        return self.n_fail == 0

    def to_dict(self) -> dict:
        return {
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "duration_ms": c.duration_ms,
                    "hint": c.hint,
                }
                for c in self.checks
            ],
            "summary": {
                "pass": self.n_pass,
                "warn": self.n_warn,
                "fail": self.n_fail,
                "skip": self.n_skip,
                "overall_pass": self.overall_pass,
            },
        }


# ─── Check runner harness ───────────────────────────────────────────────


def _run_check(
    name: str, fn: Callable[[], CheckResult], report: VerifyReport
) -> CheckResult:
    """Execute fn(), wrap any exception as FAIL, time it, append to report."""
    t0 = time.time()
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001
        result = CheckResult(
            name=name,
            status=FAIL,
            detail=f"unexpected exception: {type(e).__name__}: {e}",
        )
    result.duration_ms = int((time.time() - t0) * 1000)
    if not result.name:
        result.name = name
    report.add(result)
    return result


# ─── Quick checks (no GPU/model required) ──────────────────────────────


def _check_genesis_importable() -> CheckResult:
    try:
        importlib.import_module("vllm._genesis")
        return CheckResult(
            "C1 genesis package importable",
            PASS,
            "vllm._genesis imports OK",
        )
    except ImportError as e:
        return CheckResult(
            "C1 genesis package importable",
            FAIL,
            str(e),
            hint=(
                "Genesis source not on PYTHONPATH. Either:\n"
                "  • run install.sh (it symlinks vllm/_genesis into "
                "the vllm install)\n"
                "  • export PYTHONPATH=$GENESIS_HOME:$PYTHONPATH"
            ),
        )


def _check_dispatcher_loads() -> CheckResult:
    try:
        from vllm._genesis.dispatcher import PATCH_REGISTRY

        n = len(PATCH_REGISTRY)
        if n < 50:
            return CheckResult(
                "C2 PATCH_REGISTRY loadable",
                WARN,
                f"only {n} entries (expected ≥50 — partial install?)",
            )
        return CheckResult(
            "C2 PATCH_REGISTRY loadable",
            PASS,
            f"{n} patch entries registered",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C2 PATCH_REGISTRY loadable",
            FAIL,
            f"{type(e).__name__}: {e}",
        )


def _check_apply_all_importable() -> CheckResult:
    try:
        importlib.import_module("vllm._genesis.patches.apply_all")
        return CheckResult(
            "C3 apply_all module importable",
            PASS,
            "vllm._genesis.patches.apply_all imports OK",
        )
    except ImportError as e:
        return CheckResult(
            "C3 apply_all module importable",
            FAIL,
            str(e),
        )


def _check_cli_routes() -> CheckResult:
    try:
        from vllm._genesis.compat.cli import KNOWN_SUBCOMMANDS

        required = {"doctor", "preset", "verify"}
        missing = required - KNOWN_SUBCOMMANDS
        if missing:
            return CheckResult(
                "C4 unified CLI dispatcher",
                FAIL,
                f"missing subcommands: {sorted(missing)}",
            )
        return CheckResult(
            "C4 unified CLI dispatcher",
            PASS,
            f"{len(KNOWN_SUBCOMMANDS)} subcommands available",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C4 unified CLI dispatcher",
            FAIL,
            f"{type(e).__name__}: {e}",
        )


def _check_gpu_detected() -> CheckResult:
    try:
        from vllm._genesis.gpu_profile import detect_current_gpu
    except ImportError as e:
        return CheckResult(
            "C5 GPU detected",
            WARN,
            f"gpu_profile import failed: {e}",
        )

    gpu = detect_current_gpu()
    if gpu is None:
        return CheckResult(
            "C5 GPU detected",
            WARN,
            "no CUDA / torch unavailable — Genesis can install but "
            "patches require GPU at runtime",
            hint="If you're on CPU-only host, this is expected.",
        )
    name = gpu.get("name_canonical") or gpu.get("name_detected") or "unknown"
    if gpu.get("match_key") is None:
        return CheckResult(
            "C5 GPU detected",
            WARN,
            f"unknown GPU '{name}' — not in Genesis spec database",
            hint=(
                "Append entry to vllm/_genesis/gpu_profile.py:GPU_SPECS "
                "and submit a PR with NVIDIA datasheet specs."
            ),
        )
    return CheckResult(
        "C5 GPU detected",
        PASS,
        f"{name} ({gpu['regime']} regime, "
        f"{gpu.get('bandwidth_gb_s', '?')} GB/s, "
        f"{gpu.get('l2_mb', '?')} MB L2)",
    )


def _check_vllm_importable() -> CheckResult:
    try:
        import vllm
        ver = getattr(vllm, "__version__", "?")
        return CheckResult(
            "C6 vllm importable",
            PASS,
            f"vllm {ver}",
        )
    except ImportError as e:
        return CheckResult(
            "C6 vllm importable",
            WARN,
            f"vllm not installed: {e}",
            hint="pip install vllm",
        )


def _check_vllm_pin_match() -> CheckResult:
    try:
        import vllm

        ver = getattr(vllm, "__version__", "?")
    except ImportError:
        return CheckResult(
            "C7 vllm pin compatibility",
            SKIP,
            "vllm not importable",
        )

    # Genesis is pinned to vllm 0.20.x — anchor drift likely outside
    if "0.20" in ver:
        return CheckResult(
            "C7 vllm pin compatibility",
            PASS,
            f"vllm {ver} matches Genesis 0.20.x pin window",
        )
    return CheckResult(
        "C7 vllm pin compatibility",
        WARN,
        f"vllm {ver} outside Genesis 0.20.x pin — anchor drift likely",
        hint=(
            "Genesis text-patches target specific upstream lines; on "
            "drift they SKIP rather than corrupt. Run `genesis doctor` "
            "for per-patch status."
        ),
    )


def _check_plugin_entry_point() -> CheckResult:
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="vllm.general_plugins")
        names = [ep.name for ep in eps]
        if "genesis_v7" in names:
            return CheckResult(
                "C8 vllm plugin entry point",
                PASS,
                "vllm.general_plugins → genesis_v7 registered",
            )
        return CheckResult(
            "C8 vllm plugin entry point",
            WARN,
            f"genesis_v7 not in vllm.general_plugins (found: {names})",
            hint=(
                "Without the plugin, Genesis won't auto-load in vllm "
                "spawn workers. Setattr-based patches (P103, P67) will "
                "die on `exec vllm serve`. Install:\n"
                "  pip install -e $GENESIS_HOME/tools/genesis_vllm_plugin/"
            ),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C8 vllm plugin entry point",
            WARN,
            f"entry_points lookup failed: {e}",
        )


def _check_preset_for_current_gpu() -> CheckResult:
    try:
        from vllm._genesis.compat.presets import auto_match
    except ImportError:
        return CheckResult(
            "C9 preset for this rig",
            SKIP,
            "presets module not available",
        )

    p = auto_match()
    if p is None:
        return CheckResult(
            "C9 preset for this rig",
            WARN,
            "no preset auto-matches — pick manually",
            hint="genesis preset list",
        )
    return CheckResult(
        "C9 preset for this rig",
        PASS,
        f"{p.key} ({p.title})",
    )


# ─── Boot checks (--boot, requires CUDA + model) ───────────────────────


def _check_apply_all_dry_run() -> CheckResult:
    """Run apply_all in dry-run mode against the current vllm install."""
    try:
        from vllm._genesis.patches.apply_all import (
            run_apply_all,
            set_apply_mode,
        )
    except ImportError as e:
        return CheckResult(
            "B1 apply_all dry-run",
            FAIL,
            f"apply_all module missing: {e}",
        )

    try:
        set_apply_mode(False)  # dry-run
        results = run_apply_all()
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "B1 apply_all dry-run",
            FAIL,
            f"apply_all raised: {type(e).__name__}: {e}",
        )

    failed = [r for r in results if getattr(r, "status", None) == "failed"]
    n_total = len(results)
    if failed:
        first = failed[0]
        name = getattr(first, "name", "?")
        reason = getattr(first, "reason", "?")
        return CheckResult(
            "B1 apply_all dry-run",
            FAIL,
            f"{len(failed)} of {n_total} patches FAILED in dry-run "
            f"(first: {name} — {reason})",
        )
    return CheckResult(
        "B1 apply_all dry-run",
        PASS,
        f"{n_total} patches dry-run cleanly",
    )


def _check_p103_self_install_hook() -> CheckResult:
    """If GENESIS_ENABLE_P103=1, the hook should be present in chunk.py."""
    if os.environ.get("GENESIS_ENABLE_P103", "").strip().lower() not in (
        "1", "true", "yes", "on"
    ):
        return CheckResult(
            "B2 P103 self-install hook",
            SKIP,
            "GENESIS_ENABLE_P103 not set — skip",
        )

    try:
        from vllm._genesis.guards import resolve_vllm_file
    except ImportError as e:
        return CheckResult(
            "B2 P103 self-install hook",
            FAIL,
            f"guards import failed: {e}",
        )

    chunk_py = resolve_vllm_file("model_executor/layers/fla/ops/chunk.py")
    if chunk_py is None:
        return CheckResult(
            "B2 P103 self-install hook",
            WARN,
            "chunk.py not resolvable — vllm install layout differs",
        )
    try:
        with open(chunk_py) as f:
            content = f.read()
    except OSError as e:
        return CheckResult(
            "B2 P103 self-install hook",
            FAIL,
            f"chunk.py unreadable: {e}",
        )
    marker = "Genesis P103 v7.69 self-install"
    if marker in content:
        return CheckResult(
            "B2 P103 self-install hook",
            PASS,
            "self-install hook present in chunk.py",
        )
    return CheckResult(
        "B2 P103 self-install hook",
        FAIL,
        "GENESIS_ENABLE_P103=1 but self-install hook absent from chunk.py",
        hint=(
            "Run `python3 -m vllm._genesis.patches.apply_all` once to "
            "install the hook. P103 will fail to fire in vllm workers "
            "without it."
        ),
    )


def _check_vllm_imports_after_apply() -> CheckResult:
    """After patches apply, vllm should still import cleanly (defensive
    test against patches breaking vllm import)."""
    try:
        # Force re-import by removing cached modules then re-importing
        for mod_name in list(sys.modules):
            if mod_name.startswith("vllm.model_executor.layers.fla"):
                del sys.modules[mod_name]
        importlib.import_module(
            "vllm.model_executor.layers.fla.ops.chunk"
        )
        return CheckResult(
            "B3 vllm imports after apply",
            PASS,
            "FLA chunk.py imports cleanly post-patch",
        )
    except ImportError as e:
        return CheckResult(
            "B3 vllm imports after apply",
            WARN,
            f"FLA chunk.py not importable: {e}",
            hint="May be expected if vllm doesn't have this module.",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "B3 vllm imports after apply",
            FAIL,
            f"chunk.py import raised: {type(e).__name__}: {e}",
        )


# ─── Public entry: run_verify(level) ───────────────────────────────────


def run_verify(level: str = "quick") -> VerifyReport:
    """Run the verify checklist at the requested level.

    Args:
      level: 'quick' | 'boot' | 'full'

    Returns:
      VerifyReport with per-check results.
    """
    if level not in ("quick", "boot", "full"):
        raise ValueError(f"unknown level {level!r}")

    report = VerifyReport()

    # ─── Quick checks (always) ───
    quick_checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("C1", _check_genesis_importable),
        ("C2", _check_dispatcher_loads),
        ("C3", _check_apply_all_importable),
        ("C4", _check_cli_routes),
        ("C5", _check_gpu_detected),
        ("C6", _check_vllm_importable),
        ("C7", _check_vllm_pin_match),
        ("C8", _check_plugin_entry_point),
        ("C9", _check_preset_for_current_gpu),
    ]
    for cid, fn in quick_checks:
        _run_check(cid, fn, report)

    # ─── Boot checks (--boot or --full) ───
    if level in ("boot", "full"):
        boot_checks: list[tuple[str, Callable[[], CheckResult]]] = [
            ("B1", _check_apply_all_dry_run),
            ("B2", _check_p103_self_install_hook),
            ("B3", _check_vllm_imports_after_apply),
        ]
        for cid, fn in boot_checks:
            _run_check(cid, fn, report)

    # ─── Full checks (--full only) ───
    if level == "full":
        # F1 + F2: real boot probe + tool-call probe.
        # These need a model + free VRAM. Stub for now (Day 4 implements).
        report.add(CheckResult(
            "F1 vLLM boot probe",
            SKIP,
            "F1/F2 not yet implemented — use tools/external_probe/ scripts",
            hint=(
                "Until F1/F2 land, run after install:\n"
                "  bash $GENESIS_HOME/launch/start_*.sh    # boot vllm serve\n"
                "  curl -s localhost:8000/v1/models         # smoke test"
            ),
        ))

    return report


# ─── Pretty-print report ───────────────────────────────────────────────


def _status_glyph(status: str) -> str:
    if status == PASS:
        return _c("green", "✓ PASS")
    if status == WARN:
        return _c("yellow", "⚠ WARN")
    if status == FAIL:
        return _c("red", "✗ FAIL")
    if status == SKIP:
        return _c("gray", "− SKIP")
    return status


def render_report(report: VerifyReport, *, verbose: bool = False) -> str:
    """Pretty-print report as text. Returns the formatted string."""
    lines = []
    lines.append("")
    lines.append(_c("bold", "Genesis verify"))
    lines.append("")
    for c in report.checks:
        glyph = _status_glyph(c.status)
        ms = f"{c.duration_ms:>4}ms"
        line = f"  {glyph}  {ms}  {c.name}"
        if c.detail:
            line += f"  {_c('gray', '— ' + c.detail)}"
        lines.append(line)
        if c.hint and c.status in (WARN, FAIL):
            for hint_line in c.hint.split("\n"):
                lines.append(_c("gray", f"           ↳ {hint_line}"))
    lines.append("")

    summary = (
        f"{report.n_pass} pass / {report.n_warn} warn / "
        f"{report.n_fail} fail / {report.n_skip} skip"
    )
    if report.overall_pass:
        lines.append(_c("green", f"  ✓ overall: PASS  ({summary})"))
    else:
        lines.append(_c("red", f"  ✗ overall: FAIL  ({summary})"))
    lines.append("")
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.verify",
        description="Genesis post-install smoke test",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--quick",
        action="store_const",
        dest="level",
        const="quick",
        help="fast checks, no GPU/model required (default, ~3 sec)",
    )
    g.add_argument(
        "--boot",
        action="store_const",
        dest="level",
        const="boot",
        help="quick + apply_all dry-run + chunk.py hook check",
    )
    g.add_argument(
        "--full",
        action="store_const",
        dest="level",
        const="full",
        help="boot + real vllm boot probe + tool-call probe (~3 min)",
    )
    parser.set_defaults(level="quick")
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of text"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="show extra detail (currently a no-op; reserved)"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_verify(level=args.level)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report, verbose=args.verbose))

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
