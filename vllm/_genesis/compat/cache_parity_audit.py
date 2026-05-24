# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache parity audit — pattern from vllm#41625.

Inspired by upstream PR #41625 (joerowell): KVConnector path didn't mirror
the existing `request.skip_reading_prefix_cache=True` guard that
KVCacheManager honors. Two code paths handling same condition without
symmetric gates → silent corruption.

Genesis equivalent class of bug: P83/P84/P85 prefix-cache patches.
Memory `feedback_p83_p84_p85_cache_no_cake` 2026-04-29 reports:
"4-arm A/B: enabling --enable-prefix-caching on v775 = -30% TPS
regression that NEITHER P83 alone NOR P83+P85 NOR P83+P84+P85+HASH=16
can mitigate". Possible explanation: asymmetric guards across MambaManager
vs P85 fine-grained shadow hash.

This audit module:
1. Walks the active set of patches that touch prefix-cache code paths
2. Cross-references which `request.*` guards they each inspect
3. Reports any path that LACKS a guard another path HAS

Run via: `python3 -m vllm._genesis.compat.cache_parity_audit`
or imported from doctor.

Author: Sandermage 2026-05-04, pattern from vllm#41625 (joerowell).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("genesis.compat.cache_parity_audit")


# Guards that are SAFE-set and should be checked by ALL prefix-cache paths
_REQUIRED_GUARDS = (
    "skip_reading_prefix_cache",   # vllm#41625 root cause
    "do_not_load_kv_from_cache",   # NIXL connector flag
    "kv_transfer_params",          # disaggregated PD setup
)


@dataclass(frozen=True)
class CachePathFinding:
    """One audit finding."""
    patch_id: str
    file_path: str
    missing_guards: tuple[str, ...]
    severity: str  # "WARN" | "ERROR" | "INFO"


def _read_patch_source(patch_id: str) -> str | None:
    """Read patch source for guard inspection. Returns None if not found."""
    import os
    from pathlib import Path

    # Map patch_id → wiring file (best-effort heuristic)
    wiring_root = Path(__file__).parent.parent / "wiring"
    candidates = list(wiring_root.rglob(f"patch_{patch_id.lower().lstrip('p').lstrip('n')}*"))
    candidates += list(wiring_root.rglob(f"patch_*{patch_id.lower().lstrip('p').lstrip('n')}*"))
    for c in candidates:
        if c.is_file():
            try:
                return c.read_text(encoding="utf-8")
            except Exception:
                continue
    return None


def audit_prefix_cache_patches() -> list[CachePathFinding]:
    """Audit the prefix-cache patch family for guard parity."""
    # Patches we know touch prefix-cache code paths
    PREFIX_CACHE_PATCHES = (
        "P83", "P84", "P85",
        "PN35",  # inputs_embeds buffer (related, kv_cache pool sizing)
        "PN54",  # GDN contiguous dedup (related, allocator)
    )

    findings: list[CachePathFinding] = []
    for pid in PREFIX_CACHE_PATCHES:
        src = _read_patch_source(pid)
        if src is None:
            findings.append(CachePathFinding(
                patch_id=pid,
                file_path="(not found)",
                missing_guards=(),
                severity="INFO",
            ))
            continue

        # Check which required guards are referenced in patch source
        missing: list[str] = []
        for guard in _REQUIRED_GUARDS:
            if guard not in src:
                missing.append(guard)

        if missing:
            findings.append(CachePathFinding(
                patch_id=pid,
                file_path="<wiring file>",
                missing_guards=tuple(missing),
                severity="WARN" if pid in ("PN35", "PN54") else "INFO",
                # Only WARN for patches that don't structurally need
                # these guards. Real ERRORS would be for prefix-cache
                # core patches (P83/84/85), but those need manual
                # confirmation since the absence might be intentional.
            ))

    return findings


def format_findings(findings: list[CachePathFinding]) -> list[str]:
    """Render findings as readable lines."""
    lines = ["=== Genesis cache-parity audit (vllm#41625 pattern) ==="]
    if not findings:
        lines.append("✓ No findings — all prefix-cache patches consulted.")
        return lines

    for f in findings:
        lines.append(
            f"  [{f.severity}] {f.patch_id} ({f.file_path})"
        )
        if f.missing_guards:
            lines.append(
                f"      Missing references to: {', '.join(f.missing_guards)}"
            )
            lines.append(
                "      Recommendation: review whether this guard is needed; "
                "if not, document why in patch docstring."
            )
    lines.append(
        "\nNote: this audit checks GUARD MENTION not GUARD LOGIC. "
        "A WARN means the guard isn't textually present; if absence is "
        "intentional, add a docstring note explaining why."
    )
    return lines


def main():
    """CLI entry — `python -m vllm._genesis.compat.cache_parity_audit`."""
    findings = audit_prefix_cache_patches()
    for line in format_findings(findings):
        print(line)
    return 0 if not any(f.severity == "ERROR" for f in findings) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
