# SPDX-License-Identifier: Apache-2.0
"""Genesis compat — version-range checking.

Each Genesis patch can declare version ranges in its `applies_to` block:

    "PN14": {
        ...
        "applies_to": {
            "vllm_version_range":   (">=0.20.0", "<0.21.0"),
            "torch_version_min":    "2.0",
            "triton_version_min":   "3.0",
            "cuda_runtime_min":     "12.0",
            "compute_capability_min": (8, 6),  # sm_86
        },
    },

This module:

  1. Detects the live versions (vllm, torch, triton, cuda runtime,
     nvidia driver) once per process — caches the result.
  2. Provides `check_version_constraints(constraints) -> (ok, reasons)`
     used by the dispatcher when evaluating each patch's applies_to.

Detection is **defensive** — if a particular probe fails (e.g. nvidia-smi
not on PATH, or torch built without CUDA), we return None for that
field rather than raising. Constraints that reference None-valued fields
are treated as **conservatively satisfied** (better to apply the patch
and let lower layers fail loudly than to silently skip).

Notes on packaging.specifiers
-----------------------------
We use `packaging.specifiers.SpecifierSet` (stdlib of pip / setuptools)
to handle PEP 440 version specifiers. This means "<0.21.0" / ">=0.20.0"
work as documented, including pre-release semantics (`>=0.20.0` matches
`0.20.0rc1` only when explicit prereleases=True, which we set to True
because vllm uses dev-versions extensively).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("genesis.compat.version_check")


# ─── Result types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VersionProfile:
    """Snapshot of detected versions for the current process."""
    vllm: str | None = None
    vllm_commit: str | None = None
    torch: str | None = None
    triton: str | None = None
    cuda_runtime: str | None = None
    nvidia_driver: str | None = None
    python: str | None = None
    compute_capabilities: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    # If anything in detection raised, the error is captured here for the
    # doctor to print. Detection never raises into the caller.
    errors: tuple[str, ...] = field(default_factory=tuple)


# ─── Detection helpers ───────────────────────────────────────────────────


_CACHED_PROFILE: VersionProfile | None = None


def detect_versions(refresh: bool = False) -> VersionProfile:
    """Detect all relevant versions on the current system. Cached
    per-process; pass `refresh=True` to re-probe (e.g. for tests).
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None and not refresh:
        return _CACHED_PROFILE

    errors: list[str] = []

    # vllm — module + git commit suffix if present
    vllm_v = None
    vllm_commit = None
    try:
        import vllm
        vllm_v = getattr(vllm, "__version__", None)
        if vllm_v and "+g" in vllm_v:
            _base, _, suffix = vllm_v.partition("+g")
            vllm_commit = suffix
    except Exception as e:
        errors.append(f"vllm import: {e}")

    # torch
    torch_v = None
    try:
        import torch
        torch_v = torch.__version__
    except Exception as e:
        errors.append(f"torch import: {e}")

    # triton
    triton_v = None
    try:
        import triton
        triton_v = triton.__version__
    except Exception:
        # triton not always installed (CPU-only / ROCm); not an error
        pass

    # CUDA runtime (via torch first — most reliable, no shell dependency)
    cuda_runtime = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_runtime = torch.version.cuda
    except Exception as e:
        errors.append(f"cuda runtime probe: {e}")

    # nvidia driver — via nvidia-smi (best-effort)
    driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            driver = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        # nvidia-smi missing on CPU-only / ROCm hosts — driver stays None
        pass

    # Python
    py_v = None
    try:
        import sys
        py_v = ".".join(str(x) for x in sys.version_info[:3])
    except Exception:
        # sys.version_info is always present; defensive guard for exotic CPython forks
        pass

    # Compute capabilities (per-GPU)
    ccs: list[tuple[int, int]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                ccs.append((major, minor))
    except Exception as e:
        errors.append(f"compute capability probe: {e}")

    profile = VersionProfile(
        vllm=vllm_v,
        vllm_commit=vllm_commit,
        torch=torch_v,
        triton=triton_v,
        cuda_runtime=cuda_runtime,
        nvidia_driver=driver,
        python=py_v,
        compute_capabilities=tuple(ccs),
        errors=tuple(errors),
    )
    _CACHED_PROFILE = profile
    return profile


def reset_cache() -> None:
    """Reset the cached profile (test-only)."""
    global _CACHED_PROFILE
    _CACHED_PROFILE = None


# ─── Constraint matching ────────────────────────────────────────────────


def _split_pep440_pre_dev_local(version: str) -> str:
    """Strip a vllm-style commit suffix `0.20.1rc1.dev16+g7a1eb8ac2` so
    SpecifierSet can match cleanly. We keep the pre/dev components — they
    matter for correctness — but drop the local `+gSHA` segment which
    PEP 440 allows but specifiers usually ignore."""
    return version.partition("+")[0]


def _match_pep440(version: str, specifier: str) -> bool | None:
    """Return True/False if version matches PEP 440 specifier, or None
    if the specifier is malformed (caller should treat as 'unknown')."""
    if version is None:
        return None
    try:
        from packaging.specifiers import SpecifierSet
    except Exception:
        return None
    try:
        sset = SpecifierSet(specifier)
        # vllm uses .dev versions extensively; allow prereleases
        sset.prereleases = True
        return _split_pep440_pre_dev_local(version) in sset
    except Exception:
        return None


def _match_min(version: str, minimum: str) -> bool | None:
    """Simple ">=" check by parsed Version comparison."""
    if version is None:
        return None
    try:
        from packaging.version import Version, InvalidVersion
    except Exception:
        return None
    try:
        return Version(_split_pep440_pre_dev_local(version)) >= Version(minimum)
    except (InvalidVersion, Exception):
        return None


def _match_compute_capability(
    actual: tuple[tuple[int, int], ...], minimum: tuple[int, int],
) -> bool | None:
    """All detected GPUs must be >= min compute capability."""
    if not actual:
        return None  # no GPUs detected — be conservative
    return all(cc >= minimum for cc in actual)


@dataclass(frozen=True)
class ConstraintResult:
    """Outcome of evaluating one constraint key against a profile."""
    key: str
    constraint: Any
    actual: Any
    matched: bool | None  # True / False / None (couldn't determine)
    reason: str


def check_version_constraints(
    constraints: dict[str, Any],
    profile: VersionProfile | None = None,
) -> tuple[bool, list[ConstraintResult]]:
    """Evaluate version-related constraints from a patch's applies_to.

    Returns:
        (all_passed, results) — `all_passed=True` only if every
        constraint we COULD evaluate matched. None-valued constraints
        (we couldn't determine) are treated as 'conservatively pass'
        so the patch can still apply on imperfect detection.
    """
    if profile is None:
        profile = detect_versions()

    results: list[ConstraintResult] = []

    def _record(key, constraint, actual, matched, reason):
        results.append(ConstraintResult(
            key=key, constraint=constraint, actual=actual,
            matched=matched, reason=reason,
        ))

    # vllm version range — list of specifiers, all must match
    if "vllm_version_range" in constraints:
        spec = constraints["vllm_version_range"]
        # Accept tuple / list / single string for ergonomics
        if isinstance(spec, (str,)):
            specs = [spec]
        else:
            specs = list(spec)
        actual = profile.vllm
        if actual is None:
            _record("vllm_version_range", specs, actual, None,
                    "vllm version not detected — conservative pass")
        else:
            all_ok = True
            for s in specs:
                m = _match_pep440(actual, s)
                if m is False:
                    all_ok = False
                    break
                if m is None:
                    all_ok = False  # malformed = treat as not matching
                    break
            _record("vllm_version_range", specs, actual, all_ok,
                    f"vllm {actual} {'satisfies' if all_ok else 'violates'} {specs!r}")

    # torch / triton / cuda / driver — all just min comparisons
    for key, attr, label in [
        ("torch_version_min", "torch", "torch"),
        ("triton_version_min", "triton", "triton"),
        ("cuda_runtime_min", "cuda_runtime", "cuda"),
        ("nvidia_driver_min", "nvidia_driver", "driver"),
        ("python_version_min", "python", "python"),
    ]:
        if key not in constraints:
            continue
        minimum = constraints[key]
        actual = getattr(profile, attr)
        if actual is None:
            _record(key, minimum, actual, None,
                    f"{label} version not detected — conservative pass")
            continue
        m = _match_min(actual, minimum)
        if m is None:
            _record(key, minimum, actual, None,
                    f"{label} {actual} parse failed — conservative pass")
        else:
            _record(key, minimum, actual, m,
                    f"{label} {actual} {'>=' if m else '<'} required {minimum}")

    # Compute capability — tuple comparison
    if "compute_capability_min" in constraints:
        minimum = tuple(constraints["compute_capability_min"])
        actual = profile.compute_capabilities
        m = _match_compute_capability(actual, minimum)
        if m is None:
            _record("compute_capability_min", minimum, actual, None,
                    "no GPU compute capability detected — conservative pass")
        else:
            actual_str = ", ".join(f"sm_{a[0]}{a[1]}" for a in actual)
            _record("compute_capability_min", minimum, actual, m,
                    f"GPUs ({actual_str}) {'all >=' if m else 'some <'} "
                    f"sm_{minimum[0]}{minimum[1]}")

    if "compute_capability_max" in constraints:
        maximum = tuple(constraints["compute_capability_max"])
        actual = profile.compute_capabilities
        if not actual:
            _record("compute_capability_max", maximum, actual, None,
                    "no GPU compute capability detected — conservative pass")
        else:
            ok = all(cc <= maximum for cc in actual)
            actual_str = ", ".join(f"sm_{a[0]}{a[1]}" for a in actual)
            _record("compute_capability_max", maximum, actual, ok,
                    f"GPUs ({actual_str}) {'all <=' if ok else 'some >'} "
                    f"sm_{maximum[0]}{maximum[1]}")

    # Aggregate. None-valued = conservative pass.
    all_passed = all(r.matched is not False for r in results)
    return all_passed, results


def format_version_report(profile: VersionProfile | None = None) -> list[str]:
    """Plain-text lines describing the detected version profile (for
    `genesis doctor` output)."""
    if profile is None:
        profile = detect_versions()
    lines = []
    lines.append(f"  vllm:          {profile.vllm or '(not installed)'}")
    if profile.vllm_commit:
        lines.append(f"    commit:      {profile.vllm_commit}")
    lines.append(f"  torch:         {profile.torch or '(not installed)'}")
    lines.append(f"  triton:        {profile.triton or '(not installed)'}")
    lines.append(f"  cuda runtime:  {profile.cuda_runtime or '(none)'}")
    lines.append(f"  nvidia driver: {profile.nvidia_driver or '(none / nvidia-smi unavailable)'}")
    lines.append(f"  python:        {profile.python}")
    if profile.compute_capabilities:
        ccs = ", ".join(f"sm_{a}{b}" for a, b in profile.compute_capabilities)
        lines.append(f"  compute caps:  {ccs}")
    if profile.errors:
        lines.append("  detection errors:")
        for e in profile.errors:
            lines.append(f"    - {e}")
    return lines
