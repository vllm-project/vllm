# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit vLLM compiled libraries for PyTorch stable ABI compliance."""

import fnmatch
import sys
from pathlib import Path

from torch_abi_audit import inspect_package
from torch_abi_audit.report import ExtensionReport, PackageReport

# Temporary allowlist of extensions not yet on the stable ABI.
# Shrink and remove over time.
ALLOWED_UNSTABLE_LIBRARIES: tuple[str, ...] = (
    "vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "_flashmla_C.abi3.so",
    "_flashmla_extension_C.abi3.so",
    "third_party/deep_gemm/_C*.so",
)


def _relative_path(lib: ExtensionReport, package_root: Path) -> str:
    try:
        return lib.path.relative_to(package_root).as_posix()
    except ValueError:
        return lib.path.name


def _is_torch_unstable(lib: ExtensionReport) -> bool:
    return lib.error is None and lib.torch.uses_torch and not lib.torch.stable


def _matches_allowlist(rel_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _iter_libs(report: PackageReport) -> tuple[ExtensionReport, ...]:
    return (*report.extensions, *report.bundled_libs)


def _collect_unstable(report: PackageReport) -> list[str]:
    return sorted(
        _relative_path(lib, report.root)
        for lib in _iter_libs(report)
        if _is_torch_unstable(lib)
    )


def _find_stale_allowlist_entries(
    report: PackageReport, patterns: tuple[str, ...]
) -> list[str]:
    """Allowlist patterns that match a built library which is no longer unstable."""
    stale: list[str] = []
    for pattern in patterns:
        for lib in _iter_libs(report):
            if lib.error is not None:
                continue
            if not fnmatch.fnmatch(_relative_path(lib, report.root), pattern):
                continue
            if not _is_torch_unstable(lib):
                stale.append(pattern)
                break
    return stale


def check_torch_abi(
    package: str = "vllm",
    patterns: tuple[str, ...] = ALLOWED_UNSTABLE_LIBRARIES,
) -> int:
    report = inspect_package(package)
    if report.error:
        print(f"error: failed to inspect {package!r}: {report.error}", file=sys.stderr)
        return 2

    unstable = _collect_unstable(report)
    unexpected = [
        rel_path for rel_path in unstable if not _matches_allowlist(rel_path, patterns)
    ]
    stale = _find_stale_allowlist_entries(report, patterns)

    if unexpected or stale:
        if unexpected:
            print(
                "Not allowed: torch-unstable libraries outside "
                f"ALLOWED_UNSTABLE_LIBRARIES: {', '.join(unexpected)}",
                file=sys.stderr,
            )
        if stale:
            print(
                "Not allowed: stale ALLOWED_UNSTABLE_LIBRARIES entries: "
                f"{', '.join(stale)}",
                file=sys.stderr,
            )
        return 1

    print("Torch stable ABI check passed.")
    return 0


if __name__ == "__main__":
    print(">>> Auditing vLLM extension modules for PyTorch stable ABI compliance")
    sys.exit(check_torch_abi())
