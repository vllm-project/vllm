#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect the manylinux platform tag for a wheel and rename it in place.

vLLM's build images produce wheels with the generic ``linux_<arch>`` platform
tag, which installers like ``pip`` won't accept off PyPI/our index. We need to
rewrite the platform tag to the appropriate ``manylinux_<major>_<minor>_<arch>``
before uploading.

Historically the tag was hard-coded per build (``manylinux_2_31`` for the
Ubuntu 20.04-based image, ``manylinux_2_35`` for the Ubuntu 22.04-based
images). That is brittle: bumping the base image silently produces wheels
labelled with the wrong glibc requirement. This script asks ``auditwheel``
to derive the tag from the symbol versions actually referenced by the
binaries inside the wheel, so the label tracks reality.

We can't simply call ``auditwheel repair`` -- it tries to graft external
shared libraries into the wheel and fails on vLLM's CUDA/cuBLAS dependencies.
Instead we use ``auditwheel.wheel_abi.analyze_wheel_abi`` directly, which is
the same call that powers ``auditwheel show``, and read off
``winfo.sym_policy.name``.

Usage:
    detect-manylinux-tag.py <wheel_path>

The wheel is renamed in place; the new path is printed on stdout. All
diagnostics go to stderr so callers can capture stdout safely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auditwheel.error import (
    AuditwheelError,
    NonPlatformWheelError,
    WheelToolsError,
)
from auditwheel.wheel_abi import analyze_wheel_abi
from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc


def detect_platform_tag(wheel_path: Path) -> str:
    """Return the most precise platform tag the wheel is consistent with.

    Mirrors ``auditwheel show`` but returns ``sym_policy`` rather than
    ``overall_policy``: we only care about the glibc symbol versions used,
    not about other policy axes (ISA extensions, blacklist, etc.) that
    ``overall_policy`` folds in.
    """
    fn = wheel_path.name

    try:
        arch = get_wheel_architecture(fn)
    except (WheelToolsError, NonPlatformWheelError):
        # Architecture isn't deducible from the filename; let auditwheel
        # infer it from the ELF binaries inside the wheel.
        arch = None

    try:
        libc = get_wheel_libc(fn)
    except WheelToolsError:
        # An unrepaired wheel uses ``linux_<arch>``, which doesn't encode
        # libc. Let auditwheel infer it from the ELF binaries.
        libc = None

    winfo = analyze_wheel_abi(
        libc,
        arch,
        wheel_path,
        frozenset(),
        disable_isa_ext_check=False,
        allow_graft=False,
    )
    return winfo.sym_policy.name


def rename_wheel(wheel_path: Path, new_platform_tag: str) -> Path:
    """Rename the wheel in place, replacing only its platform tag."""
    # Wheel filename per PEP 427:
    #   {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    # The platform tag is always the last ``-``-separated token before
    # ``.whl``. Compound tags like ``manylinux_2_31_x86_64`` use ``_`` as the
    # internal separator, so ``-``-splitting is unambiguous.
    parts = wheel_path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unrecognised wheel filename: {wheel_path.name}")
    parts[-1] = new_platform_tag
    new_path = wheel_path.with_name("-".join(parts) + ".whl")
    if new_path != wheel_path:
        wheel_path.rename(new_path)
    return new_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect a wheel's manylinux platform tag with "
        "auditwheel and rename the wheel in place."
    )
    parser.add_argument(
        "wheel",
        type=Path,
        help="Path to the wheel to inspect and rename.",
    )
    args = parser.parse_args()

    wheel_path: Path = args.wheel
    if not wheel_path.is_file():
        print(f"error: {wheel_path} is not a file", file=sys.stderr)
        return 1

    # Catch the things that ``analyze_wheel_abi`` and ``rename_wheel`` can
    # raise: any subclass of ``AuditwheelError`` (pure-Python wheels,
    # invalid libc, malformed wheels), filesystem errors, or our own
    # ``ValueError`` for an unrecognised wheel filename. Print a single
    # ``ERROR_TYPE: message`` line to stderr instead of a Python
    # traceback, which is much friendlier in CI logs.
    try:
        new_tag = detect_platform_tag(wheel_path)
        print(f"detected platform tag: {new_tag}", file=sys.stderr)
        new_path = rename_wheel(wheel_path, new_tag)
    except (AuditwheelError, ValueError, OSError) as e:
        print(
            f"error: failed to retag {wheel_path.name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2

    if new_path != wheel_path:
        print(f"renamed {wheel_path.name} -> {new_path.name}", file=sys.stderr)
    else:
        print(f"wheel already tagged {new_tag}", file=sys.stderr)

    print(new_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
