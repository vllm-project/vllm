#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stamp the *installed* vLLM version metadata to a given string (e.g. a commit).

Why this exists
---------------
In our image pipeline vLLM is installed editable in the **base** image, so its
version (written by setuptools-scm into the dist-info ``METADATA`` and into
``vllm/_version.py``) is frozen at the *base* build. ``Dockerfile.update`` copies
new source on top but deliberately does **not** reinstall vLLM (that would
trigger the multi-hour CUDA rebuild), so ``pip show vllm`` / ``pip list`` keep
reporting the *base* commit instead of the commit the image was built from.

This script rewrites the two places that decide the reported version, in-place
and without any recompilation:
  1. the installed ``vllm-*.dist-info/METADATA`` ``Version:`` field (pip show)
  2. ``vllm/_version.py`` (drives ``vllm.__version__``)

``__version_tuple__`` is kept as ``(0, 0, 0)`` so vLLM keeps treating the tree as
a dev build (see ``vllm/version.py``); vLLM never parses its own ``__version__``
with ``packaging``, so an arbitrary string (a bare commit hash) is safe.

Usage: ``python docker/stamp_vllm_version.py <version-string> [--append]``
"""

import glob
import pathlib
import site
import sys


def _find_vllm_dist_info() -> pathlib.Path:
    # Preferred: ask importlib.metadata where vLLM's dist-info lives.
    try:
        import importlib.metadata as md

        dist = md.distribution("vllm")
        p = getattr(dist, "_path", None)
        if p is not None:
            return pathlib.Path(p)
    except Exception:
        pass
    # Fallback: scan the interpreter's site-packages dirs.
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        hits = sorted(glob.glob(str(pathlib.Path(root) / "vllm-*.dist-info")))
        if hits:
            return pathlib.Path(hits[0])
    raise SystemExit("stamp_vllm_version: could not locate vllm dist-info")


def main(version: str, append: bool = False) -> None:
    dist_info = _find_vllm_dist_info()

    # 1) dist-info METADATA -> pip show / pip list / importlib.metadata.version
    meta = dist_info / "METADATA"
    lines = meta.read_text(encoding="utf-8").splitlines()
    idx = None
    current = ""
    for i, line in enumerate(lines):
        if line.startswith("Version:"):
            current = line[len("Version:") :].strip()
            idx = i
            break
    if idx is None:
        raise SystemExit(f"stamp_vllm_version: no 'Version:' line in {meta}")

    # --append: keep what's already there (e.g. the base image's "base(...)" stamp)
    # and tack the new segment on, so a release image reads "base(...)release(...)".
    new_version = (current + version) if append else version
    lines[idx] = f"Version: {new_version}"
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2) vllm/_version.py drives vllm.__version__ (this script is at
    #    <repo>/docker/, so _version.py is at <repo>/vllm/_version.py)
    version_py = pathlib.Path(__file__).resolve().parent.parent / "vllm" / "_version.py"
    version_py.write_text(
        f"__version__ = version = {new_version!r}\n"
        f"__version_tuple__ = version_tuple = (0, 0, 0)\n",
        encoding="utf-8",
    )

    print(f"stamp_vllm_version: {meta} -> {new_version!r} (append={append})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    append = "--append" in argv
    positional = [a for a in argv if a != "--append"]
    if len(positional) != 1 or not positional[0].strip():
        raise SystemExit(
            "usage: python docker/stamp_vllm_version.py <version-string> [--append]"
        )
    main(positional[0].strip(), append=append)
