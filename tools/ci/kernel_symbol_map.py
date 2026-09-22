#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which source file produced each GPU kernel symbol in this build.

Runs inside the csrc-build image stage, after the CMake build, and reads the
objects it left behind. For every compiled object it records the source file,
the CMake target, the headers the compiler reported it depended on (from
ninja's dependency log), and, for device objects, the kernel entry symbols
`cuobjdump` finds in the embedded fatbin. Those mangled names are exactly
what a CUPTI kernel recorder sees at run time, so CI can join "this step
launched kernel K" with "K came from csrc/foo.cu" and select tests for a
kernel change by evidence instead of by image membership.

    kernel_symbol_map.py --build-root build --source-root . \\
        --out dist/kernel_symbol_map.json.gz

Never fails the image build. If there is nothing to read (a precompiled-wheel
build, a missing cuobjdump) it writes a map with an empty object list and a
`reason`, which the consumer treats as "no evidence", not as "no kernels".

Output (gzipped JSON):

    {
      "version": 1,
      "commit": "...",              # from --commit or the environment
      "cuda": "13.0",               # nvcc release when found
      "generated_at": "2026-09-19T...",
      "build_dirs": ["build/temp.linux-x86_64-cpython-312"],
      "incomplete": false,          # true when any object's extraction failed;
                                    # objects is then [] and reason is set
      "errors": [],                 # [{"object", "source", "error"}, ...]
      "objects": [
        {
          "source": "csrc/libtorch_stable/fused_qknorm_rope_kernel.cu",
          "target": "_C",
          "object": "CMakeFiles/_C.dir/csrc/.../fused_qknorm_rope_kernel.cu.o",
          "device": true,
          "symbols": ["_Z21fusedQKNormRopeKernelI...", ...],
          "deps": ["csrc/libtorch_stable/fused_qknorm_rope_kernel.cu",
                   "csrc/cuda_compat.h", ...]
          # "error": "..."          # present when cuobjdump failed on it
        },
        ...
      ],
      "stats": {...},
      "reason": "..."               # only when the map is empty
    }

The consumer's contract is deliberately simple: an empty `objects` list with
a `reason` means "no evidence for any file", fall back to the static rule.
A partial map would need every consumer to know which files the unreadable
objects touched, and one that forgot would drop tests that a kernel change
can reach. So the producer never publishes a partial map: if any object is
still unreadable after a retry, `objects` is emptied, `incomplete` is set
and `errors` names the objects so the failure can be diagnosed.

Paths are relative to --source-root when the file lives inside it, otherwise
absolute (FetchContent sources under the build directory, CUDA headers).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import regex as re

DEVICE_SUFFIXES = (".cu.o", ".hip.o")
SOURCE_SUFFIXES = (".cu", ".hip", ".cpp", ".cc", ".cxx", ".c")
DEPS_HEADER = re.compile(r"^(\S+): #deps (\d+), deps mtime \d+ \((VALID|STALE)\)$")
# FetchContent sub-builds nest their own CMakeFiles/ under _deps/<name>-build/.
OBJ_DIR = re.compile(r"(?:^|/)CMakeFiles/([^/]+)\.dir/(.+)\.o$")


def log(msg: str) -> None:
    print(f"kernel_symbol_map: {msg}", flush=True)


def find_tool(name: str, *candidates: str) -> str | None:
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return shutil.which(name)


def cuda_release(nvcc: str | None) -> str:
    if not nvcc:
        return os.environ.get("CUDA_VERSION", "")
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
        m = re.search(r"release (\d+\.\d+)", out)
        return m.group(1) if m else os.environ.get("CUDA_VERSION", "")
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("CUDA_VERSION", "")


def find_build_dirs(build_root: Path) -> list[Path]:
    """Directories holding a top-level build.ninja (one per CMake build tree)."""
    return sorted(
        {
            p.parent
            for p in build_root.rglob("build.ninja")
            if (p.parent / ".ninja_deps").exists() or (p.parent / "CMakeFiles").is_dir()
        }
    )


def ninja_deps(ninja: str | None, build_dir: Path) -> dict[str, list[str]]:
    """Object (relative to build_dir) -> paths the compiler reported it read."""
    if not ninja or not (build_dir / ".ninja_deps").exists():
        return {}
    try:
        out = subprocess.run(
            [ninja, "-C", str(build_dir), "-t", "deps"],
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log(f"ninja -t deps failed in {build_dir}: {e}")
        return {}
    deps: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in out.splitlines():
        m = DEPS_HEADER.match(line)
        if m:
            cur = deps.setdefault(m.group(1), [])
            continue
        if cur is not None and line.startswith("    "):
            cur.append(line.strip())
        elif not line.strip():
            cur = None
    return deps


def parse_symbols(text: str) -> list[str]:
    """Kernel entry points from `cuobjdump -symbols` output.

    Lines look like `STT_FUNC  STB_GLOBAL  STO_ENTRY  _Z21fusedQKNormRope...`.
    Entry points are what the driver launches, so they are what a recorder
    sees. If a toolchain prints no STO_ENTRY marker, fall back to every global
    function symbol; over-attribution is the safe direction.
    """
    entries: set[str] = set()
    globals_: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "STT_FUNC":
            continue
        if parts[2] == "STO_ENTRY":
            entries.add(parts[3])
        elif parts[1] == "STB_GLOBAL":
            globals_.add(parts[3])
    return sorted(entries or globals_)


def device_symbols(cuobjdump: str, obj: Path) -> tuple[list[str], str | None]:
    """(symbols, error). An error means the object is unknown, not kernel-free."""
    syms, err = _cuobjdump_symbols(cuobjdump, obj)
    if err:  # transient failures happen on large fatbins; one retry is cheap
        syms, err = _cuobjdump_symbols(cuobjdump, obj)
    return syms, err


def _cuobjdump_symbols(cuobjdump: str, obj: Path) -> tuple[list[str], str | None]:
    try:
        r = subprocess.run(
            [cuobjdump, "-symbols", str(obj)],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return [], "timeout"
    except OSError as e:
        return [], str(e)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        return [], (msg[-1] if msg else f"exit {r.returncode}")
    return parse_symbols(r.stdout), None


def relpath(p: str | Path, source_root: Path) -> str:
    path = Path(p)
    try:
        return path.resolve().relative_to(source_root).as_posix()
    except ValueError:
        return path.as_posix()


def source_of(
    obj_rel: str, deps: list[str], build_dir: Path, source_root: Path
) -> tuple[str, str]:
    """(source path, cmake target) for an object.

    The dependency log lists the source first, which is the reliable answer.
    Failing that, CMake's object layout `CMakeFiles/<target>.dir/<source>.o`
    names it, with `__/` segments standing for `..`.
    """
    target = ""
    m = OBJ_DIR.search(obj_rel)  # search: sub-builds nest CMakeFiles/
    if m:
        target = m.group(1)
    for d in deps:
        if d.endswith(SOURCE_SUFFIXES):
            return relpath(
                d if os.path.isabs(d) else build_dir / d, source_root
            ), target
    if m:
        guess = m.group(2).replace("__/", "../")
        if guess.endswith(SOURCE_SUFFIXES):
            for base in (source_root, build_dir):
                cand = base / guess
                if cand.exists():
                    return relpath(cand, source_root), target
            return guess, target
    return obj_rel, target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-root", type=Path, default=Path("build"))
    ap.add_argument("--source-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--commit", default="", help="commit the build is of")
    a = ap.parse_args()
    t0 = time.time()
    source_root = a.source_root.resolve()

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuobjdump = a.cuobjdump or find_tool(
        "cuobjdump", f"{cuda_home}/bin/cuobjdump", "/usr/local/cuda/bin/cuobjdump"
    )
    nvcc = find_tool("nvcc", f"{cuda_home}/bin/nvcc", "/usr/local/cuda/bin/nvcc")
    ninja = shutil.which("ninja")

    result: dict = {
        "version": 1,
        "commit": a.commit
        or os.environ.get("BUILDKITE_COMMIT")
        or os.environ.get("VLLM_BUILD_COMMIT")
        or "",
        "cuda": cuda_release(nvcc),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "build_dirs": [],
        "incomplete": False,
        "errors": [],
        "objects": [],
        "stats": {},
    }

    build_dirs = find_build_dirs(a.build_root) if a.build_root.exists() else []
    result["build_dirs"] = [relpath(b, source_root) for b in build_dirs]
    if not build_dirs:
        result["reason"] = (
            f"no CMake build tree under {a.build_root} (precompiled wheel build?)"
        )
    elif not cuobjdump:
        result["reason"] = "cuobjdump not found"
    log(
        f"build dirs: {result['build_dirs'] or 'none'}; "
        f"cuobjdump={cuobjdump}; ninja={ninja}"
    )

    jobs: list[tuple[Path, str, Path]] = []  # (build_dir, obj_rel, obj_path)
    all_deps: dict[Path, dict[str, list[str]]] = {}
    if "reason" not in result:
        for bd in build_dirs:
            all_deps[bd] = ninja_deps(ninja, bd)
            for obj in bd.rglob("*.o"):
                rel = obj.relative_to(bd).as_posix()
                # CMake's compiler-identification probes are objects too.
                if "CompilerId" in rel:
                    continue
                if "/CMakeFiles/" in f"/{rel}" or rel.startswith("CMakeFiles/"):
                    jobs.append((bd, rel, obj))
        n_deps = sum(len(d) for d in all_deps.values())
        log(f"{len(jobs)} objects, {n_deps} dependency records")

    def work(job):
        bd, rel, obj = job
        deps = all_deps.get(bd, {}).get(rel, [])
        source, target = source_of(rel, deps, bd, source_root)
        device = rel.endswith(DEVICE_SUFFIXES)
        syms, err = device_symbols(cuobjdump, obj) if device else ([], None)
        rel_deps = sorted(
            {relpath(d if os.path.isabs(d) else bd / d, source_root) for d in deps}
        )
        entry = {
            "source": source,
            "target": target,
            "object": rel,
            "device": device,
            "symbols": syms,
            "deps": rel_deps,
        }
        if err:
            entry["error"] = err
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        result["objects"] = sorted(
            ex.map(work, jobs), key=lambda e: (e["source"], e["target"])
        )

    # An object cuobjdump could not read is unknown, not kernel-free. Say so
    # at the top so a consumer never has to scan for it, and name the objects
    # so it can widen the "unknown" set to every file they touch.
    result["errors"] = [
        {"object": e["object"], "source": e["source"], "error": e["error"]}
        for e in result["objects"]
        if "error" in e
    ]
    result["incomplete"] = bool(result["errors"])
    for e in result["errors"][:5]:
        log(f"cuobjdump failed on {e['object']}: {e['error']}")
    if result["incomplete"]:
        # Publish nothing rather than something partial: see the docstring.
        result["objects"] = []
        result["reason"] = (
            f"{len(result['errors'])} objects unreadable by cuobjdump, e.g. "
            f"{result['errors'][0]['object']}: {result['errors'][0]['error']}"
        )

    result["stats"] = {
        "objects": len(jobs),
        "device_objects": sum(1 for e in result["objects"] if e["device"]),
        "objects_with_symbols": sum(1 for e in result["objects"] if e["symbols"]),
        "symbols": sum(len(e["symbols"]) for e in result["objects"]),
        "cuobjdump_errors": len(result["errors"]),
        "seconds": round(time.time() - t0, 1),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    note = f"; reason: {result['reason']}" if "reason" in result else ""
    log(f"wrote {a.out} ({a.out.stat().st_size // 1024} KiB): {result['stats']}{note}")

    # One line a human can check in the build log.
    probe = [
        e
        for e in result["objects"]
        if any("fusedQKNormRopeKernel" in s for s in e["symbols"])
    ]
    if probe:
        log(f"probe: fusedQKNormRopeKernel -> {sorted({e['source'] for e in probe})}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never fail the image build
        log(f"unexpected error, continuing without a map: {e!r}")
        sys.exit(0)
