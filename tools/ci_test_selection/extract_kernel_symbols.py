#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static artifact->kernel edge exporter (MVP D1, kernel-join prerequisite).

Enumerates CUDA kernel identities from built shared objects without a GPU or
CUDA toolkit, using the host-side __device_stub__ symbols that nvcc emits for
every runtime-API kernel launch. Each stub's mangled identifier embeds the
kernel's true device mangled name, which is the same identity nsys/CUPTI
reports at trace time, so joining on it avoids demangling drift entirely.

Known blind spots (quantified at trace time, not here):
  - runtime-JIT kernels (Triton, deep_gemm) have no stub and no .so mapping
  - cubins loaded via the driver API without nvcc stubs
  - kernels from external libraries (torch, cublas, cudnn, nccl) belong to
    pip-package provenance, not vllm build targets

Input: a wheel (.whl) or a directory containing .so files.
Output: JSONL edge rows on stdout in the frozen MVP edge shape:
  {"source_kind":"artifact","source":"elf-build-id:<id>","edge_kind":"defines_kernel",
   "destination_kind":"kernel","destination":"<mangled>",
   "artifact_build_id":"<gnu-build-id>","artifact_path":"<wheel path>"}
plus a summary table on stderr (per-artifact counts, cross-artifact
collisions, stripped/build-id status).

Usage:
    python3 extract_kernel_symbols.py <wheel-or-dir> [--out edges.jsonl]
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict

import regex as re

# nvcc host stubs: _Z<len>__device_stub__<kernel-id><param-mangling>, plus
# guard-variable (_ZGVZ) and internal-linkage (L) decorated variants. The
# <len>-bounded identifier minus the __device_stub__ prefix, re-prefixed with
# "_", is the kernel's device mangled name.
STUB = re.compile(r"_Z(?:GVZ)?L?(\d+)__device_stub__(\S+)")
PREFIX = "__device_stub__"


def gnu_build_id(path):
    out = subprocess.run(
        ["readelf", "-n", path], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(r"Build ID:\s*([0-9a-f]+)", out)
    return m.group(1) if m else None


def is_stripped(path):
    out = subprocess.run(
        ["file", "-b", path], capture_output=True, text=True, check=True
    ).stdout
    return "not stripped" not in out


def kernel_names(path):
    out = subprocess.run(
        ["nm", path], capture_output=True, text=True, check=True
    ).stdout
    names = set()
    for line in out.splitlines():
        m = STUB.search(line)
        if m:
            ident = (PREFIX + m.group(2))[: int(m.group(1))]
            names.add("_" + ident[len(PREFIX) :])
    return names


def iter_shared_objects(source):
    """Yield (relative_name, filesystem_path) for every .so in the input."""
    if os.path.isdir(source):
        for root, directories, files in os.walk(source):
            directories.sort()
            for f in sorted(files):
                if f.endswith(".so"):
                    p = os.path.join(root, f)
                    yield os.path.relpath(p, source), p
        return
    with zipfile.ZipFile(source) as z, tempfile.TemporaryDirectory() as tmp:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if info.filename.endswith(".so"):
                yield info.filename, z.extract(info, tmp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="wheel file or directory of .so files")
    ap.add_argument("--out", default="-", help="edge JSONL output (default stdout)")
    args = ap.parse_args(argv)

    src_sha = None
    if os.path.isfile(args.source):
        digest = hashlib.sha256()
        with open(args.source, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        src_sha = digest.hexdigest()

    artifacts, edges, name_to_artifacts = [], [], defaultdict(set)
    for rel, path in iter_shared_objects(args.source):
        bid = gnu_build_id(path)
        names = kernel_names(path)
        artifacts.append((rel, bid, is_stripped(path), len(names)))
        if not bid and names:
            raise SystemExit(f"CUDA artifact has no GNU build-id: {rel}")
        for name in sorted(names):
            identity = f"elf-build-id:{bid}"
            name_to_artifacts[name].add(identity)
            edge = {
                "source_kind": "artifact",
                "source": identity,
                "edge_kind": "defines_kernel",
                "destination_kind": "kernel",
                "destination": name,
                "artifact_path": rel,
            }
            if bid:
                edge["artifact_build_id"] = bid
            edges.append(edge)
    if not artifacts:
        raise SystemExit(f"no shared objects found in {args.source}")

    if args.out == "-":
        for edge in edges:
            print(json.dumps(edge, sort_keys=True))
    else:
        output = pathlib.Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            for edge in edges:
                stream.write(json.dumps(edge, sort_keys=True) + "\n")
        temporary.replace(output)

    err = sys.stderr
    if src_sha:
        print(f"# source sha256: {src_sha}", file=err)
    print(f"# {'artifact':55s} {'build_id':8s} {'stripped':8s} kernels", file=err)
    for rel, bid, stripped, n in artifacts:
        print(
            f"# {rel:55s} {'yes' if bid else 'NO':8s} "
            f"{'YES' if stripped else 'no':8s} {n:7d}",
            file=err,
        )
    collisions = {k: sorted(v) for k, v in name_to_artifacts.items() if len(v) > 1}
    print(
        f"# unique kernels: {len(name_to_artifacts)}; "
        f"cross-artifact collisions: {len(collisions)}",
        file=err,
    )
    for name, arts in sorted(collisions.items()):
        print(f"# COLLISION {name} -> {arts}", file=err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
