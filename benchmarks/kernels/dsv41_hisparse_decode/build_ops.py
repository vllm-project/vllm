# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build isolated research ops without replacing the installed vLLM extension."""

import argparse
import hashlib
import json
import os
import site
from pathlib import Path

import regex as re
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.repo / "csrc"
    names = [
        "hisparse_resolve_residency",
        "hisparse_gather_plan",
        "hisparse_gather_compact",
    ]
    pattern = re.compile(r"\b(" + "|".join(names) + r")\b")
    for name in ["hisparse_kernels.cu", "ops.h"]:
        text = pattern.sub(lambda m: "dsv41_" + m[0], (source / name).read_text())
        (args.output / name).write_text(text)
    bindings = (source / "torch_bindings.cpp").read_text()
    definitions = []
    for name in names:
        match = re.search(r'  ops.def\(\s*"' + name + r"\(.*?;\n", bindings, re.S)
        assert match is not None, name
        definitions.append(match[0])
    registration = '#include "ops.h"\n#include <torch/csrc/stable/library.h>\n'
    registration += "STABLE_TORCH_LIBRARY_FRAGMENT(_C_dsv41_hisparse, ops) {\n"
    registration += "".join(definitions) + "}\n"
    registration += "STABLE_TORCH_LIBRARY_IMPL(_C_dsv41_hisparse, CUDA, ops) {\n"
    registration += "".join(f'ops.impl("{n}", TORCH_BOX(&dsv41_{n}));\n' for n in names)
    registration += "}\n"
    (args.output / "bindings.cpp").write_text(registration)
    vendor = args.output / "cublas-headers"
    vendor.mkdir(exist_ok=True)
    for root in site.getsitepackages():
        for header in Path(root).glob("nvidia/**/cublas*.h"):
            (vendor / header.name).write_bytes(header.read_bytes())
    library = load(
        name="dsv41_hisparse_ops",
        sources=[
            str(args.output / "bindings.cpp"),
            str(args.output / "hisparse_kernels.cu"),
        ],
        extra_include_paths=[
            str(source),
            str(Path(os.environ["CUDA_HOME"]) / "include"),
            str(vendor),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )
    library = Path(library)
    receipt = dict(
        library=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest()
    )
    (args.output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
