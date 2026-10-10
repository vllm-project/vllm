#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#
# A command line tool for running pytorch's hipify preprocessor on CUDA
# source files.
#
# See https://github.com/ROCm/hipify_torch
# and <torch install dir>/utils/hipify/hipify_python.py
#

import argparse
import json
import os
import shutil
from pathlib import Path

from torch.utils.hipify.hipify_python import get_hip_file_path, hipify


def _expected_hip_build_path(source_abs: str, output_directory: str) -> str:
    """Match torch.utils.hipify.hipify_python.preprocessor fout_path naming."""
    rel = os.path.relpath(source_abs, output_directory)
    return os.path.abspath(
        os.path.join(
            output_directory, get_hip_file_path(rel, is_pytorch_extension=True)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Project directory where all the source + include files live.
    parser.add_argument(
        "-p",
        "--project_dir",
        help="The project directory.",
    )

    # Directory where hipified files are written.
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    # Source files to convert.
    parser.add_argument(
        "sources", help="Source files to hipify.", nargs="*", default=[]
    )
    parser.add_argument(
        "--manifest", help="Write generated-to-original source/header provenance."
    )

    args = parser.parse_args()

    # Limit include scope to project_dir only
    includes = [os.path.join(args.project_dir, "*")]

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    provenance = {
        str(output_dir / source.relative_to(project_dir)): str(source)
        for source in project_dir.rglob("*")
        if source.is_file()
    }
    source_copies = {
        os.path.abspath(source): str(
            output_dir / Path(source).resolve().relative_to(project_dir)
        )
        for source in args.sources
    }
    extra_files = list(source_copies.values())

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)

    hipify_result = hipify(
        project_directory=args.project_dir,
        output_directory=args.output_dir,
        # Hipify resolves quoted includes next to the including file first; vLLM
        # uses paths relative to csrc/ (e.g. "libtorch_stable/torch_utils.h"
        # from quantization/w8a8/fp8/*.cu). Without an include root here, those
        # headers are never found and are not hipified or rewritten in dependents.
        header_include_dirs=["."],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    copied_sources = dict(provenance)
    for source, result in hipify_result.items():
        if result.hipified_path is not None:
            original = copied_sources.get(
                os.path.abspath(source), os.path.abspath(source)
            )
            provenance[os.path.abspath(result.hipified_path)] = original

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        copied = source_copies[s_abs]
        if copied in hipify_result and hipify_result[copied].hipified_path is not None:
            path = hipify_result[copied].hipified_path
            # PyTorch skips writing when is_pytorch_extension and text unchanged;
            # hipified_path then stays *.cu. CMake expects *.hip under output_dir.
            if s_abs.endswith(".cu") and path.endswith(".cu"):
                dest = _expected_hip_build_path(copied, args.output_dir)
                if os.path.normpath(path) != os.path.normpath(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(path, dest)
                hipified_s_abs = dest
            else:
                hipified_s_abs = path
        else:
            hipified_s_abs = copied
        hipified_sources.append(hipified_s_abs)
        provenance[os.path.abspath(hipified_s_abs)] = s_abs

    assert len(hipified_sources) == len(args.sources)

    manifest = (
        Path(args.manifest) if args.manifest else output_dir / "hipify-source-map.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)

    # Print hipified source files.
    print("\n".join(hipified_sources))
