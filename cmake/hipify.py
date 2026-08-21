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
import os
import shutil

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

    args = parser.parse_args()

    # Limit include scope to project_dir only
    includes = [os.path.join(args.project_dir, "*")]

    # Get absolute path for all source files.
    extra_files = [os.path.abspath(s) for s in args.sources]

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

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None:
            path = hipify_result[s_abs].hipified_path
            # PyTorch skips writing when is_pytorch_extension and text unchanged;
            # hipified_path then stays *.cu. CMake expects *.hip under output_dir.
            if s_abs.endswith(".cu") and path.endswith(".cu"):
                dest = _expected_hip_build_path(s_abs, args.output_dir)
                if os.path.normpath(path) != os.path.normpath(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(path, dest)
                hipified_s_abs = dest
            else:
                hipified_s_abs = path
        else:
            hipified_s_abs = s_abs
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(args.sources)

    # Print hipified source files.
    print("\n".join(hipified_sources))
