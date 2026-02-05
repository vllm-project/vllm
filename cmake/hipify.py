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

from torch.utils.hipify.hipify_python import hipify

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

    # Search project root for includes (needed for files in subdirectories
    # that include headers from the project root).
    parser.add_argument(
        "--search-project-root",
        action="store_true",
        help="Add project root to header include search paths.",
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
        # When --search-project-root is set, include the project directory root
        # so that includes like #include "cuda_compat.h" from csrc/stable/
        # can find csrc/cuda_compat.h
        header_include_dirs=[""] if args.search_project_root else [],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(args.sources)

    # Print hipified source files.
    print("\n".join(hipified_sources))
