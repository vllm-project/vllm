#!/usr/bin/env python3

import argparse
import shutil
import os

from torch.utils.hipify.hipify_python import hipify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--build_dir",
        help="The build directory.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    parser.add_argument(
        "-i",
        "--include_dir",
        help="Include directory",
        action="append",
        default=[],
    )

    parser.add_argument("sources",
                        help="Source files to hipify.",
                        nargs="*",
                        default=[])

    args = parser.parse_args()

    # limit scope to build_dir only
    includes = [os.path.join(args.build_dir, '*')]

    extra_files = [os.path.abspath(s) for s in args.sources]

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.build_dir, args.output_dir, dirs_exist_ok=True)

    hipify_result = hipify(project_directory=args.build_dir,
                           output_directory=args.output_dir,
                           header_include_dirs=[],
                           includes=includes,
                           extra_files=extra_files,
                           show_detailed=True,
                           is_pytorch_extension=True,
                           hipify_extra_files_only=True)

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (hipify_result[s_abs].hipified_path if
                          (s_abs in hipify_result
                           and hipify_result[s_abs].hipified_path is not None)
                          else s_abs)
        if True:
            hipified_sources.append(hipified_s_abs)
        else:
            hipified_sources.append(
                os.path.relpath(
                    hipified_s_abs,
                    os.path.abspath(os.path.join(args.build_dir, os.pardir))))

    assert (len(hipified_sources) == len(args.sources))

    # Print hipified source files.
    print("\n".join(hipified_sources))
