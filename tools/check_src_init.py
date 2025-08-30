# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

VLLM_SRC = "./vllm"


def check_init_file_in_package(directory):
    """
    Check if a Python package directory contains __init__.py file.
    A directory is considered a Python package if it contains `.py` files
    and an `__init__.py` file.
    """
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Warning: Directory does not exist: {directory}")
        return False

    # If any .py file exists, we expect an __init__.py
    if any(f.endswith('.py') for f in files):
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.isfile(init_file):
            return False
    return True


def find_missing_init_dirs(src_dir):
    """
    Walk through the src_dir and return subdirectories missing __init__.py.
    """
    missing_init = set()
    for dirpath, _, _ in os.walk(src_dir):
        if not check_init_file_in_package(dirpath):
            missing_init.add(dirpath)
    return missing_init


def main():
    all_missing = set()

    for src in [VLLM_SRC]:
        missing = find_missing_init_dirs(src)
        all_missing.update(missing)

    if all_missing:
        print(
            "❌ Missing '__init__.py' files in the following " \
            "Python package directories:"
        )
        for pkg in sorted(all_missing):
            print(f" - {pkg}")
        sys.exit(1)
    else:
        print("✅ All Python packages have __init__.py files.")


if __name__ == "__main__":
    main()
