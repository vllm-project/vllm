# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import glob
import sys

# Only strip targeted libraries when checking prefix.
# pyproject.toml dependencies can appear with double or single quotes and
# with or without spaces around the version operator, so we cover all variants.
TORCH_LIB_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    "torchvision=",
    "torchaudio=",
    # pyproject.toml – double-quoted, with/without space
    '"torch==',
    '"torch =',
    '"torchvision==',
    '"torchvision =',
    '"torchaudio==',
    '"torchaudio =',
    # pyproject.toml – single-quoted, with/without space
    "'torch==",
    "'torch =",
    "'torchvision==",
    "'torchvision =",
    "'torchaudio==",
    "'torchaudio =",
)

# Prefixes for torch-only mode: strip only the core torch package version pin,
# preserving torchvision and torchaudio as standalone runtime dependencies.
TORCH_ONLY_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    # pyproject.toml – double-quoted, with/without space
    '"torch==',
    '"torch =',
    # pyproject.toml – single-quoted, with/without space
    "'torch==",
    "'torch =",
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Strip torch lib requirements to use installed version."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prefix",
        action="store_true",
        help=(
            "Strip prefix matches only (default: False). Removes lines whose "
            "stripped content starts with torch=, torchvision=, or torchaudio= "
            "(and their pyproject.toml equivalents)."
        ),
    )
    mode.add_argument(
        "--torch-only",
        action="store_true",
        help=(
            "Strip only the core torch version pin, leaving torchvision and "
            "torchaudio version pins intact. Use this when the pre-installed "
            "torch must be reused but torchvision/torchaudio still need to be "
            "installed from requirements (e.g. GH200 arm64 builds)."
        ),
    )
    args = parser.parse_args(argv)

    if args.torch_only:
        active_prefixes = TORCH_ONLY_PREFIXES
    elif args.prefix:
        active_prefixes = TORCH_LIB_PREFIXES
    else:
        active_prefixes = None  # strip all lines containing 'torch'

    for file in (
        *glob.glob("requirements/*.txt"),
        *glob.glob("requirements/*.in"),
        "pyproject.toml",
    ):
        with open(file) as f:
            lines = f.readlines()
        if "torch" in "".join(lines).lower():
            with open(file, "w") as f:
                for line in lines:
                    if active_prefixes is not None:
                        keep = not line.lower().strip().startswith(active_prefixes)
                    else:
                        keep = "torch" not in line.lower()
                    if keep:
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
