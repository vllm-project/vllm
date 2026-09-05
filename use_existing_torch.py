# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import glob
import re
import sys

# Only strip targeted libraries when checking prefix
TORCH_LIB_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    "torchvision=",
    "torchaudio=",
    "torchcodec=",
    # pyproject.toml
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
    '"torchcodec =',
)

# Matches a line that is exactly one of torch/torchvision/torchaudio,
# optionally followed by a version specifier (e.g. "torch==2.5.1") or a
# pyproject.toml-style quoted entry (e.g. '"torch>=2.5.1"'). Deliberately
# narrower than a plain substring check so packages like terratorch,
# open_clip_torch, or vector-quantize-pytorch, and comment lines like
# "# via torch", are left untouched.
TORCH_LIB_LINE_RE = re.compile(
    r'^\s*"?(torch|torchvision|torchaudio)"?\s*([=<>!~].*)?$', re.IGNORECASE
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Strip torch lib requirements to use installed version."
    )
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Strip prefix matches only (default: False)",
    )
    args = parser.parse_args(argv)

    for file in (
        *glob.glob("requirements/**/*.txt", recursive=True),
        *glob.glob("requirements/**/*.in", recursive=True),
        *glob.glob("pyproject.toml"),
    ):
        with open(file) as f:
            lines = f.readlines()
        if "torch" in "".join(lines).lower():
            with open(file, "w") as f:
                for line in lines:
                    if (
                        args.prefix
                        and not line.lower().strip().startswith(TORCH_LIB_PREFIXES)
                        or not args.prefix
                        and not TORCH_LIB_LINE_RE.match(line.strip())
                    ):
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
