# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import glob
import sys

# Only strip targeted libraries when checking prefix
TORCH_LIB_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    "torchvision=",
    "torchaudio=",
    # pyproject.toml
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
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
        *glob.glob("requirements/*.txt"),
        *glob.glob("requirements/*.in"),
        "pyproject.toml",
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
                        and "torch" not in line.lower()
                    ):
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
