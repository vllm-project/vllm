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
    "torchcodec=",
    # pyproject.toml
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
    '"torchcodec =',
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
                continued = False
                for line in lines:
                    if not continued:
                        keep = (
                            not line.lower().strip().startswith(TORCH_LIB_PREFIXES)
                            if args.prefix
                            else "torch" not in line.lower()
                        )
                    if keep:
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())
                    continued = line.rstrip().endswith("\\")


if __name__ == "__main__":
    main(sys.argv[1:])
