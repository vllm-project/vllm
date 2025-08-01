# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generates specialized requirements files for nightly PyTorch testing.

This script reads the main test requirements input file (`requirements/test.in`)
and splits its content into two files:
1.  `requirements/nightly_torch_test.txt`: Contains dependencies
except PyTorch-related.
2.  `nightly_torch_test.txt.txt`: Contains only PyTorch-related packages.
"""

input_file = "requirements/test.in"
output_file = "requirements/nightly_torch_test.txt"

# white list of packages that are not compatible with
# PyTorch nightly directlywith pip install. Please add
# your package to this list if it is not compatible or
# make the dependency test fails. If you find the
# compatibile version, add the package in here, and put
# it in the nightly_torch_test_manual.txt.
white_list = [
    "torch", "torchaudio", "torchvision", "mamba_ssm", "schemathesis"
]

with open(input_file) as f:
    lines = f.readlines()

skip_next = False

filtered_lines = []
skip_next = False

for line in lines:
    stripped = line.strip()

    if skip_next:
        if line.startswith((" ", "\t")) or stripped == "":
            # Skip continuation lines
            continue
        skip_next = False

    if any(pkg in stripped.lower() for pkg in white_list):
        skip_next = True
        continue

    filtered_lines.append(line)

with open(output_file, "w") as f:
    f.writelines(filtered_lines)
