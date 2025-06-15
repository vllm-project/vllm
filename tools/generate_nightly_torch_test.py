# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generates specialized requirements files for nightly PyTorch testing.

This script reads the main test requirements input file (`requirements/test.in`)
and splits its content into two files:
1.  `requirements/nightly_torch_test.txt`: Contains dependencies
except PyTorch-related.
2.  `torch_nightly_test.txt`: Contains only PyTorch-related packages.
"""

input_file = "requirements/test.in"
output_file = "requirements/nightly_torch_test.txt"

keywords = ["torch", "torchaudio", "torchvision", "mamba_ssm"]

with open(input_file) as f:
    lines = f.readlines()

cleaned = []
# clean all immediately following indented or empty lines to
skip_next = False

for line in lines:
    if skip_next:
        if line.startswith((" ", "\t")) or line.strip() == "":
            continue
        skip_next = False

    if any(k in line.lower() for k in keywords):
        skip_next = True
        continue

    cleaned.append(line)

with open(output_file, "w") as f:
    f.writelines(cleaned)
