# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
input_file = "requirements/test.in"
cleaned_output = "requirements/nightly_torch_test.txt"

keywords = ["torch", "torchaudio", "torchvision", "mamba_ssm"]

with open(input_file) as f:
    lines = f.readlines()

cleaned_lines = []
torch_lines = []
skip_next = False

for line in lines:
    line_lower = line.lower()

    if not skip_next:
        if any(keyword in line_lower for keyword in keywords):
            torch_lines.append(line)
            print(f"  Removed: {line.strip()}")
            skip_next = True
            continue
    else:
        if line.startswith(" ") or line.startswith("\t") or line.strip() == "":
            torch_lines.append(line)
            print(f"  Removed (context): {line.strip()}")
            continue
        else:
            skip_next = False

    if not skip_next:
        cleaned_lines.append(line)

# Write cleaned lines to new file
with open(cleaned_output, "w") as f:
    f.writelines(cleaned_lines)
print(f">>> Cleaned file written to {cleaned_output}")
