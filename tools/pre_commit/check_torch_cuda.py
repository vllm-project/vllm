# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import regex as re

# --------------------------------------------------------------------------- #
# Regex: match `torch.cuda.xxx` but allow `torch.accelerator.xxx`
# --------------------------------------------------------------------------- #
_TORCH_CUDA_PATTERNS = [
    r"\btorch\.cuda\.empty_cache\b",
]

ALLOWED_FILES = {"vllm/platforms/", "vllm/device_allocator/"}


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for pattern in _TORCH_CUDA_PATTERNS:
        for match in re.finditer(pattern, content, re.MULTILINE):
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                "Found torch.cuda API call"
            )
            return 1
    return 0


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if any(filename.startswith(prefix) for prefix in ALLOWED_FILES):
            continue
        returncode |= scan_file(filename)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
