# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import regex as re

# --------------------------------------------------------------------------- #
# Regex: match `torch.cuda.xxx` but allow `torch.accelerator.xxx`
# --------------------------------------------------------------------------- #
_DEVICE_COUNT_PATTERNS = [
    r"\bcuda_device_count_stateless\b",
]

ALLOWED_FILES = {
    "vllm/platforms/",
    "vllm/utils/torch_utils.py",
    "tests/distributed/test_utils.py",
    "tools/pre_commit/check_device_count.py",
}


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for pattern in _DEVICE_COUNT_PATTERNS:
        for match in re.finditer(pattern, content, re.MULTILINE):
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                "Found cuda_device_count_stateless. Please refer RFC "
                "https://github.com/vllm-project/vllm/issues/37849, use "
                "current_platform.device_count() instead."
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
