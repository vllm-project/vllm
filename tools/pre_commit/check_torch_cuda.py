# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys

import regex as re

# --------------------------------------------------------------------------- #
# Regex: match `torch.cuda.xxx` but allow `torch.accelerator.xxx`
# --------------------------------------------------------------------------- #
_TORCH_CUDA_RE = re.compile(r"\btorch\.cuda\.empty_cache\b")


ALLOWED_FILES = {"tests/", "benchmarks/", "vllm/platforms/*"}


def get_diff(diff_type: str) -> str:
    if diff_type == "staged":
        return subprocess.check_output(
            ["git", "diff", "--cached", "--unified=0"], text=True
        )
    elif diff_type == "unstaged":
        return subprocess.check_output(["git", "diff", "--unified=0"], text=True)
    else:
        raise ValueError(f"Unknown diff_type: {diff_type}")


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if _TORCH_CUDA_RE.search(line):
                print(
                    f"{path}:{i}: "
                    "\033[91merror:\033[0m "  # red color
                    "Found torch.cuda API call"
                )
                return 1
    return 0


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if filename in ALLOWED_FILES:
            continue
        returncode |= scan_file(filename)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
