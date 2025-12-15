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


def is_allowed_file(current_file: str) -> bool:
    return current_file in ALLOWED_FILES


def is_forbidden_torch_cuda_api(line: str) -> bool:
    stripped = line.strip()
    return bool(_TORCH_CUDA_RE.match(stripped))


def parse_diff(diff: str) -> list[str]:
    violations = []
    current_file = None
    current_lineno = None
    skip_allowed_file = False

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            skip_allowed_file = is_allowed_file(current_file)
        elif skip_allowed_file:
            continue
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_lineno = int(match.group(1)) - 1  # next "+ line" is here
        elif line.startswith("+") and not line.startswith("++"):
            current_lineno += 1
            code_line = line[1:]
            if is_forbidden_torch_cuda_api(code_line):
                violations.append(
                    f"{current_file}:{current_lineno}: {code_line.strip()}"
                )
    return violations


def get_diff(diff_type: str) -> str:
    if diff_type == "staged":
        return subprocess.check_output(
            ["git", "diff", "--cached", "--unified=0"], text=True
        )
    elif diff_type == "unstaged":
        return subprocess.check_output(["git", "diff", "--unified=0"], text=True)
    else:
        raise ValueError(f"Unknown diff_type: {diff_type}")


def main():
    all_violations = []
    for diff_type in ["staged", "unstaged"]:
        try:
            diff_output = get_diff(diff_type)
            violations = parse_diff(diff_output)
            all_violations.extend(violations)
        except subprocess.CalledProcessError as e:
            print(f"[{diff_type}] Git diff failed: {e}", file=sys.stderr)

    if all_violations:
        print(
            "❌ Forbidden direct `torch.cuda.empty_cache` detected."
            " ➤ Use `torch.accelerator.empty_cache` instead.\n"
        )
        for v in all_violations:
            print(f"❌ {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
