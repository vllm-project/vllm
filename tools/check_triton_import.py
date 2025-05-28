# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import regex as re

FORBIDDEN_IMPORT_RE = re.compile(r"^(from|import)\s+triton(\s|\.|$)")

# the way allowed to import triton
ALLOWED_LINES = {
    "from vllm.triton_utils import triton",
    "from vllm.triton_utils import tl",
    "from vllm.triton_utils import tl, triton",
}


def is_forbidden_import(line: str) -> bool:
    stripped = line.strip()
    return bool(
        FORBIDDEN_IMPORT_RE.match(stripped)) and stripped not in ALLOWED_LINES


def parse_diff(diff: str) -> list[str]:
    violations = []
    current_file = None
    current_lineno = None

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_lineno = int(
                    match.group(1)) - 1  # next "+ line" is here
        elif line.startswith("+") and not line.startswith("++"):
            current_lineno += 1
            code_line = line[1:]
            if is_forbidden_import(code_line):
                violations.append(
                    f"{current_file}:{current_lineno}: {code_line.strip()}")
    return violations


def get_diff(diff_type: str) -> str:
    if diff_type == "staged":
        return subprocess.check_output(
            ["git", "diff", "--cached", "--unified=0"], text=True)
    elif diff_type == "unstaged":
        return subprocess.check_output(["git", "diff", "--unified=0"],
                                       text=True)
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
        print("❌ Forbidden direct `import triton` detected."
              " ➤ Use `from vllm.triton_utils import triton` instead.\n")
        for v in all_violations:
            print(f"❌ {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
