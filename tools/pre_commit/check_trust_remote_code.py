# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint: detect hardcoded `trust_remote_code=True` inside vLLM.

Passing a literal `True` silently overrides the user's `--trust-remote-code`
setting, so a malicious model repository can execute code even when the user
opted out (CVE-2026-27893):

    AutoModel.from_config(cfg, trust_remote_code=True)          # BUG

Pass the user's setting through instead:

    AutoModel.from_config(cfg, trust_remote_code=model_config.trust_remote_code)

Being AST-based, this check ignores docstrings and comments. If a call is
genuinely safe, suppress it with a reason on the same line:

    load(path, trust_remote_code=True)  # trust-remote-code: ok (reason)
"""

import ast
import sys

SUPPRESS_MARKER = "# trust-remote-code: ok"


def check_source(source: str, filepath: str = "<string>") -> list[str]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    lines = source.splitlines()
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "trust_remote_code"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                and SUPPRESS_MARKER not in lines[kw.value.lineno - 1]
            ):
                violations.append(
                    f"{filepath}:{kw.value.lineno}: hardcoded "
                    "`trust_remote_code=True` overrides the user's setting"
                )
    return violations


def check_file(filepath: str) -> list[str]:
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    return check_source(source, filepath)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_trust_remote_code.py <file> ...", file=sys.stderr)
        return 1

    violations = [v for filepath in sys.argv[1:] for v in check_file(filepath)]
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
