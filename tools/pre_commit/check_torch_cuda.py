# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import sys

_BANNED_CALLS = {
    "cuda_device_count_stateless": "Use current_platform.device_count() instead.",
    "current_platform.mem_get_info": "Use torch.accelerator.get_memory_info() instead.",
}


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def check_source(source: str, filepath: str = "<string>") -> list[str]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func)
        if name in _BANNED_CALLS:
            violations.append(
                f"{filepath}:{node.lineno}: Found {name}() API call. "
                f"{_BANNED_CALLS[name]}"
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
    violations = [v for filepath in sys.argv[1:] for v in check_file(filepath)]
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
