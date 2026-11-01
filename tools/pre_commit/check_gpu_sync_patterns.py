# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static detection of non_blocking CPU<->CUDA copies that silently sync.

vLLM #53491 adds a runtime checker that raises when a `non_blocking=True`
copy touches a CPU tensor that is not pinned or not densely laid out; the
CUDA driver silently stages such copies through pageable memory and blocks
the host, so the `non_blocking=True` hint is dropped. This script catches
the two most common source-level shapes of that bug at commit time:

    # (1) unpinned CPU tensor pushed H2D via .to / .cuda
    torch.from_numpy(x).to(device, non_blocking=True)
    torch.tensor([...], device="cpu").to(device, non_blocking=True)

    # (2) unpinned CPU source used in .copy_(..., non_blocking=True)
    gpu_buffer.copy_(torch.from_numpy(x), non_blocking=True)

Both shapes should either route through `async_tensor_h2d`
(vllm/utils/torch_utils.py) or set `pin_memory=PIN_MEMORY` on the CPU-side
construction.

Escape hatch: a `# gpu-sync-ok: <reason>` comment on the line where the call
starts silences the check.
"""

import ast
import sys

# Torch tensor constructors that produce an unpinned CPU tensor by default.
# `torch.from_numpy` never pins; the others need explicit `pin_memory=True`
# (or a truthy `pin_memory=<var>`) to end up pinned.
_ALWAYS_UNPINNED = frozenset({"from_numpy"})
_UNPINNED_UNLESS_KWARG = frozenset(
    {"tensor", "zeros", "empty", "ones", "full", "arange"}
)

_PRAGMA = "gpu-sync-ok"


def _is_torch_ctor(expr: ast.AST, names: frozenset[str]) -> bool:
    """Return True when `expr` is `torch.<name>(...)` for name in `names`."""
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in names
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _kwarg_is_truthy(call: ast.Call, key: str) -> bool:
    """Return True when `call` has `key=<truthy>` (constant True or non-`False`
    name reference like `PIN_MEMORY`)."""
    for kw in call.keywords:
        if kw.arg != key:
            continue
        value = kw.value
        if isinstance(value, ast.Constant):
            return bool(value.value)
        # A bare Name (e.g. PIN_MEMORY) is treated as opt-in.
        if isinstance(value, ast.Name):
            return True
        # Any other expression: treat as truthy (best-effort, we don't want
        # false positives on `pin_memory=some_flag`).
        return True
    return False


def _kwarg_equals(call: ast.Call, key: str, value: object) -> bool:
    """Return True when `call` has `key=<constant value>`."""
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant):
            return kw.value.value == value
    return False


def _produces_unpinned_cpu_tensor(expr: ast.AST) -> bool:
    """Best-effort: does `expr` construct an unpinned CPU tensor?"""
    if _is_torch_ctor(expr, _ALWAYS_UNPINNED):
        return True
    if _is_torch_ctor(expr, _UNPINNED_UNLESS_KWARG):
        assert isinstance(expr, ast.Call)
        if _kwarg_is_truthy(expr, "pin_memory"):
            return False
        # A tensor constructed directly on a non-CPU device isn't the CPU
        # staging bug we are chasing. Treat `device=<anything except the
        # literal string "cpu">` as non-CPU (a bare Name like `self.device`
        # is almost always a device, and a false negative there is safer
        # than a false positive that flags perfectly fine on-device work).
        for kw in expr.keywords:
            if kw.arg == "device":
                # Explicit `device="cpu"` keeps this as an unpinned CPU
                # construction; anything else means the tensor lands on a
                # non-CPU device and is not our concern.
                return (
                    isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                    and kw.value.value == "cpu"
                )
        return True
    return False


def _targets_cuda(call: ast.Call) -> bool:
    """Rough: does `.to(...)` / `.cuda(...)` land the tensor on CUDA?"""
    func = call.func
    assert isinstance(func, ast.Attribute)
    if func.attr == "cuda":
        return True
    # `.to(device=..., ...)` — device kwarg wins.
    for kw in call.keywords:
        if kw.arg == "device":
            return _value_looks_like_device(kw.value)
    # `.to(<positional>, ...)` — first positional arg is the device.
    if call.args:
        return _value_looks_like_device(call.args[0])
    return False


def _value_looks_like_device(node: ast.AST) -> bool:
    """`node` is a device argument. Return False only when we are sure it is
    'cpu' — anything else (name reference, attribute like `self.device`,
    formatted string) is assumed to be a CUDA-like destination."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() != "cpu"
    return True


class SyncPatternChecker(ast.NodeVisitor):
    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.violations: list[tuple[int, str]] = []

    def _call_muted(self, call: ast.Call) -> bool:
        """Check the whole call span for a `gpu-sync-ok` comment.

        Formatters routinely break a long call across lines and land the
        trailing comment on the closing paren line, so checking only the
        call's opening line loses the pragma. Iterate `lineno`..`end_lineno`
        inclusive (Python 3.8+ populates `end_lineno`); if either is out of
        range the check falls back to a no-mute rather than crashing.
        """
        start = getattr(call, "lineno", None)
        end = getattr(call, "end_lineno", None) or start
        if start is None:
            return False
        for lineno in range(start, end + 1):
            if (
                1 <= lineno <= len(self.source_lines)
                and _PRAGMA in self.source_lines[lineno - 1]
            ):
                return True
        return False

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in ("to", "cuda"):
                self._check_h2d(node)
            elif func.attr == "copy_":
                self._check_copy(node)
        self.generic_visit(node)

    def _check_h2d(self, call: ast.Call) -> None:
        if not _kwarg_equals(call, "non_blocking", True):
            return
        func = call.func
        assert isinstance(func, ast.Attribute)
        # Only flag when the destination is CUDA (or unknown, which we
        # conservatively treat as CUDA). Explicit `.to("cpu")` is safe.
        if not _targets_cuda(call):
            return
        if not _produces_unpinned_cpu_tensor(func.value):
            return
        if self._call_muted(call):
            return
        self.violations.append(
            (
                call.lineno,
                (
                    "unpinned CPU tensor pushed via `non_blocking=True`; the "
                    "CUDA driver stages such copies through pageable memory "
                    "and blocks the host (see #53491). Use "
                    "`async_tensor_h2d(...)` from `vllm.utils.torch_utils` or "
                    "pass `pin_memory=PIN_MEMORY` on the CPU-side "
                    "construction. Suppress with a `# gpu-sync-ok: <reason>` "
                    "comment on this line."
                ),
            )
        )

    def _check_copy(self, call: ast.Call) -> None:
        if not _kwarg_equals(call, "non_blocking", True):
            return
        if not call.args:
            return
        source = call.args[0]
        if not _produces_unpinned_cpu_tensor(source):
            return
        if self._call_muted(call):
            return
        self.violations.append(
            (
                call.lineno,
                (
                    "unpinned CPU source in `.copy_(..., non_blocking=True)`; "
                    "same silent host stall as above (see #53491). Materialize "
                    "the source via `np_to_pinned_tensor(...)` from "
                    "`vllm.utils.torch_utils` or set `pin_memory=PIN_MEMORY` "
                    "on the construction. Suppress with a "
                    "`# gpu-sync-ok: <reason>` comment on this line."
                ),
            )
        )


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return 0
    lines = source.splitlines()
    checker = SyncPatternChecker(path, lines)
    checker.visit(tree)
    for lineno, msg in checker.violations:
        print(f"{path}:{lineno}: \033[91merror:\033[0m {msg}")
    return 1 if checker.violations else 0


def main() -> int:
    rc = 0
    for path in sys.argv[1:]:
        rc |= scan_file(path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
