# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard: the CPU-offload (gate-off) path must not import NIXL/ZMQ at module
load. NIXL support modules may import them; they are only loaded behind the
ec_enable_nixl gate."""

import ast
import pathlib

_CPU_DIR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "vllm/distributed/ec_transfer/ec_connector/cpu"
)
# Modules on the gate-off path — must have no top-level nixl/zmq/msgspec import.
_GATE_OFF_MODULES = ("scheduler.py", "connector.py", "worker.py", "common.py")
_FORBIDDEN = ("nixl", "zmq", "msgspec")


def _toplevel_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:  # top level only
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_gate_off_modules_have_no_toplevel_nixl_imports():
    offenders = []
    for name in _GATE_OFF_MODULES:
        path = _CPU_DIR / name
        for mod in _toplevel_imports(path):
            low = mod.lower()
            if any(f in low for f in _FORBIDDEN):
                offenders.append((name, mod))
    assert not offenders, f"top-level nixl/zmq/msgspec on gate-off path: {offenders}"
