# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With ec_enable_nixl off, ECCPUConnector must not import NIXL or the P2P
transport modules. ECCPUScheduler._setup_nixl imports them lazily."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cpu_test

_CPU = "vllm.distributed.ec_transfer.ec_connector.cpu"

GATE_OFF_MODULES = [
    f"{_CPU}.connector",
    f"{_CPU}.common",
    f"{_CPU}.scheduler",
    f"{_CPU}.worker",
]

P2P_MODULES = [
    f"{_CPU}.control.zmq",
    f"{_CPU}.data.nixl",
    f"{_CPU}.protocol",
    f"{_CPU}.session",
]


def test_gate_off_path_does_not_import_nixl():
    # Fresh interpreter, since other tests in the session import NIXL.
    script = textwrap.dedent(f"""
        import importlib
        import importlib.abc
        import sys

        attempted = []

        class RecordNixl(importlib.abc.MetaPathFinder):
            # Catches the attempt even when NIXL is not installed and the
            # caller swallows the ImportError.
            def find_spec(self, name, path, target=None):
                if name.partition(".")[0] in ("nixl", "nixl_rocm"):
                    attempted.append(name)
                return None

        sys.meta_path.insert(0, RecordNixl())
        for mod in {GATE_OFF_MODULES!r}:
            importlib.import_module(mod)

        assert not attempted, f"NIXL imported: {{attempted}}"
        leaked = [m for m in {P2P_MODULES!r} if m in sys.modules]
        assert not leaked, f"P2P modules imported: {{leaked}}"
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
