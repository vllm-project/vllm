# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def test_precompiled_wheel_extracts_legacy_moe_extension() -> None:
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    setup_source = setup_py.read_text()

    assert '"vllm/_moe_C.abi3.so"' in setup_source
