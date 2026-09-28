# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Meta-tests for vLLM IR op infrastructure.

Ensures all registered ops have input generators defined.
Per-op correctness tests live alongside their op definitions
(e.g. tests/kernels/ir/test_layernorm.py).
"""

import vllm.kernels  # noqa: F401 — registers provider implementations
from vllm.ir.op import IrOp


def test_all_ops_have_input_generator():
    missing = [name for name, op in IrOp.registry.items() if not op.has_input_generator]
    assert not missing, (
        f"IR ops without input generators: {missing}. "
        f"Register one with @ir.ops.<name>.register_input_generator"
    )
