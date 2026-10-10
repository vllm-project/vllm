# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against the EAGLE3 aux-hidden-state method-name typo.

``SupportsEagle3`` declares ``get_eagle3_default_aux_hidden_state_layers`` and
every runtime caller (``gpu_model_runner`` / eagle3 utils) invokes exactly that
name. Several models historically overrode a misspelled
``get_eagle3_aux_hidden_state_layers`` (missing ``default``); because nothing
calls that name, those overrides were silently dead code. This test fails if the
misspelled name reappears anywhere under ``vllm/model_executor/models``.
"""

import ast
from pathlib import Path

import pytest

import vllm.model_executor.models as models_pkg
from vllm.model_executor.models.interfaces import SupportsEagle3

CORRECT_NAME = "get_eagle3_default_aux_hidden_state_layers"
MISSPELLED_NAME = "get_eagle3_aux_hidden_state_layers"


@pytest.mark.cpu_test
def test_interface_uses_correct_method_name():
    assert hasattr(SupportsEagle3, CORRECT_NAME)
    assert not hasattr(SupportsEagle3, MISSPELLED_NAME)


@pytest.mark.cpu_test
def test_no_model_defines_misspelled_eagle3_aux_layers_method():
    models_dir = Path(models_pkg.__file__).parent
    offenders = []
    for path in sorted(models_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == MISSPELLED_NAME
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"Found method(s) named {MISSPELLED_NAME!r} (missing 'default'). The "
        f"EAGLE3 runtime only ever calls {CORRECT_NAME!r}, so such overrides are "
        f"dead code. Rename them. Offenders: {offenders}"
    )
