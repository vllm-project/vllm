# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local conftest for the zaqeris_axiom tests."""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Drop the repo root from sys.path so the installed compiled vLLM is found
# first. pytest's `prepend` import mode inserts rootdir before importing us.
for _p in list(sys.path):
    if _p == REPO_ROOT or _p.startswith(REPO_ROOT + os.sep):
        sys.path.remove(_p)

print("[conftest] sys.path:", sys.path[:5], file=sys.stderr)

_OVERLAY = {
    "vllm.model_executor.models.zaqeris_axiom": (
        f"{REPO_ROOT}/vllm/model_executor/models/zaqeris_axiom.py"
    ),
    "vllm.model_executor.layers.quantization.zaqeris_m3": (
        f"{REPO_ROOT}/vllm/model_executor/layers/quantization/zaqeris_m3.py"
    ),
}


def _load_overlay(modname: str, path: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def pytest_configure(config):
    import vllm  # noqa: F401
    import vllm.model_executor.layers.quantization  # noqa: F401

    for modname, path in _OVERLAY.items():
        try:
            _load_overlay(modname, path)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[conftest] failed to overlay {modname}: {exc}", file=sys.stderr)

    from vllm.model_executor.models import registry as me_models
    from vllm.model_executor.models.zaqeris_axiom import (
        ZaqerisAxiomForCausalLM,
    )

    me_models.ModelRegistry.register_model(
        "ZaqerisAxiomForCausalLM", ZaqerisAxiomForCausalLM
    )
