# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the PDV verify kernel.

The shipping UltraQuant decode stays the default. Set
``VLLM_ULTRAQUANT_PDV_EXPERIMENT=hoist_k`` to build the PDV 1.1 kernel,
which reads each KV tile once and scores every verify token with it.
"""

from __future__ import annotations

import importlib
import os

_MODULES = {
    "hoist_k": "vllm.v1.attention.ops.flydsl_kernels.pdv_flydsl_decode_hd256_hoist_k",
}


def pdv_experiment_name() -> str:
    """Return the selected experiment. Empty means the shipping kernel."""
    return os.environ.get("VLLM_ULTRAQUANT_PDV_EXPERIMENT", "").strip()


def load_pdv_module():
    """Import the PDV module for the current experiment."""
    name = pdv_experiment_name()
    module_name = _MODULES.get(name)
    if module_name is None:
        known = ", ".join(sorted(_MODULES))
        raise ValueError(
            "VLLM_ULTRAQUANT_PDV_EXPERIMENT must be one of "
            f"{known} when PDV verify is requested; got {name!r}"
        )
    return importlib.import_module(module_name)
