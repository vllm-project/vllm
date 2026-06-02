# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Spawn-safe registration of the predictable dummy models.

These models back the ``extract_hidden_states`` integration tests. vLLM calls
``load_general_plugins()`` in *every* process it spins up -- the main process,
the short-lived model-inspection subprocess, and each engine worker -- under
both ``fork`` and ``spawn``. Registering the models through this entry point is
therefore the only mechanism that makes them visible everywhere, independent of
the multiprocessing start method and of the directory pytest was launched from.

A plain in-process ``ModelRegistry.register_model`` call (the previous test-only
approach) is lost across a ``spawn``, and a lazy import string pointing at the
``tests.*`` package only resolves when the repo root happens to be importable in
the child process. Packaging the models here avoids both pitfalls.

Install once with::

    pip install -e tests/plugins/vllm_add_predictable_models
"""

from vllm import ModelRegistry

# Lazy import strings (resolved per-process by vLLM) keep heavyweight torch/vllm
# model imports out of plugin load, which runs in every process.
_MODELS = {
    "PredictableLlamaForCausalLM": (
        "vllm_add_predictable_models.predictable_llama:PredictableLlamaForCausalLM"
    ),
    "PredictableHybridForCausalLM": (
        "vllm_add_predictable_models.predictable_hybrid:PredictableHybridForCausalLM"
    ),
}


def register():
    supported = ModelRegistry.get_supported_archs()
    for arch, model_path in _MODELS.items():
        if arch not in supported:
            ModelRegistry.register_model(arch, model_path)
