# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility shims for external multimodal plugins.

Some external vLLM plugins (e.g. model-specific plugin packages) import
`BaseDummyInputsBuilder` / `ProcessorInputs` from `vllm.multimodal.profiling`.

In this vLLM version, these classes live under `vllm.multimodal.processing`.
"""

from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs

__all__ = [
    "BaseDummyInputsBuilder",
    "ProcessorInputs",
]
