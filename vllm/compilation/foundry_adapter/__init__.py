# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM lifecycle integration for the optional Foundry CUDA backend."""

from typing import TYPE_CHECKING

from .config import CUDAGraphExtensionMode, get_graph_extension_mode

if TYPE_CHECKING:
    from vllm.config import CompilationConfig


def install_hooks(compilation_config: "CompilationConfig") -> None:
    from .hooks import install_hooks as install

    install(compilation_config)


__all__ = [
    "CUDAGraphExtensionMode",
    "get_graph_extension_mode",
    "install_hooks",
]
