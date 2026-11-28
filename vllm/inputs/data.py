# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt types exposed as ``vllm.inputs.data`` for external (Ray) integrations.

Ray Data's LLM batch pipeline imports ``vllm.inputs.data.TextPrompt`` and
``vllm.inputs.data.TokensPrompt``. Those types live in :mod:`vllm.inputs.llm`;
this submodule re-exports them so ``import vllm.inputs`` always binds a real
``data`` attribute (subpackages are not auto-loaded on attribute access).
"""

from .llm import TextPrompt, TokensPrompt

__all__ = ["TextPrompt", "TokensPrompt"]
