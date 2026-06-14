# SPDX-License-Identifier: Apache-2.0
"""BitNet b1.58 BF16 plugin for vLLM.

This module provides out-of-tree registration for BitNetForCausalLM.
If the model is already registered upstream (in vllm.model_executor.models),
this is a no-op.
"""


def register():
    """Register BitNetForCausalLM with vLLM's model registry.

    Safe to call multiple times — skips if already registered.
    """
    from vllm import ModelRegistry

    if "BitNetForCausalLM" not in ModelRegistry.models:
        from bitnet_vllm.bitnet import BitNetForCausalLM

        ModelRegistry.register_model(
            "BitNetForCausalLM",
            BitNetForCausalLM,
        )
