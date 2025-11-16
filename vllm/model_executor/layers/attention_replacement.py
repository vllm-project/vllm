# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for replacing TorchTitan attention layers with vLLM trainable attention.

This module provides functions to automatically replace TorchTitan's attention
layers with vLLM's optimized trainable attention layers (TrainableFlashAttention
or TrainableMLA) while preserving weights.

Example usage:
    ```python
    from torchtitan.models.qwen3.model import Qwen3Model
    from vllm.model_executor.layers.attention_replacement import (
        replace_with_trainable_attention,
    )

    # Create TorchTitan model
    model = Qwen3Model(model_args)

    # Replace attention layers with vLLM trainable attention
    replace_with_trainable_attention(model, use_mla=False)
    ```
"""

import torch.nn as nn

from vllm.model_executor.layers.trainable_attention import TrainableFlashAttention
from vllm.model_executor.layers.trainable_mla_attention import (
    MLAConfig,
    TrainableMLA,
)


def replace_with_trainable_attention(
    model: nn.Module,
    use_mla: bool = False,
) -> None:
    """
    Replace TorchTitan attention layers with vLLM trainable attention.

    This function performs in-place module surgery, replacing all attention
    layers in model.layers with either TrainableFlashAttention or TrainableMLA
    while preserving the original weights.

    Args:
        model: TorchTitan model with .layers attribute (dict or nn.ModuleDict)
        use_mla: If True, use TrainableMLA; otherwise use TrainableFlashAttention

    Raises:
        AttributeError: If model doesn't have .layers attribute
        ValueError: If attention layer structure is not recognized
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        old_attn = layer.attention

        if use_mla:
            # Create TrainableMLA and transfer weights
            new_attn = _create_trainable_mla_from_torchtitan(old_attn)
        else:
            # Create TrainableFlashAttention and transfer weights
            new_attn = _create_trainable_flash_attention_from_torchtitan(old_attn)

        # Replace attention module
        layer.attention = new_attn


def _create_trainable_flash_attention_from_torchtitan(
    torchtitan_attn: nn.Module,
) -> TrainableFlashAttention:
    """
    Create TrainableFlashAttention from TorchTitan attention and transfer weights.

    Args:
        torchtitan_attn: TorchTitan Attention module

    Returns:
        TrainableFlashAttention with transferred weights
    """
    # Extract config from TorchTitan attention
    hidden_size = torchtitan_attn.wq.weight.shape[1]
    num_heads = torchtitan_attn.n_heads
    num_kv_heads = getattr(torchtitan_attn, "n_kv_heads", num_heads)
    head_dim = torchtitan_attn.head_dim
    use_qk_norm = (
        hasattr(torchtitan_attn, "q_norm") and torchtitan_attn.q_norm is not None
    )

    # Create vLLM attention
    vllm_attn = TrainableFlashAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_fused_qkv=False,  # TorchTitan uses separate wq/wk/wv
        use_qk_norm=use_qk_norm,
    )

    # Transfer weights (TorchTitan and vLLM use same naming: wq, wk, wv, wo)
    vllm_attn.wq.weight.data.copy_(torchtitan_attn.wq.weight.data)
    vllm_attn.wk.weight.data.copy_(torchtitan_attn.wk.weight.data)
    vllm_attn.wv.weight.data.copy_(torchtitan_attn.wv.weight.data)
    vllm_attn.wo.weight.data.copy_(torchtitan_attn.wo.weight.data)

    # Transfer QK norm weights if present
    if use_qk_norm:
        vllm_attn.q_norm.weight.data.copy_(torchtitan_attn.q_norm.weight.data)
        vllm_attn.k_norm.weight.data.copy_(torchtitan_attn.k_norm.weight.data)

    return vllm_attn


def _create_trainable_mla_from_torchtitan(
    torchtitan_attn: nn.Module,
) -> TrainableMLA:
    """
    Create TrainableMLA from TorchTitan MLA attention and transfer weights.

    Args:
        torchtitan_attn: TorchTitan MLA Attention module

    Returns:
        TrainableMLA with transferred weights
    """
    # Extract MLA config from TorchTitan attention
    config = MLAConfig(
        hidden_size=torchtitan_attn.dim,
        num_heads=torchtitan_attn.n_heads,
        q_lora_rank=torchtitan_attn.q_lora_rank,
        kv_lora_rank=torchtitan_attn.kv_lora_rank,
        qk_nope_head_dim=torchtitan_attn.qk_nope_head_dim,
        qk_rope_head_dim=torchtitan_attn.qk_rope_head_dim,
        v_head_dim=torchtitan_attn.v_head_dim,
        norm_eps=1e-5,  # Standard value for DeepSeek
        dropout=0.0,
        scale=torchtitan_attn.softmax_scale,
        causal=True,
    )

    # Create vLLM MLA
    vllm_mla = TrainableMLA(config)

    # Transfer weights
    if vllm_mla.q_lora_rank == 0:
        # Direct Q projection
        vllm_mla.wq.weight.data.copy_(torchtitan_attn.wq.weight.data)
    else:
        # LoRA Q projection
        assert vllm_mla.q_norm is not None  # q_norm exists when q_lora_rank > 0
        vllm_mla.wq_a.weight.data.copy_(torchtitan_attn.wq_a.weight.data)
        vllm_mla.wq_b.weight.data.copy_(torchtitan_attn.wq_b.weight.data)
        vllm_mla.q_norm.weight.data.copy_(torchtitan_attn.q_norm.weight.data)

    # KV projection (always LoRA)
    vllm_mla.wkv_a.weight.data.copy_(torchtitan_attn.wkv_a.weight.data)
    vllm_mla.wkv_b.weight.data.copy_(torchtitan_attn.wkv_b.weight.data)
    vllm_mla.kv_norm.weight.data.copy_(torchtitan_attn.kv_norm.weight.data)

    # Output projection
    vllm_mla.wo.weight.data.copy_(torchtitan_attn.wo.weight.data)

    return vllm_mla
