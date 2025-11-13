# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Using Megatron-LM Tensor Parallel Layers with vLLM

This example demonstrates how to use NVIDIA's Megatron-LM tensor parallel
layers with vLLM's inference engine. This is the key use case: bringing your
own parallelism implementation instead of being forced to use vLLM's internals.

Key features:
- Use Megatron-LM's ColumnParallelLinear and RowParallelLinear for MLPs
- Use vLLM's TrainableFlashAttention for training-compatible attention
- Full transformer architecture with attention + MLP blocks
- Configure Megatron from vLLM's parallel context
- Test with actual LLM() API and worker spawning

This demonstrates the complete integration:
1. Attention: vLLM's TrainableFlashAttention (with backward pass support)
2. MLP: Megatron's ColumnParallelLinear → GELU → RowParallelLinear

Requirements:
    pip install megatron-core flash-attn

For more details on Megatron-LM:
    https://github.com/NVIDIA/Megatron-LM
"""

import json
import os
import tempfile

import torch
import torch.nn as nn

from vllm.model_executor.layers.trainable_attention import (
    TrainableFlashAttention,  # vLLM's training-compatible attention
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


class MegatronTransformer(nn.Module):
    """
    Example model using Megatron-LM's tensor parallel layers for MLPs
    and vLLM's TrainableFlashAttention for attention.

    This demonstrates how users can leverage:
    1. External parallelism libraries (Megatron-LM) for custom components
    2. vLLM's training-compatible modules (TrainableFlashAttention)

    Architecture:
    - Attention: vLLM's TrainableFlashAttention (supports backward pass)
    - MLP: Megatron's ColumnParallelLinear → GELU → RowParallelLinear
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 4,
        num_attention_heads: int = 32,
        tp_size: int = 1,
        tp_group=None,  # vLLM's tensor parallel group
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.tp_size = tp_size
        self.head_dim = hidden_size // num_attention_heads

        # Standard embedding (not parallelized)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Build layers with both attention and MLP
        self.layers = nn.ModuleList()

        # Create Megatron config
        megatron_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
        )

        for _ in range(num_layers):
            # Build transformer layer with vLLM attention + Megatron MLP
            layer = nn.ModuleDict(
                {
                    # === ATTENTION BLOCK ===
                    # Use vLLM's TrainableFlashAttention
                    # (includes QKV + output projections)
                    "attn": TrainableFlashAttention(
                        hidden_size=hidden_size,
                        num_heads=num_attention_heads,
                        dropout=0.0,
                        causal=True,
                    ),
                    "attn_norm": nn.LayerNorm(hidden_size),
                    # === MLP BLOCK ===
                    # Use Megatron's parallel layers for MLP
                    "fc1": ColumnParallelLinear(
                        hidden_size,
                        intermediate_size,
                        config=megatron_config,
                        init_method=nn.init.xavier_normal_,
                        bias=False,
                        gather_output=False,  # Keep output sharded
                        tp_group=tp_group,
                    ),
                    "act": nn.GELU(),
                    "fc2": RowParallelLinear(
                        intermediate_size,
                        hidden_size,
                        config=megatron_config,
                        init_method=nn.init.xavier_normal_,
                        bias=False,
                        input_is_parallel=True,  # Input is sharded
                        skip_bias_add=False,
                        tp_group=tp_group,
                    ),
                    "mlp_norm": nn.LayerNorm(hidden_size),
                }
            )
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM."""
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Required by vLLM."""
        # Handle both 2D [batch, seq_len] and 1D [total_tokens] input shapes
        if input_ids.dim() == 1:
            # V1 engine passes flattened tokens
            input_ids = input_ids.unsqueeze(0)  # [total_tokens] -> [1, total_tokens]

        # Clamp input_ids to valid range for embedding lookup
        # (warmup may pass out-of-bounds values)
        input_ids = input_ids.clamp(0, self.vocab_size - 1)

        hidden_states = self.embeddings(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            # === ATTENTION BLOCK ===
            residual = hidden_states
            attn_output = layer["attn"](hidden_states)
            hidden_states = layer["attn_norm"](residual + attn_output)

            # === MLP BLOCK ===
            residual = hidden_states
            hidden_states = layer["fc1"](hidden_states)
            # Handle Megatron output (tuple) for fc1
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer["act"](hidden_states)

            hidden_states = layer["fc2"](hidden_states)
            # Handle Megatron output (tuple) for fc2
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer["mlp_norm"](hidden_states + residual)

        hidden_states = self.final_norm(hidden_states)

        # CRITICAL: vLLM V1 expects [total_num_tokens, hidden_size]
        # NOT [batch, seq_len, hidden_size]
        # Flatten the output if it's 3D
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute output logits.

        Note: hidden_states here are ALREADY indexed by vLLM using logit_indices_device.
        We just need to apply the LM head to project to vocabulary space.
        """
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        """Load weights (dummy implementation for demo)."""
        # For this demo, we use random weights
        # In production, you'd load actual weights here
        pass


def build_megatron_model(vllm_config, parallel_context: ParallelContext):
    """
    Factory that builds a model using Megatron-LM parallelism.

    This shows how to:
    1. Get TP info from vLLM's parallel context
    2. Get vLLM's tensor parallel process group
    3. Pass the process group to Megatron layers
    """
    # Import Megatron here to avoid CUDA initialization before fork
    global ColumnParallelLinear, RowParallelLinear, TransformerConfig

    from megatron.core.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.transformer.transformer_config import (
        TransformerConfig,
    )

    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()

    print(f"\n{'=' * 60}")
    print(f"Building Megatron model on TP rank {tp_rank}/{tp_size}")
    print(f"{'=' * 60}\n")

    # Get vLLM's tensor parallel process group
    from vllm.distributed import parallel_state as vllm_parallel_state

    tp_coordinator = vllm_parallel_state.get_tp_group()
    tp_group = tp_coordinator.device_group

    assert tp_group is not None, "Failed to get TP process group from vLLM!"
    print(f"✓ Got vLLM's TP device group: {tp_group}")

    # Set Megatron's global tensor parallel group to vLLM's group
    # Megatron layers require this even though they also accept tp_group as parameter
    import megatron.core.parallel_state as megatron_parallel_state

    # Set the minimum required global variables for Megatron to work
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = list(range(tp_size))

    # Also set the cached rank/world_size values that Megatron uses
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = tp_rank
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = tp_size

    print(f"✓ Set Megatron's TP group to vLLM's group (rank={tp_rank}, size={tp_size})")

    # Extract config from vLLM's config
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "hf_config"
    ):
        hf_config = vllm_config.model_config.hf_config
        vocab_size = getattr(hf_config, "vocab_size", 32000)
        hidden_size = getattr(hf_config, "hidden_size", 4096)
        num_attention_heads = getattr(hf_config, "num_attention_heads", 32)
    else:
        # Fallback to defaults
        vocab_size = 32000
        hidden_size = 4096
        num_attention_heads = 32

    # Build model with Megatron TP support
    model = MegatronTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_layers=4,
        num_attention_heads=num_attention_heads,
        tp_size=tp_size,
        tp_group=tp_group,
    )

    # Convert model to vLLM's dtype
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "dtype"
    ):
        model = model.to(dtype=vllm_config.model_config.dtype)
        print(f"✓ Converted model to {vllm_config.model_config.dtype}")

    print("✓ Using Megatron-LM tensor parallel layers!")
    print(f"  TP size: {tp_size}")
    print("  Attention: vLLM's TrainableFlashAttention (with backward pass)")
    print("  MLP: Megatron ColumnParallelLinear + RowParallelLinear")

    return model


# Register the model
ModelRegistry.register_model("MegatronModel", build_megatron_model)


if __name__ == "__main__":
    print("=" * 70)
    print("Megatron-LM + vLLM Integration Demo")
    print("=" * 70)

    print("\nWhat this demonstrates:")
    print("  ✓ External parallelism (Megatron-LM) for MLP layers")
    print("  ✓ vLLM's TrainableFlashAttention for attention")
    print("  ✓ Combining both in a single model with TP=4")

    print("\n[Test] Running with LLM() API and TP=4")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model_type": "gpt2",
            "architectures": ["MegatronModel"],
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
        }

        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        from vllm import LLM

        llm = LLM(
            model=tmpdir,
            tokenizer=None,
            tensor_parallel_size=4,  # Always use TP=4
            max_model_len=128,
            max_num_seqs=1,
            enforce_eager=True,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            # NOTE: We disable prefix caching for this demo since our simple
            # TrainableFlashAttention doesn't integrate with V1's KV cache
            # infrastructure. For production use, you'd want to use vLLM's
            # built-in Attention layers or implement full KV cache support.
            enable_prefix_caching=False,
        )

        print("\n" + "=" * 70)
        print("✅ SUCCESS! Megatron-LM + vLLM integration works!")
        print("=" * 70)
