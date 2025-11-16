# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 + TorchTitan Integration Example.

This demonstrates using TorchTitan's Qwen3 model with vLLM by:
1. Importing the TorchTitan model architecture
2. Replacing attention layers with vLLM's TrainableFlashAttention
3. Registering with vLLM's model registry

Requirements:
    pip install torchtitan

Example:
    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )
    ```

IMPORTANT: TorchTitan imports are deferred to avoid CUDA initialization
before vLLM's multiprocessing fork.
"""

import torch
import torch.nn as nn

from vllm.model_executor.custom_models import (
    load_external_weights,
    store_positions_in_context,
)
from vllm.model_executor.layers.attention_replacement import (
    replace_with_trainable_attention,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.custom_model_wrapper import VLLMModelForCausalLM
from vllm.model_executor.parallel_context import ParallelContext


class Qwen3TorchTitanForCausalLM(VLLMModelForCausalLM):
    """
    vLLM-compatible wrapper for TorchTitan's Qwen3 model.

    This class integrates TorchTitan's Qwen3Model with vLLM by:
    1. Importing TorchTitan's model architecture
    2. Replacing attention with vLLM's TrainableFlashAttention
    3. Implementing the vLLM model interface
    """

    supports_pp = False
    supports_multimodal = False

    def __init__(
        self,
        vllm_config,
        parallel_context: ParallelContext | None = None,
        **kwargs,
    ):
        super().__init__()

        # vLLM config is required for this example
        assert vllm_config is not None, "vllm_config is required"

        # Import TorchTitan's Qwen3 model (deferred import to avoid CUDA init issues)
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
        from torchtitan.models.qwen3.model.model import Qwen3Model

        # Map HuggingFace config to TorchTitan ModelArgs
        hf_config = vllm_config.hf_config
        model_args = Qwen3ModelArgs(
            vocab_size=getattr(hf_config, "vocab_size", 151936),
            dim=getattr(hf_config, "hidden_size", 2048),
            n_layers=getattr(hf_config, "num_hidden_layers", 4),
            n_heads=getattr(hf_config, "num_attention_heads", 16),
            n_kv_heads=getattr(hf_config, "num_key_value_heads", 2),
            head_dim=getattr(hf_config, "head_dim", 128),
            hidden_dim=getattr(hf_config, "intermediate_size", 11008),
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 8192),
            rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
            qk_norm=getattr(hf_config, "qk_norm", True),
        )

        # Create TorchTitan model
        self.model = Qwen3Model(model_args)
        self.parallel_context = parallel_context

        # Replace attention with vLLM's TrainableFlashAttention
        # (This happens before TP so TP can shard the attention weights)
        replace_with_trainable_attention(self.model, use_mla=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [batch, seq_len] (optional if inputs_embeds provided)
            positions: Position indices from vLLM for RoPE
            inputs_embeds: Pre-computed embeddings (optional, used by vLLM)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states before LM head
        """
        # Store positions in forward context for attention layers
        store_positions_in_context(positions)

        # Get embeddings
        h = (
            inputs_embeds
            if inputs_embeds is not None
            else self.model.tok_embeddings(input_ids)
        )

        # Get RoPE cache
        seqlen = h.shape[1] if h.dim() == 3 else h.shape[0]
        rope_cache = self.model.rope_cache[:seqlen]

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None)

        # Final norm
        return self.model.norm(h)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.model.output(hidden_states)

    def load_weights(self, weights_iter):
        """
        Load weights from HuggingFace checkpoint.

        Maps HF Qwen weight names → TorchTitan naming convention.
        """
        # HF → TorchTitan name mapping
        hf_to_tt = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "norm.weight",
            # Attention weights
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # MLP weights
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        }

        # Load weights using utility function
        loaded, skipped = load_external_weights(
            model=self.model, weights_iter=weights_iter, name_mapping=hf_to_tt
        )

        print(f"✅ Loaded {loaded} parameters, skipped {skipped}")


def build_qwen3_torchtitan(vllm_config, parallel_context: ParallelContext) -> nn.Module:
    """
    Factory function to build Qwen3 with TorchTitan + vLLM.

    This is registered with vLLM's ModelRegistry to enable:
        LLM(model="Qwen/Qwen3-0.6B", ...)

    Args:
        vllm_config: vLLM configuration object
        parallel_context: Parallelism context with TP/PP info

    Returns:
        Qwen3TorchTitanForCausalLM instance
    """
    # Create model
    model = Qwen3TorchTitanForCausalLM(
        vllm_config=vllm_config, parallel_context=parallel_context
    )

    # Apply tensor parallelism if TP > 1
    # This must happen AFTER model creation and attention replacement
    # but BEFORE dtype conversion (to avoid dtype issues with DTensors)
    if parallel_context is not None:
        tp_size = parallel_context.get_tensor_parallel_world_size()
        if tp_size > 1:
            from torch.distributed.device_mesh import init_device_mesh
            from torchtitan.models.qwen3.infra.parallelize import apply_non_moe_tp

            print(f"🔧 Applying Tensor Parallelism (TP={tp_size})...")

            # Create DeviceMesh for TorchTitan
            tp_mesh = init_device_mesh(
                "cuda",
                (tp_size,),
                mesh_dim_names=("tp",),
            )

            # Apply TorchTitan's tensor parallelism to shard weights
            apply_non_moe_tp(
                model.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,  # Don't shard the output for loss computation
                enable_float8_tensorwise_tp=False,
                enable_async_tp=False,
            )

            print(f"✅ Applied Tensor Parallelism (TP={tp_size})")

    # Convert to dtype if specified (happens after TP)
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "dtype"
    ):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model


# Register with vLLM's ModelRegistry
ModelRegistry.register_model("Qwen3TorchTitan", build_qwen3_torchtitan)


if __name__ == "__main__":
    print("=" * 70)
    print("Qwen3 + TorchTitan Integration Test")
    print("=" * 70)

    # Test model creation without vLLM context
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
    from torchtitan.models.qwen3.model.model import Qwen3Model

    model_args = Qwen3ModelArgs(
        vocab_size=151936,
        dim=2048,
        n_layers=4,
        n_heads=16,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=11008,
        norm_eps=1e-6,
        max_seq_len=8192,
        rope_theta=1000000.0,
        qk_norm=True,
    )

    model = Qwen3Model(model_args)

    # Replace attention with vLLM's TrainableFlashAttention
    replace_with_trainable_attention(model, use_mla=False)

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        h = model.tok_embeddings(input_ids)
        rope_cache = model.rope_cache[:seq_len]
        for layer in model.layers.values():
            h = layer(h, rope_cache, None)
        h = model.norm(h)
        logits = model.output(h)

    print(f"✓ Forward pass: {input_ids.shape} → {logits.shape}")
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
