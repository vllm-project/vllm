# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepSeek V3 with TorchTitan Integration Example.

This demonstrates using TorchTitan's DeepSeek V3 model with vLLM by:
1. Importing the TorchTitan model architecture
2. Replacing attention layers with vLLM's TrainableMLA
3. Registering with vLLM's model registry

Requirements:
    pip install torchtitan

Example:
    ```python
    from vllm import LLM

    llm = LLM(
        model="deepseek-ai/DeepSeek-V3-Base",
        trust_remote_code=True,
    )
    ```
"""

import torch
import torch.nn as nn

from vllm.model_executor.custom_models import (
    convert_freqs_cis_to_real,
    create_mla_kv_cache_spec,
    load_external_weights,
    store_positions_in_context,
)
from vllm.model_executor.layers.attention_replacement import (
    replace_with_trainable_attention,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.custom_model_wrapper import VLLMModelForCausalLM
from vllm.model_executor.parallel_context import ParallelContext


class DeepSeekV3TorchTitanForCausalLM(VLLMModelForCausalLM):
    """
    vLLM-compatible wrapper for TorchTitan's DeepSeek V3 model.

    This class integrates TorchTitan's DeepSeekV3Model with vLLM by:
    1. Importing TorchTitan's model architecture
    2. Replacing attention with vLLM's TrainableMLA
    3. Implementing the vLLM model interface
    """

    supports_pp = False  # Pipeline parallelism not supported yet
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

        # Import TorchTitan's DeepSeek V3 model
        from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
        from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model

        # Map HuggingFace config to TorchTitan ModelArgs
        hf_config = vllm_config.hf_config
        model_args = DeepSeekV3ModelArgs(
            vocab_size=getattr(hf_config, "vocab_size", 102400),
            dim=getattr(hf_config, "hidden_size", 2048),
            inter_dim=getattr(hf_config, "intermediate_size", 10944),
            moe_inter_dim=getattr(hf_config, "moe_intermediate_size", 1408),
            n_layers=getattr(hf_config, "num_hidden_layers", 27),
            n_dense_layers=getattr(hf_config, "num_dense_layers", 1),
            n_heads=getattr(hf_config, "num_attention_heads", 16),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 4096),
            q_lora_rank=getattr(hf_config, "q_lora_rank", 0),
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", 512),
            qk_nope_head_dim=getattr(hf_config, "qk_nope_head_dim", 128),
            qk_rope_head_dim=getattr(hf_config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(hf_config, "v_head_dim", 128),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            use_flex_attn=False,  # We'll use vLLM's attention
        )

        # Create TorchTitan model
        self.model = DeepSeekV3Model(model_args)
        self.config = model_args

        # Replace attention with vLLM's TrainableMLA
        # (This happens before TP so TP can shard the attention weights)
        replace_with_trainable_attention(self.model, use_mla=True)

        # Convert freqs_cis to real format (required for vLLM)
        self.model.freqs_cis = convert_freqs_cis_to_real(self.model.freqs_cis)

    def get_kv_cache_spec(self, vllm_config):
        """Return KV cache specification for MLA."""
        return create_mla_kv_cache_spec(
            kv_lora_rank=self.config.kv_lora_rank,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            block_size=vllm_config.cache_config.block_size,
            dtype=vllm_config.model_config.dtype,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Args:
            input_ids: Token indices [batch, seq_len]
            positions: Position indices from vLLM for RoPE indexing
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Hidden states before final projection [batch, seq_len, hidden]
        """
        # Store positions in forward context for attention layers
        store_positions_in_context(positions)

        # Get embeddings
        h = self.model.tok_embeddings(input_ids)

        # Forward through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, self.model.freqs_cis, attention_masks=None)

        # Final layer norm
        h = self.model.norm(h)

        return h

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

        Maps HF safetensor names to TorchTitan parameter names using the
        same mapping as TorchTitan's DeepSeekV3StateDictAdapter.
        """
        # Build HF → TorchTitan name mapping
        hf_to_tt = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention KV (always present)
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # MLP
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # Output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Add Q projection mapping based on q_lora_rank
        if self.config.q_lora_rank != 0:
            hf_to_tt.update(
                {
                    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
                    "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
                    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
                }
            )
        else:
            hf_to_tt.update(
                {
                    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
                }
            )

        # Load weights using utility function
        loaded, skipped = load_external_weights(
            model=self.model, weights_iter=weights_iter, name_mapping=hf_to_tt
        )

        print(f"✅ Loaded {loaded} parameters, skipped {skipped}")


def build_deepseek_v3_torchtitan(
    vllm_config, parallel_context: ParallelContext
) -> nn.Module:
    """
    Factory function to build DeepSeek V3 with TorchTitan + vLLM.

    This is registered with vLLM's ModelRegistry to enable:
        LLM(model="deepseek-ai/DeepSeek-V3-Base", ...)

    Args:
        vllm_config: vLLM configuration object
        parallel_context: Parallelism context with TP/PP info

    Returns:
        DeepSeekV3TorchTitanForCausalLM instance
    """
    # Create model
    model = DeepSeekV3TorchTitanForCausalLM(
        vllm_config=vllm_config, parallel_context=parallel_context
    )

    # Apply tensor parallelism if TP > 1
    # This must happen AFTER model creation and attention replacement
    # but BEFORE dtype conversion (to avoid dtype issues with DTensors)
    if parallel_context is not None:
        tp_size = parallel_context.get_tensor_parallel_world_size()
        if tp_size > 1:
            from torch.distributed.device_mesh import init_device_mesh
            from torchtitan.models.deepseek_v3.infra.parallelize import (
                apply_non_moe_tp,
            )

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
                loss_parallel=False,  # Don't shard output for loss computation
                enable_float8_tensorwise_tp=False,
                use_flex_attn=False,
            )

            print(f"✅ Applied Tensor Parallelism (TP={tp_size})")

    # Convert to dtype if specified (happens after TP)
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "dtype"
    ):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model


# Register with vLLM's ModelRegistry
ModelRegistry.register_model("DeepSeekV3TorchTitan", build_deepseek_v3_torchtitan)


if __name__ == "__main__":
    print("=" * 70)
    print("DeepSeek V3 + TorchTitan Integration Test")
    print("=" * 70)

    # Test model creation without vLLM context
    # Use small config for fast testing (all dense layers to avoid MoE on CPU)
    from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
    from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model

    model_args = DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        n_layers=4,
        n_dense_layers=4,  # All dense (no MoE)
        n_heads=16,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_seq_len=4096,
        use_flex_attn=False,
    )

    model = DeepSeekV3Model(model_args)

    # Replace attention with vLLM's TrainableMLA
    replace_with_trainable_attention(model, use_mla=True)

    # Convert freqs_cis to real format
    model.freqs_cis = convert_freqs_cis_to_real(model.freqs_cis)

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        h = model.tok_embeddings(input_ids)
        for layer in model.layers.values():
            h = layer(h, model.freqs_cis, attention_masks=None)
        h = model.norm(h)
        logits = model.output(h)

    print(f"✓ Forward pass: {input_ids.shape} → {logits.shape}")
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
