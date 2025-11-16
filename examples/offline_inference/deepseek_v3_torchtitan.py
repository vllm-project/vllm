# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepSeek V3 with TorchTitan Integration Example.

This demonstrates using TorchTitan's DeepSeek V3 model with vLLM by:
1. Importing the TorchTitan model architecture
2. Replacing attention layers with vLLM's TrainableMLA
3. Registering with vLLM's model registry using callback pattern

Requirements:
    pip install torchtitan

Example:
    ```python
    from vllm import LLM

    llm = LLM(
        model="deepseek-ai/DeepSeek-V3.1-Base",
        trust_remote_code=True,
    )
    ```
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.trainable_mla_attention import (
    MLAConfig,
    TrainableMLA,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


def extract_mla_config_from_torchtitan_attention(
    torchtitan_attn: nn.Module,
) -> MLAConfig:
    """
    Extract MLA configuration from a TorchTitan Attention module.

    Args:
        torchtitan_attn: TorchTitan's Attention module

    Returns:
        MLAConfig with parameters extracted from torchtitan_attn
    """
    return MLAConfig(
        hidden_size=torchtitan_attn.dim,
        num_heads=torchtitan_attn.n_heads,
        q_lora_rank=torchtitan_attn.q_lora_rank,
        kv_lora_rank=torchtitan_attn.kv_lora_rank,
        qk_nope_head_dim=torchtitan_attn.qk_nope_head_dim,
        qk_rope_head_dim=torchtitan_attn.qk_rope_head_dim,
        v_head_dim=torchtitan_attn.v_head_dim,
        norm_eps=1e-5,  # Default from DeepSeek V3
        dropout=0.0,
        scale=torchtitan_attn.softmax_scale,
        causal=True,
    )


def transfer_mla_weights(torchtitan_attn: nn.Module, vllm_mla: TrainableMLA) -> None:
    """
    Transfer weights from TorchTitan Attention to vLLM TrainableMLA.

    Args:
        torchtitan_attn: Source TorchTitan Attention module
        vllm_mla: Target vLLM TrainableMLA module
    """
    # Query projection
    if vllm_mla.q_lora_rank == 0:
        # Direct projection
        vllm_mla.wq.weight.data.copy_(torchtitan_attn.wq.weight.data)
    else:
        # LoRA projection with norm
        vllm_mla.wq_a.weight.data.copy_(torchtitan_attn.wq_a.weight.data)
        vllm_mla.wq_b.weight.data.copy_(torchtitan_attn.wq_b.weight.data)
        vllm_mla.q_norm.weight.data.copy_(torchtitan_attn.q_norm.weight.data)

    # Key-value projection (always LoRA)
    vllm_mla.wkv_a.weight.data.copy_(torchtitan_attn.wkv_a.weight.data)
    vllm_mla.wkv_b.weight.data.copy_(torchtitan_attn.wkv_b.weight.data)
    vllm_mla.kv_norm.weight.data.copy_(torchtitan_attn.kv_norm.weight.data)

    # Output projection
    vllm_mla.wo.weight.data.copy_(torchtitan_attn.wo.weight.data)


def replace_attention_with_vllm_mla(model: nn.Module) -> None:
    """
    Replace all TorchTitan Attention modules with vLLM TrainableMLA.

    This performs module surgery in-place, replacing the attention implementation
    while preserving all weights and the rest of the model structure.

    Args:
        model: TorchTitan DeepSeekV3Model instance
    """
    print("🔧 Replacing TorchTitan attention with vLLM TrainableMLA...")

    for layer_name, layer in model.layers.items():
        # Extract config from original attention
        config = extract_mla_config_from_torchtitan_attention(layer.attention)

        # Create new vLLM MLA attention
        vllm_mla = TrainableMLA(config)

        # Transfer weights
        transfer_mla_weights(layer.attention, vllm_mla)

        # Replace the attention module
        layer.attention = vllm_mla

        print(f"  ✓ Replaced attention in layer {layer_name}")

    print(f"✅ Successfully replaced attention in {len(model.layers)} layers")


class DeepSeekV3TorchTitanForCausalLM(nn.Module):
    """
    vLLM-compatible wrapper for TorchTitan's DeepSeek V3 model.

    This class:
    1. Imports TorchTitan's DeepSeekV3Model
    2. Replaces attention layers with vLLM's TrainableMLA
    3. Implements vLLM's VllmModelForTextGeneration interface

    Interface:
        - __init__(vllm_config, parallel_context)
        - get_input_embeddings(input_ids)
        - forward(input_ids, positions, **kwargs)
        - compute_logits(hidden_states, sampling_metadata)
        - load_weights(weights_iter)
    """

    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        vllm_config=None,
        parallel_context: ParallelContext | None = None,
        **kwargs,
    ):
        super().__init__()

        # Import TorchTitan's DeepSeek V3 model
        from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
        from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model

        # Extract configuration
        if vllm_config is not None and hasattr(vllm_config, "hf_config"):
            hf_config = vllm_config.hf_config

            # Map HuggingFace config to TorchTitan ModelArgs
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
        else:
            # Use default small config for testing (all dense layers to avoid MoE on CPU)
            model_args = DeepSeekV3ModelArgs(
                vocab_size=102400,
                dim=2048,
                inter_dim=10944,
                n_layers=4,  # Use fewer layers for faster testing
                n_dense_layers=4,  # All dense (no MoE) to avoid CPU histogram issue
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                max_seq_len=4096,
                use_flex_attn=False,
            )

        # Create TorchTitan model
        print("\n" + "=" * 70)
        print("Building DeepSeek V3 with TorchTitan + vLLM")
        print("=" * 70)
        print(f"  Layers: {model_args.n_layers}")
        print(f"  Hidden: {model_args.dim}")
        print(f"  Heads: {model_args.n_heads}")
        print(f"  Q LoRA rank: {model_args.q_lora_rank}")
        print(f"  KV LoRA rank: {model_args.kv_lora_rank}")
        print(f"  QK nope dim: {model_args.qk_nope_head_dim}")
        print(f"  QK rope dim: {model_args.qk_rope_head_dim}")
        print(f"  V head dim: {model_args.v_head_dim}")

        self.model = DeepSeekV3Model(model_args)
        self.config = model_args

        # Perform module surgery: replace attention with vLLM's TrainableMLA
        replace_attention_with_vllm_mla(self.model)

        # Convert freqs_cis from complex to real format BEFORE dtype conversion
        # This prevents the complex tensor from being cast to real (losing imaginary part)
        if self.model.freqs_cis.is_complex():
            print("🔄 Converting freqs_cis from complex to real format...")
            freqs_cos = self.model.freqs_cis.real  # [max_seq_len, qk_rope_head_dim//2]
            freqs_sin = self.model.freqs_cis.imag  # [max_seq_len, qk_rope_head_dim//2]
            # Concatenate cos and sin: [max_seq_len, qk_rope_head_dim]
            freqs_real = torch.cat([freqs_cos, freqs_sin], dim=-1)
            self.model.freqs_cis = freqs_real
            print(
                f"  ✓ Converted freqs_cis: {self.model.freqs_cis.shape} ({self.model.freqs_cis.dtype})"
            )

        print("=" * 70)
        print("✅ Model built successfully!")
        print("=" * 70 + "\n")

    def get_kv_cache_spec(self, vllm_config):
        """Return KV cache specification for MLA."""
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        # MLA uses compressed KV cache: kv_lora_rank + qk_rope_head_dim per token
        head_size = self.config.kv_lora_rank + self.config.qk_rope_head_dim

        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,  # MLA shares K_PE across all heads
            head_size=head_size,  # 512 + 64 = 576
            dtype=vllm_config.model_config.dtype,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply token embeddings to input_ids.

        Required by VllmModel interface.
        """
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
        # Store positions in forward context so TrainableMLA can access them
        if positions is not None:
            try:
                from vllm.forward_context import get_forward_context

                forward_ctx = get_forward_context()
                # Store positions in a custom attribute
                forward_ctx._torchtitan_positions = positions
            except Exception:
                pass

        # Get embeddings
        h = self.model.tok_embeddings(input_ids)

        # Forward through transformer layers
        # Note: TorchTitan's forward passes freqs_cis and attention_masks
        # We use the precomputed freqs_cis from the model
        for layer in self.model.layers.values():
            # TorchTitan layers expect freqs_cis and attention_masks
            # For now, pass None for attention_masks (causal is default)
            h = layer(h, self.model.freqs_cis, attention_masks=None)

        # Final layer norm
        h = self.model.norm(h)

        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Required by VllmModelForTextGeneration interface.
        """
        logits = self.model.output(hidden_states)
        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HuggingFace checkpoint.

        This maps HF safetensor names to TorchTitan parameter names using the
        same mapping as TorchTitan's DeepSeekV3StateDictAdapter.

        Args:
            weights_iter: Iterator yielding (name, tensor) from HF checkpoint
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Build HF → TorchTitan name mapping (from state_dict_adapter.py)
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

        # Get all parameter names in the model
        params_dict = dict(self.model.named_parameters())

        print("\n" + "=" * 70)
        print("Loading weights from HuggingFace checkpoint...")
        print("=" * 70)

        loaded_count = 0
        skipped_count = 0

        # Convert iterator to list to check if empty
        weights_list = list(weights_iter)
        if len(weights_list) == 0:
            print("  ⚠️  No weight files found - using random initialization")
            print("=" * 70 + "\n")
            return

        for hf_name, loaded_weight in weights_list:
            # Try to find matching pattern in hf_to_tt
            tt_name = None

            # Check if it's a layer-specific weight
            if "layers" in hf_name:
                # Extract layer number
                import regex as re

                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_num = layer_match.group(1)

                    # Try to find matching pattern
                    for hf_pattern, tt_pattern in hf_to_tt.items():
                        if "{}" in hf_pattern:
                            hf_concrete = hf_pattern.format(layer_num)
                            if hf_name == hf_concrete:
                                tt_name = tt_pattern.format(layer_num)
                                break
            else:
                # Non-layer weight (embeddings, norms, output)
                tt_name = hf_to_tt.get(hf_name)

            if tt_name is None:
                # Skip MoE weights and other unmapped weights
                if (
                    "mlp.experts" in hf_name
                    or "mlp.gate" in hf_name
                    or "mlp.shared_experts" in hf_name
                ):
                    # MoE weights - skip silently
                    skipped_count += 1
                    continue
                else:
                    print(f"  ⚠️  No mapping for: {hf_name}")
                    skipped_count += 1
                continue

            # Check if parameter exists in model
            if tt_name not in params_dict:
                print(f"  ⚠️  Parameter not found in model: {tt_name}")
                skipped_count += 1
                continue

            # Load the weight
            param = params_dict[tt_name]

            # Verify shapes match
            if param.shape != loaded_weight.shape:
                print(f"  ⚠️  Shape mismatch for {tt_name}:")
                print(f"      Model: {param.shape}, Checkpoint: {loaded_weight.shape}")
                skipped_count += 1
                continue

            # Load the weight
            default_weight_loader(param, loaded_weight)
            loaded_count += 1

            # Log first few loads for verification
            if loaded_count <= 5:
                print(f"  ✓ Loaded {tt_name}: {loaded_weight.shape}")

        print(f"\n{'=' * 70}")
        print("✅ Weight loading complete!")
        print(f"  Loaded: {loaded_count} parameters")
        print(f"  Skipped: {skipped_count} parameters")
        print(f"{'=' * 70}\n")


def build_deepseek_v3_torchtitan(
    vllm_config, parallel_context: ParallelContext
) -> nn.Module:
    """
    Factory function to build DeepSeek V3 with TorchTitan + vLLM.

    This is registered with vLLM's ModelRegistry to enable:
        LLM(model="deepseek-ai/DeepSeek-V3.1-Base", ...)

    Args:
        vllm_config: vLLM configuration object
        parallel_context: Parallelism context with TP/PP info

    Returns:
        DeepSeekV3TorchTitanForCausalLM instance
    """
    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()

    print(f"\n{'=' * 70}")
    print(f"Factory: Building DeepSeek V3 (TorchTitan) on TP {tp_rank}/{tp_size}")
    print(f"{'=' * 70}\n")

    # Build model
    model = DeepSeekV3TorchTitanForCausalLM(
        vllm_config=vllm_config, parallel_context=parallel_context
    )

    # Convert to dtype if specified
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
    print("\n🧪 Testing model creation...")

    model = DeepSeekV3TorchTitanForCausalLM()

    print("\n✓ Model created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n🧪 Testing forward pass...")

    batch_size = 2
    seq_len = 16
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        hidden_states = model(input_ids)
        logits = model.compute_logits(hidden_states)

    print("\n✓ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Logits shape: {logits.shape}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
