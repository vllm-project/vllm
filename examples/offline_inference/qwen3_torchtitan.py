# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 + TorchTitan integration with vLLM's TrainableFlashAttention.

This demonstrates the minimal-surgery pattern:
- Import TorchTitan's Qwen3Model as-is
- Replace ONLY attention layers with TrainableFlashAttention
- Let TorchTitan handle all parallelism (TP, embeddings, MLPs, etc.)
- No vLLM parallel layers used

IMPORTANT: TorchTitan imports are deferred to avoid CUDA initialization
before vLLM's multiprocessing fork.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.trainable_attention import TrainableFlashAttention
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


def replace_attention_with_vllm_trainable(model) -> None:
    """
    Replace TorchTitan's attention layers with vLLM's TrainableFlashAttention.

    This is module surgery - we replace ONLY the attention, keeping everything
    else from TorchTitan (embeddings, MLPs, norms, etc.).
    """
    print("\n" + "=" * 70)
    print("Performing Module Surgery: Replacing Attention Layers")
    print("=" * 70)

    # Qwen3Model uses ModuleDict with string keys ("0", "1", ...)
    num_layers = len(model.layers)

    for layer_idx, (layer_key, layer) in enumerate(model.layers.items()):
        # Extract attention config from TorchTitan attention
        old_attn = layer.attention

        # Infer hidden_size from weight shapes (wq.weight.shape = [out_features, in_features])
        # where in_features is the actual model hidden_size
        hidden_size = old_attn.wq.weight.shape[1]  # Input dimension
        num_heads = old_attn.n_heads
        # Extract num_kv_heads for GQA support (Qwen3 uses GQA)
        num_kv_heads = old_attn.n_kv_heads if hasattr(old_attn, 'n_kv_heads') else num_heads
        head_dim = old_attn.head_dim

        # Check if Qwen3 uses QK normalization
        use_qk_norm = hasattr(old_attn, 'q_norm') and old_attn.q_norm is not None

        # Create vLLM TrainableFlashAttention with TorchTitan-compatible settings
        new_attn = TrainableFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,  # Important for GQA!
            head_dim=head_dim,
            use_fused_qkv=False,  # Use separate wq/wk/wv for TorchTitan compatibility
            use_qk_norm=use_qk_norm,  # Match Qwen3's QK normalization
        )

        # Transfer weights from TorchTitan → vLLM
        # Both use wq, wk, wv, wo (same naming!)
        new_attn.wq.weight.data.copy_(old_attn.wq.weight.data)
        new_attn.wk.weight.data.copy_(old_attn.wk.weight.data)
        new_attn.wv.weight.data.copy_(old_attn.wv.weight.data)
        new_attn.wo.weight.data.copy_(old_attn.wo.weight.data)

        # Transfer QK norm weights if present
        if use_qk_norm:
            new_attn.q_norm.weight.data.copy_(old_attn.q_norm.weight.data)
            new_attn.k_norm.weight.data.copy_(old_attn.k_norm.weight.data)

        # Replace attention module
        layer.attention = new_attn

        if layer_idx == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_key}: Replaced Attention")

    if num_layers > 2:
        print(f"  ... (replaced {num_layers - 2} more layers)")

    print(f"\n✓ Successfully replaced {num_layers} attention layers")
    print("=" * 70 + "\n")


class Qwen3TorchTitanForCausalLM(nn.Module):
    """
    Qwen3 model using TorchTitan's implementation with vLLM's trainable attention.

    This demonstrates the minimal-surgery integration pattern:
    - Use TorchTitan's Qwen3Model directly
    - Replace only attention with TrainableFlashAttention
    - Let TorchTitan handle all parallelism
    """

    supports_pp = False
    supports_multimodal = False

    def __init__(
        self,
        vllm_config=None,
        parallel_context: ParallelContext | None = None,
        **kwargs
    ):
        super().__init__()

        # IMPORTANT: Import TorchTitan HERE (after fork) to avoid CUDA init issues
        from torchtitan.models.qwen3.model.model import Qwen3Model
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

        self.parallel_context = parallel_context

        # Extract config from vLLM
        config = None
        if vllm_config is not None:
            if hasattr(vllm_config, "hf_config"):
                config = vllm_config.hf_config
            elif hasattr(vllm_config, "model_config") and hasattr(vllm_config.model_config, "hf_config"):
                config = vllm_config.model_config.hf_config

        if config is not None:
            # Get head_dim from config (don't compute from hidden_size // num_heads!)
            # Qwen3-0.6B uses head_dim=128 with hidden_size=1024, num_heads=16
            # So hidden_size != num_heads * head_dim for this model!
            hidden_size = getattr(config, "hidden_size", 2048)
            num_heads = getattr(config, "num_attention_heads", 16)
            head_dim = getattr(config, "head_dim", 128)  # Use config value, default to 128

            # Create TorchTitan Qwen3ModelArgs
            model_args = Qwen3ModelArgs(
                vocab_size=getattr(config, "vocab_size", 151936),
                dim=hidden_size,
                n_layers=getattr(config, "num_hidden_layers", 4),
                n_heads=num_heads,
                n_kv_heads=getattr(config, "num_key_value_heads", 2),
                head_dim=head_dim,  # Use config value
                hidden_dim=getattr(config, "intermediate_size", 11008),
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                max_seq_len=getattr(config, "max_position_embeddings", 8192),
                rope_theta=getattr(config, "rope_theta", 1000000.0),
                qk_norm=getattr(config, "qk_norm", True),
                # Note: We don't set enable_weight_tying - TorchTitan creates separate layers
            )
        else:
            # Default small config for testing
            hidden_size = 2048
            num_heads = 16
            head_dim = hidden_size // num_heads  # Compute correctly
            model_args = Qwen3ModelArgs(
                vocab_size=151936,
                dim=hidden_size,
                n_layers=4,
                n_heads=num_heads,
                n_kv_heads=2,
                head_dim=head_dim,  # Use computed value
                hidden_dim=11008,
                norm_eps=1e-6,
                max_seq_len=8192,
                rope_theta=1000000.0,
                qk_norm=True,
            )

        print("\n" + "=" * 70)
        print("Creating TorchTitan Qwen3 Model")
        print("=" * 70)
        print(f"  Layers: {model_args.n_layers}")
        print(f"  Hidden size: {model_args.dim}")
        print(f"  Heads: {model_args.n_heads}")
        print(f"  KV heads: {model_args.n_kv_heads}")
        print(f"  Head dim: {model_args.head_dim}")  # Debug: Show head_dim
        print(f"  Vocab size: {model_args.vocab_size}")
        print(f"  Max seq len: {model_args.max_seq_len}")

        # Create TorchTitan Qwen3 model
        self.model = Qwen3Model(model_args)

        # Perform module surgery: replace attention only
        replace_attention_with_vllm_trainable(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

        # IMPORTANT: Synchronize weights across TP ranks
        # Without this, each rank has different random weights!
        if parallel_context is not None:
            tp_size = parallel_context.get_tensor_parallel_world_size()
            tp_rank = parallel_context.get_tensor_parallel_rank()
            if tp_size > 1:
                print(f"\n🔄 Synchronizing weights across {tp_size} TP ranks...")
                print(f"   Rank {tp_rank}: Broadcasting from rank 0...")

                # Import distributed utilities
                import torch.distributed as dist

                # Broadcast all parameters from rank 0 to all ranks
                for name, param in self.model.named_parameters():
                    dist.broadcast(param.data, src=0)

                print(f"   Rank {tp_rank}: ✓ Weights synchronized")

        print("=" * 70 + "\n")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings (required by vLLM)."""
        return self.model.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len] or [total_tokens] (optional if inputs_embeds provided)
            positions: Position indices from vLLM for RoPE
            inputs_embeds: Pre-computed embeddings (optional, used by vLLM)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states before LM head
        """
        # Debug: Log positions for all forward calls
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1

        if self._forward_count <= 5 or (self._forward_count % 10 == 0):
            if positions is not None:
                unique_pos = torch.unique(positions)
                # Also log input_ids for decode steps (when positions > 5)
                if unique_pos[0] > 5 and input_ids is not None:
                    print(f"\n[FWD #{self._forward_count}] DECODE STEP")
                    print(f"  positions (full): {positions}")
                    print(f"  input_ids (full): {input_ids}")
                    print(f"  input_ids.shape: {input_ids.shape if input_ids is not None else None}")
                    print(f"  positions.shape: {positions.shape if positions is not None else None}")
                else:
                    print(f"\n[FWD #{self._forward_count}] positions unique: {unique_pos[:20]}, inputs_embeds: {inputs_embeds is not None}")
            else:
                print(f"\n[FWD #{self._forward_count}] positions: None, inputs_embeds: {inputs_embeds is not None}")

        # Store positions in forward context so TrainableFlashAttention can access them
        if positions is not None:
            try:
                from vllm.forward_context import get_forward_context
                forward_ctx = get_forward_context()
                # Store positions in a custom attribute
                forward_ctx._torchtitan_positions = positions
            except:
                pass

        # Get embeddings (either from inputs_embeds or by applying embedding layer)
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.model.tok_embeddings(input_ids)

        # Debug: Check shape
        if not hasattr(self, '_shape_debug'):
            self._shape_debug = True
            print(f"\n[WRAPPER DEBUG] Embeddings shape: {h.shape}")
            if input_ids is not None:
                print(f"[WRAPPER DEBUG] input_ids.shape: {input_ids.shape}")

        # Get RoPE cache (TorchTitan generates this internally)
        seqlen = h.shape[1] if h.dim() == 3 else h.shape[0]
        rope_cache = self.model.rope_cache[:seqlen]

        # Debug: Check rope_cache shape
        if self._shape_debug:
            print(f"[WRAPPER DEBUG] rope_cache.shape: {rope_cache.shape}")

        # Pass through transformer layers
        # TorchTitan signature: forward(x, freqs_cis, attention_masks)
        for layer_idx, layer in enumerate(self.model.layers.values()):
            # DEBUG: Log layer 0, 1, 2, and last layer
            should_debug = layer_idx in [0, 1, 2, 27]
            if should_debug and not hasattr(self, f'_layer_{layer_idx}_debug'):
                setattr(self, f'_layer_{layer_idx}_debug', True)
                if h.dim() == 2 and h.shape[0] > 1 and h.shape[0] < 100:  # flattened, non-warmup
                    print(f"\n[VLLM TRANSFORMER DEBUG] Layer {layer_idx} input: {h[0,:5]}")

            h = layer(h, rope_cache, None)  # None for attention_masks

            # DEBUG: Log after layer
            if should_debug and hasattr(self, f'_layer_{layer_idx}_debug') and h.dim() == 2 and h.shape[0] > 1 and h.shape[0] < 100:
                print(f"[VLLM TRANSFORMER DEBUG] After layer {layer_idx}: {h[0,:5]}")

        # Final norm
        h = self.model.norm(h)

        # DEBUG: Log final hidden states
        if hasattr(self, '_layer_0_debug') and h.dim() == 2 and h.shape[0] > 1 and h.shape[0] < 100:
            print(f"[VLLM WRAPPER DEBUG] Final hidden states (after norm): {h[0,:5]}")

        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute logits from hidden states (required by vLLM)."""
        # DEBUG: Always log to track calls
        if not hasattr(self, '_logits_call_count'):
            self._logits_call_count = 0
        self._logits_call_count += 1

        if self._logits_call_count <= 5 or self._logits_call_count % 10 == 0:
            print(f"\n[COMPUTE_LOGITS #{self._logits_call_count}] hidden_states.shape: {hidden_states.shape}")
            if hidden_states.shape[0] < 10:  # Only log for small batches (actual generation)
                print(f"[COMPUTE_LOGITS #{self._logits_call_count}] hidden_states[0,:5]: {hidden_states[0,:5]}")

        # DEBUG: Check LM head weights on first call
        if self._logits_call_count == 1:
            print(f"[LM HEAD CHECK] output.weight.shape: {self.model.output.weight.shape}")
            print(f"[LM HEAD CHECK] output.weight[0,:5]: {self.model.output.weight[0,:5]}")
            print(f"[LM HEAD CHECK] output.weight[:5,0]: {self.model.output.weight[:5,0]}")

        logits = self.model.output(hidden_states)

        if self._logits_call_count <= 5 and hidden_states.shape[0] < 10:
            print(f"[COMPUTE_LOGITS #{self._logits_call_count}] logits.shape: {logits.shape}")
            print(f"[COMPUTE_LOGITS #{self._logits_call_count}] logits[0,:10]: {logits[0,:10]}")
            print(f"[COMPUTE_LOGITS #{self._logits_call_count}] argmax(logits[0]): {logits[0].argmax().item()}")

        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HuggingFace checkpoint.

        Maps HF Qwen weight names → TorchTitan naming convention.
        """
        print("\n" + "=" * 70)
        print("Loading weights...")
        print("=" * 70)

        weights_list = list(weights_iter)
        if len(weights_list) == 0:
            print("  ⚠️  No weight files found - using random initialization")
            print("=" * 70 + "\n")
            return

        # HF → TorchTitan name mapping
        def map_weight_name(hf_name: str) -> str:
            """Map HuggingFace weight name to TorchTitan name."""
            # Embeddings
            if hf_name == "model.embed_tokens.weight":
                return "tok_embeddings.weight"
            # Output head
            if hf_name == "lm_head.weight":
                return "output.weight"
            # Final norm
            if hf_name == "model.norm.weight":
                return "norm.weight"

            # Layer weights
            if hf_name.startswith("model.layers."):
                # Extract layer number
                parts = hf_name.split(".")
                layer_idx = parts[2]  # model.layers.{idx}.xxx
                rest = ".".join(parts[3:])  # self_attn.q_proj.weight

                # Map attention weights (now handled by TrainableFlashAttention)
                if rest.startswith("self_attn."):
                    attn_part = rest.replace("self_attn.", "")
                    # Map q_proj → wq, k_proj → wk, v_proj → wv, o_proj → wo
                    attn_part = attn_part.replace("q_proj", "wq")
                    attn_part = attn_part.replace("k_proj", "wk")
                    attn_part = attn_part.replace("v_proj", "wv")
                    attn_part = attn_part.replace("o_proj", "wo")
                    return f"layers.{layer_idx}.attention.{attn_part}"

                # Map MLP weights
                if rest.startswith("mlp."):
                    mlp_part = rest.replace("mlp.", "")
                    mlp_part = mlp_part.replace("gate_proj", "w1")
                    mlp_part = mlp_part.replace("up_proj", "w3")
                    mlp_part = mlp_part.replace("down_proj", "w2")
                    return f"layers.{layer_idx}.feed_forward.{mlp_part}"

                # Map norms
                if rest == "input_layernorm.weight":
                    return f"layers.{layer_idx}.attention_norm.weight"
                if rest == "post_attention_layernorm.weight":
                    return f"layers.{layer_idx}.ffn_norm.weight"

            return None  # Unknown weight

        # Load weights
        param_dict = dict(self.model.named_parameters())
        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in weights_list:
            tt_name = map_weight_name(name)
            if tt_name is None:
                # print(f"  ⚠️  Skipping unknown weight: {name}")
                skipped_count += 1
                continue

            if tt_name not in param_dict:
                print(f"  ⚠️  Target parameter not found: {tt_name} (from {name})")
                skipped_count += 1
                continue

            param = param_dict[tt_name]
            if param.shape != loaded_weight.shape:
                print(f"  ⚠️  Shape mismatch for {tt_name}:")
                print(f"      Expected: {param.shape}, Got: {loaded_weight.shape}")
                skipped_count += 1
                continue

            # Load the weight
            param.data.copy_(loaded_weight)
            loaded_count += 1

            # DEBUG: Log lm_head loading
            if 'output' in tt_name or 'lm_head' in name:
                print(f"  ✓ Loaded {name} → {tt_name}")
                print(f"      Shape: {loaded_weight.shape}, First values: {loaded_weight.flatten()[:5]}")

                # Verify it was actually loaded
                if tt_name == "output.weight":
                    actual_param = param_dict[tt_name]
                    print(f"      After loading, param[0,:5]: {actual_param[0,:5]}")
                    # Move to same device for comparison
                    matches = torch.allclose(actual_param.cpu(), loaded_weight.cpu(), rtol=1e-3)
                    print(f"      Matches loaded weight: {matches}")

        print(f"  ✓ Loaded {loaded_count} weights")
        if skipped_count > 0:
            print(f"  ⚠️  Skipped {skipped_count} weights")
        print("=" * 70 + "\n")


def build_qwen3_torchtitan(
    vllm_config, parallel_context: ParallelContext
) -> nn.Module:
    """Factory function for vLLM ModelRegistry."""
    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()

    print(f"\n{'=' * 70}")
    print(f"Factory: Building Qwen3+TorchTitan on TP {tp_rank}/{tp_size}")
    print(f"{'=' * 70}\n")

    model = Qwen3TorchTitanForCausalLM(
        vllm_config=vllm_config,
        parallel_context=parallel_context
    )

    # Convert to dtype if specified
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "dtype"
    ):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model


# Register with vLLM's ModelRegistry
ModelRegistry.register_model("Qwen3TorchTitan", build_qwen3_torchtitan)
