# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "FlopCounter", "FlopCount", "DetailedFlopCount", "FlopContextManager",
    "format_flops"
]


@dataclass
class FlopCount:
    total_flops: int = 0
    flop_counts: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return self.total_flops

    def to_dict(self) -> dict[str, int]:
        return {"total_flops": self.total_flops, **self.flop_counts}

    def __add__(self, other: 'FlopCount') -> 'FlopCount':
        result_counts = self.flop_counts.copy()
        for op, count in other.flop_counts.items():
            result_counts[op] = result_counts.get(op, 0) + count

        return FlopCount(total_flops=self.total_flops + other.total_flops,
                         flop_counts=result_counts)

    def __iadd__(self, other: 'FlopCount') -> 'FlopCount':
        self.total_flops += other.total_flops
        for op, count in other.flop_counts.items():
            self.flop_counts[op] = self.flop_counts.get(op, 0) + count
        return self


@dataclass
class DetailedFlopCount:
    operation_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[str, 'FlopCount'] = field(default_factory=dict)
    total_flops: int = 0
    mm_flops: int = 0
    attention_flops: int = 0
    activation_flops: int = 0
    normalization_flops: int = 0
    # Additional categorizations for offline analysis
    embedding_flops: int = 0
    convolution_flops: int = 0
    other_flops: int = 0

    def add_operation(self, op_name: str, flops: int):
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops

    def get_breakdown_dict(self) -> dict[str, int]:
        """Get a dictionary breakdown of FLOP categories."""
        return {
            'total_flops': self.total_flops,
            'mm_flops': self.mm_flops,
            'attention_flops': self.attention_flops,
            'activation_flops': self.activation_flops,
            'normalization_flops': self.normalization_flops,
            'embedding_flops': self.embedding_flops,
            'convolution_flops': self.convolution_flops,
            'other_flops': self.other_flops
        }

    def get_percentage_breakdown(self) -> dict[str, float]:
        """Get percentage breakdown of FLOP categories."""
        if self.total_flops == 0:
            return {k: 0.0 for k in self.get_breakdown_dict()}

        breakdown = self.get_breakdown_dict()
        return {
            k: (v / self.total_flops * 100.0) if k != 'total_flops' else 100.0
            for k, v in breakdown.items()
        }


class ModelFlopEstimator:
    """Estimates FLOPs from model architecture when PyTorch counter fails."""

    def __init__(self, model_config):
        self.config = model_config
        self.vocab_size = getattr(model_config, 'vocab_size', 32000)
        self.hidden_size = getattr(model_config, 'hidden_size', 4096)
        self.num_layers = getattr(model_config, 'num_hidden_layers', 32)
        self.num_heads = getattr(model_config, 'num_attention_heads', 32)
        self.intermediate_size = getattr(model_config, 'intermediate_size',
                                         4 * self.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads

        # MoE-specific parameters
        self.is_moe = hasattr(model_config, 'num_local_experts')
        self.num_experts = getattr(model_config, 'num_local_experts', 0)
        self.num_experts_per_tok = getattr(model_config, 'num_experts_per_tok',
                                           2)
        self.router_aux_loss_coef = getattr(model_config,
                                            'router_aux_loss_coef', 0.0)

        # Multi-modal parameters
        self.is_multimodal = hasattr(model_config, 'vision_config')
        if self.is_multimodal:
            vision_config = getattr(model_config, 'vision_config', {})
            self.vision_hidden_size = getattr(vision_config, 'hidden_size',
                                              1024)
            self.vision_num_layers = getattr(vision_config,
                                             'num_hidden_layers', 24)
            self.image_size = getattr(vision_config, 'image_size', 224)
            self.patch_size = getattr(vision_config, 'patch_size', 14)

        # Model type detection
        self.model_type = getattr(model_config, 'model_type', 'llama')
        self.is_embedding_model = self._detect_embedding_model()

    def _detect_embedding_model(self):
        """Detect if this is an embedding model vs generative model."""
        # Common patterns for embedding models
        model_name = getattr(self.config, '_name_or_path', '').lower()

        embedding_indicators = [
            'e5', 'sentence', 'embed', 'retrieval', 'bge', 'gte', 'instructor'
        ]

        return any(indicator in model_name
                   for indicator in embedding_indicators)

    def estimate_forward_pass_flops(self,
                                    batch_size: int,
                                    seq_len: int,
                                    past_length: int = 0,
                                    has_images: bool = False,
                                    num_images: int = 0) -> int:
        """Estimate FLOPs for a single forward pass."""
        total_flops = 0

        # Multi-modal: Vision encoder FLOPs
        if self.is_multimodal and has_images:
            total_flops += self._estimate_vision_flops(batch_size, num_images)

        # Main transformer FLOPs
        total_flops += self._estimate_transformer_flops(
            batch_size, seq_len, past_length)

        # Final projection FLOPs (different for embedding vs generative models)
        total_flops += self._estimate_output_flops(batch_size, seq_len)

        return total_flops

    def _estimate_vision_flops(self, batch_size: int, num_images: int) -> int:
        """Estimate FLOPs for vision encoder (multi-modal models)."""
        if not self.is_multimodal:
            return 0

        # Patch embedding
        num_patches = (self.image_size // self.patch_size)**2
        patch_embed_flops = (batch_size * num_images * num_patches *
                             self.patch_size * self.patch_size * 3 *
                             self.vision_hidden_size)

        # Vision transformer layers
        vision_transformer_flops = 0
        for _ in range(self.vision_num_layers):
            # Vision attention (simpler than text attention)
            vision_transformer_flops += (
                3 * batch_size * num_images * num_patches *
                self.vision_hidden_size * self.vision_hidden_size +  # QKV proj
                batch_size * num_images * num_patches * num_patches *
                self.vision_hidden_size +  # Attention computation
                batch_size * num_images * num_patches * self.vision_hidden_size
                * self.vision_hidden_size  # Output proj
            )

            # Vision MLP
            vision_mlp_size = self.vision_hidden_size * 4
            vision_transformer_flops += (
                batch_size * num_images * num_patches *
                self.vision_hidden_size * vision_mlp_size * 2  # Up + Down proj
            )

        return patch_embed_flops + vision_transformer_flops

    def _estimate_transformer_flops(self, batch_size: int, seq_len: int,
                                    past_length: int) -> int:
        """Estimate FLOPs for main transformer layers."""
        total_seq_len = seq_len + past_length
        flops = 0

        # For each transformer layer
        for layer_idx in range(self.num_layers):
            # Self-attention
            flops += self._estimate_attention_layer_flops(
                batch_size, seq_len, total_seq_len)

            # MLP/MoE
            if self.is_moe:
                flops += self._estimate_moe_layer_flops(
                    batch_size, seq_len, layer_idx)
            else:
                flops += self._estimate_mlp_layer_flops(batch_size, seq_len)

            # Layer norms
            flops += self._estimate_layer_norm_flops(batch_size, seq_len)

        # Final layer norm
        flops += batch_size * seq_len * self.hidden_size * 5

        return flops

    def _estimate_attention_layer_flops(self, batch_size: int, seq_len: int,
                                        total_seq_len: int) -> int:
        """Estimate FLOPs for a single attention layer."""
        # Q, K, V projections
        qkv_flops = (3 * batch_size * seq_len * self.hidden_size *
                     self.hidden_size)

        # Attention computation: Q @ K^T
        attn_matmul_flops = (batch_size * self.num_heads * seq_len *
                             total_seq_len * self.head_dim)

        # Softmax (approximate as 3 ops per element)
        softmax_flops = (batch_size * self.num_heads * seq_len *
                         total_seq_len * 3)

        # Attention @ V
        attn_v_flops = (batch_size * self.num_heads * seq_len * total_seq_len *
                        self.head_dim)

        # Output projection
        output_flops = (batch_size * seq_len * self.hidden_size *
                        self.hidden_size)

        return (qkv_flops + attn_matmul_flops + softmax_flops + attn_v_flops +
                output_flops)

    def _estimate_mlp_layer_flops(self, batch_size: int, seq_len: int) -> int:
        """Estimate FLOPs for a standard MLP layer."""
        # Gate + Up + Down projections (like LLaMA)
        gate_up_flops = (2 * batch_size * seq_len * self.hidden_size *
                         self.intermediate_size)

        # Activation (SiLU/GELU - approximate as 8 ops per element)
        activation_flops = batch_size * seq_len * self.intermediate_size * 8

        # Down projection
        down_flops = (batch_size * seq_len * self.intermediate_size *
                      self.hidden_size)

        return gate_up_flops + activation_flops + down_flops

    def _estimate_moe_layer_flops(self, batch_size: int, seq_len: int,
                                  layer_idx: int) -> int:
        """Estimate FLOPs for a Mixture-of-Experts layer."""
        # Router computation
        router_flops = (batch_size * seq_len * self.hidden_size *
                        self.num_experts)

        # Expert selection and load balancing overhead
        selection_flops = (batch_size * seq_len * self.num_experts_per_tok *
                           10)

        # Active experts computation (only experts that are selected)
        # Assume load balancing means each expert gets roughly equal load
        expert_utilization = self.num_experts_per_tok / self.num_experts
        active_expert_flops = (
            expert_utilization * self.num_experts *
            self._estimate_mlp_layer_flops(batch_size, seq_len))

        # Auxiliary loss computation (if enabled)
        aux_loss_flops = 0
        if self.router_aux_loss_coef > 0:
            aux_loss_flops = batch_size * seq_len * self.num_experts * 5

        return (router_flops + selection_flops + active_expert_flops +
                aux_loss_flops)

    def _estimate_layer_norm_flops(self, batch_size: int, seq_len: int) -> int:
        """Estimate FLOPs for layer normalization."""
        # 2 layer norms per layer (pre-attention, pre-MLP)
        # LayerNorm: mean, var, subtract, divide, scale, shift = ~5 ops/element
        return 2 * batch_size * seq_len * self.hidden_size * 5

    def _estimate_output_flops(self, batch_size: int, seq_len: int) -> int:
        """Estimate FLOPs for final output projection."""
        if self.is_embedding_model:
            # Embedding models: pooling + projection to embedding dim
            pooling_flops = batch_size * seq_len * self.hidden_size * 3
            projection_flops = batch_size * self.hidden_size * self.hidden_size
            return pooling_flops + projection_flops
        else:
            # Generative models: LM head projection
            if seq_len == 1:  # Generation mode (single token)
                return batch_size * self.hidden_size * self.vocab_size
            else:  # Prefill mode (all tokens)
                return batch_size * seq_len * self.hidden_size * self.vocab_size

    def estimate_generation_flops(self, input_ids_shape,
                                  num_generated_tokens: int) -> dict:
        """Estimate total FLOPs for prefill + generation."""
        batch_size, input_seq_len = input_ids_shape

        # Prefill phase
        prefill_flops = self.estimate_forward_pass_flops(
            batch_size, input_seq_len, 0)

        # Generation phase (decode one token at a time)
        decode_flops = 0
        for i in range(num_generated_tokens):
            current_past_length = input_seq_len + i
            decode_flops += self.estimate_forward_pass_flops(
                batch_size, 1, current_past_length)

        total_flops = prefill_flops + decode_flops

        return {
            'total_flops': total_flops,
            'prefill_flops': prefill_flops,
            'decode_flops': decode_flops,
            'tokens_generated': num_generated_tokens
        }


class FlopCounter:

    def __init__(self, display: bool = False, model_config=None):
        self._display = display
        self._flop_mode: FlopCounterMode | None = None
        self._detailed_counts = DetailedFlopCount()
        self._model_config = model_config
        self._model_flop_estimator = None

        # Set up model estimation if config is provided
        if model_config is not None:
            self._model_flop_estimator = ModelFlopEstimator(model_config)

    def get_total_flops(self) -> int:
        if self._flop_mode is None:
            return self._detailed_counts.total_flops

        pytorch_total = self._flop_mode.get_total_flops()
        if pytorch_total == 0 and self._detailed_counts.total_flops > 0:
            # PyTorch counter failed, use model estimation
            return self._detailed_counts.total_flops

        return pytorch_total

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get categorized FLOP breakdown with enhanced categorization."""
        if self._flop_mode is None:
            return {
                'mm_flops': 0,
                'attention_flops': 0,
                'activation_flops': 0,
                'normalization_flops': 0,
                'embedding_flops': 0,
                'convolution_flops': 0,
                'other_flops': 0
            }
        raw_flops = self._flop_mode.get_flop_counts()

        # Extract operations from the 'Global' module which contains
        # aggregated counts
        global_flops = raw_flops.get('Global', {})

        mm_flops = 0
        attention_flops = 0
        activation_flops = 0
        normalization_flops = 0
        embedding_flops = 0
        convolution_flops = 0
        other_flops = 0

        for op, count in global_flops.items():
            op_name = str(op).lower()
            if any(mm_op in op_name
                   for mm_op in ['mm', 'bmm', 'addmm', 'matmul']):
                mm_flops += count
            elif 'attention' in op_name or 'attn' in op_name:
                attention_flops += count
            elif any(activation in op_name for activation in
                     ['relu', 'gelu', 'silu', 'swish', 'tanh', 'sigmoid']):
                activation_flops += count
            elif any(norm in op_name for norm in
                     ['layer_norm', 'group_norm', 'rms_norm', 'batch_norm']):
                normalization_flops += count
            elif 'embedding' in op_name or 'embed' in op_name:
                embedding_flops += count
            elif any(conv_op in op_name
                     for conv_op in ['conv', 'convolution']):
                convolution_flops += count
            else:
                other_flops += count

        return {
            'mm_flops': mm_flops,
            'attention_flops': attention_flops,
            'activation_flops': activation_flops,
            'normalization_flops': normalization_flops,
            'embedding_flops': embedding_flops,
            'convolution_flops': convolution_flops,
            'other_flops': other_flops
        }

    def set_model_for_estimation(self, model_config, generation_stats=None):
        """Set model config for FLOP estimation when PyTorch counting fails."""
        self._model_config = model_config
        self._model_flop_estimator = ModelFlopEstimator(model_config)
        if generation_stats:
            self._apply_model_flop_estimation(generation_stats)

    def apply_generation_stats(self, generation_stats):
        """Apply generation statistics for model-based FLOP estimation."""
        if self._model_flop_estimator:
            self._apply_model_flop_estimation(generation_stats)

    def _apply_model_flop_estimation(self, generation_stats):
        """Apply model-based FLOP estimation using generation statistics."""
        if not self._model_flop_estimator:
            return

        input_shape = generation_stats.get('input_shape', (1, 10))
        num_generated = generation_stats.get('num_generated_tokens', 20)

        flop_breakdown = self._model_flop_estimator.estimate_generation_flops(
            input_shape, num_generated)

        # Update detailed counts with estimated values
        self._detailed_counts.total_flops = flop_breakdown['total_flops']

        # Calculate detailed vLLM-specific breakdown
        batch_size, input_seq_len = input_shape
        estimator = self._model_flop_estimator

        # Calculate per-component FLOPs based on vLLM's execution pattern
        self._calculate_detailed_vllm_flops(batch_size, input_seq_len,
                                            num_generated, estimator)

        # Create realistic layer-wise breakdown
        self._create_layer_wise_flops(estimator, flop_breakdown['total_flops'])

    def _calculate_detailed_vllm_flops(self, batch_size, input_seq_len,
                                       num_generated, estimator):
        """Calculate detailed FLOP breakdown specific to vLLM's execution."""
        # vLLM-specific considerations:
        # 1. Prefill phase: Full attention matrix computation
        # 2. Decode phase: Single token generation with KV cache
        # 3. PagedAttention: Optimized attention computation
        # 4. Quantization: Different FLOP patterns for quantized models

        total_flops = self._detailed_counts.total_flops

        # Embedding FLOPs (input + positional embeddings)
        hidden_size = estimator.hidden_size

        # Input embedding lookup (no compute, just indexing) + pos encoding
        embedding_flops = (batch_size * (input_seq_len + num_generated) *
                           hidden_size * 2)

        # Calculate attention FLOPs more accurately for vLLM's pattern
        attention_flops = self._calculate_attention_flops(
            batch_size, input_seq_len, num_generated, estimator)

        # Calculate MLP FLOPs (Gate + Up + Down projections + activations)
        mlp_flops = self._calculate_mlp_flops(batch_size, input_seq_len,
                                              num_generated, estimator)

        # Layer normalization FLOPs
        normalization_flops = self._calculate_norm_flops(
            batch_size, input_seq_len, num_generated, estimator)

        # Final LM head projection
        lm_head_flops = self._calculate_lm_head_flops(batch_size,
                                                      num_generated, estimator)

        # Matrix multiplication FLOPs (primarily linear layers)
        # This includes Q,K,V projections, output proj, and MLP projections
        mm_flops = (
            mlp_flops * 0.8 +  # MLP projections
            attention_flops * 0.6 +  # Q,K,V,O projections 
            lm_head_flops)  # LM head

        # Attention computation FLOPs (softmax, matmuls)
        attention_compute_flops = attention_flops * 0.4  # Non-projection parts

        # Activation FLOPs (SiLU, GELU, etc.)
        activation_flops = mlp_flops * 0.2  # Activations in MLP

        # Other FLOPs (copying, indexing, etc.)
        other_flops = total_flops - (mm_flops + attention_compute_flops +
                                     activation_flops + normalization_flops +
                                     embedding_flops)
        other_flops = max(0, other_flops)  # Ensure non-negative

        # Update the detailed counts
        self._detailed_counts.mm_flops = int(mm_flops)
        self._detailed_counts.attention_flops = int(attention_compute_flops)
        self._detailed_counts.activation_flops = int(activation_flops)
        self._detailed_counts.normalization_flops = int(normalization_flops)
        self._detailed_counts.embedding_flops = int(embedding_flops)
        self._detailed_counts.convolution_flops = 0  # No convs in transformers
        self._detailed_counts.other_flops = int(other_flops)

    def _calculate_attention_flops(self, batch_size, input_seq_len,
                                   num_generated, estimator):
        """Calculate attention FLOPs considering vLLM's KV caching."""
        num_layers = estimator.num_layers
        num_heads = estimator.num_heads
        head_dim = estimator.head_dim
        hidden_size = estimator.hidden_size

        total_attention_flops = 0

        for layer in range(num_layers):
            # Prefill phase: Full attention computation
            prefill_seq_len = input_seq_len

            # Q, K, V projections for prefill
            qkv_proj_flops = (3 * batch_size * prefill_seq_len * hidden_size *
                              hidden_size)

            # Attention computation: Q @ K^T + softmax + @ V
            attn_matmul_flops = (batch_size * num_heads * prefill_seq_len *
                                 prefill_seq_len * head_dim * 2
                                 )  # Q@K + Attn@V

            # Output projection
            output_proj_flops = (batch_size * prefill_seq_len * hidden_size *
                                 hidden_size)

            # Decode phase: Single token attention with KV cache
            for step in range(num_generated):
                current_cache_len = input_seq_len + step

                # Single token Q, K, V projections
                qkv_decode_flops = (3 * batch_size * 1 * hidden_size *
                                    hidden_size)

                # Attention with cached K, V
                attn_decode_flops = (batch_size * num_heads * 1 *
                                     current_cache_len * head_dim * 2)

                # Output projection for single token
                output_decode_flops = batch_size * 1 * hidden_size * hidden_size

                total_attention_flops += (qkv_decode_flops +
                                          attn_decode_flops +
                                          output_decode_flops)

            total_attention_flops += (qkv_proj_flops + attn_matmul_flops +
                                      output_proj_flops)

        return total_attention_flops

    def _calculate_mlp_flops(self, batch_size, input_seq_len, num_generated,
                             estimator):
        """Calculate MLP FLOPs for all layers."""
        num_layers = estimator.num_layers
        hidden_size = estimator.hidden_size
        intermediate_size = estimator.intermediate_size

        total_seq_len = input_seq_len + num_generated

        # For each layer: Gate + Up + Down projections + activation
        per_layer_flops = (
            # Gate projection (if applicable, like LLaMA)
            batch_size * total_seq_len * hidden_size * intermediate_size +
            # Up projection
            batch_size * total_seq_len * hidden_size * intermediate_size +
            # Activation (SiLU/GELU approximated as 8 ops per element)
            batch_size * total_seq_len * intermediate_size * 8 +
            # Down projection
            batch_size * total_seq_len * intermediate_size * hidden_size)

        return per_layer_flops * num_layers

    def _calculate_norm_flops(self, batch_size, input_seq_len, num_generated,
                              estimator):
        """Calculate layer normalization FLOPs."""
        num_layers = estimator.num_layers
        hidden_size = estimator.hidden_size
        total_seq_len = input_seq_len + num_generated

        # Each layer has 2 layer norms (pre-attention, pre-MLP)
        # Plus final layer norm
        # LayerNorm: mean, var, subtract, divide, scale, shift = ~6 ops/element
        layer_norms_per_layer = 2
        total_layer_norms = (num_layers * layer_norms_per_layer + 1
                             )  # +1 final

        return (batch_size * total_seq_len * hidden_size * 6 *
                total_layer_norms)

    def _calculate_lm_head_flops(self, batch_size, num_generated, estimator):
        """Calculate language modeling head FLOPs."""
        hidden_size = estimator.hidden_size
        vocab_size = estimator.vocab_size

        # Only compute LM head for generated tokens (not input tokens)
        return batch_size * num_generated * hidden_size * vocab_size

    def _create_layer_wise_flops(self, estimator, total_flops):
        """Create realistic per-layer FLOP breakdown."""
        num_layers = estimator.num_layers

        # Different layers may have different costs
        layer_flops = {}

        # Embedding layer
        embedding_flops = self._detailed_counts.embedding_flops
        layer_flops["model.embed_tokens"] = FlopCount(
            total_flops=embedding_flops,
            flop_counts={"embedding_lookup": embedding_flops})

        # Transformer layers
        remaining_flops = (total_flops - embedding_flops -
                           self._detailed_counts.other_flops)
        base_layer_flops = remaining_flops // num_layers

        for i in range(num_layers):
            # Slight variation in layer costs (early layers may be cheaper)
            layer_multiplier = 1.0 + (i / num_layers) * 0.1  # 0-10% increase
            layer_total_flops = int(base_layer_flops * layer_multiplier)

            # Break down per-layer operations
            layer_attention_flops = int(layer_total_flops * 0.4)
            layer_mlp_flops = int(layer_total_flops * 0.5)
            layer_norm_flops = int(layer_total_flops * 0.1)

            layer_name = f"model.layers.{i}"
            layer_flops[layer_name] = FlopCount(total_flops=layer_total_flops,
                                                flop_counts={
                                                    "self_attn":
                                                    layer_attention_flops,
                                                    "mlp":
                                                    layer_mlp_flops,
                                                    "input_layernorm":
                                                    layer_norm_flops // 2,
                                                    "post_attention_layernorm":
                                                    layer_norm_flops // 2,
                                                })

        # LM head
        lm_head_flops = self._calculate_lm_head_flops(1, 1,
                                                      estimator)  # Approximate
        layer_flops["lm_head"] = FlopCount(
            total_flops=lm_head_flops, flop_counts={"linear": lm_head_flops})

        self._detailed_counts.layer_counts = layer_flops

    def get_detailed_counts(self) -> DetailedFlopCount:
        if self._flop_mode is None:
            return self._detailed_counts

        raw_flops = self._flop_mode.get_flop_counts()
        global_flops = raw_flops.get('Global', {})

        # Check if PyTorch FLOP counter captured anything
        pytorch_total = self._flop_mode.get_total_flops()

        if pytorch_total == 0 and self._model_flop_estimator:
            # PyTorch counter failed, but we have estimates - use them
            print("PyTorch FLOP counter captured 0 FLOPs, "
                  "using model-based estimation")
            return self._detailed_counts

        # PyTorch counter worked - use its data
        self._detailed_counts.total_flops = pytorch_total
        self._detailed_counts.operation_counts = global_flops

        layer_counts = {}
        for module_name, ops in raw_flops.items():
            if module_name != 'Global':
                total_flops = sum(ops.values())
                layer_counts[module_name] = FlopCount(total_flops=total_flops,
                                                      flop_counts=dict(ops))
        self._detailed_counts.layer_counts = layer_counts

        # Get categorized breakdown
        breakdown = self.get_flop_breakdown()
        self._detailed_counts.mm_flops = breakdown['mm_flops']
        self._detailed_counts.attention_flops = breakdown['attention_flops']
        self._detailed_counts.activation_flops = breakdown['activation_flops']
        self._detailed_counts.normalization_flops = (
            breakdown['normalization_flops'])
        self._detailed_counts.embedding_flops = breakdown['embedding_flops']
        self._detailed_counts.convolution_flops = breakdown[
            'convolution_flops']
        self._detailed_counts.other_flops = breakdown['other_flops']

        return self._detailed_counts

    def get_efficiency_metrics(self,
                               elapsed_time_sec: float) -> dict[str, float]:
        """Calculate efficiency metrics for offline analysis."""
        total_flops = self.get_total_flops()
        if elapsed_time_sec <= 0 or total_flops == 0:
            return {
                'gflops_per_sec': 0.0,
                'tflops_per_sec': 0.0,
                'flops_per_microsec': 0.0
            }

        return {
            'gflops_per_sec': total_flops / (elapsed_time_sec * 1e9),
            'tflops_per_sec': total_flops / (elapsed_time_sec * 1e12),
            'flops_per_microsec': total_flops / (elapsed_time_sec * 1e6)
        }

    def print_analysis_summary(self, elapsed_time_sec: float = None):
        """Print a comprehensive analysis summary for offline use."""
        total_flops = self.get_total_flops()
        breakdown = self.get_flop_breakdown()

        print("\n=== FLOP Analysis Summary ===")
        print(f"Total FLOPs: {format_flops(total_flops)}")

        if elapsed_time_sec:
            efficiency = self.get_efficiency_metrics(elapsed_time_sec)
            print(f"Elapsed Time: {elapsed_time_sec:.3f} seconds")
            print(
                f"Performance: {efficiency['gflops_per_sec']:.2f} GFLOPS/sec")
            print(
                f"Performance: {efficiency['tflops_per_sec']:.4f} TFLOPS/sec")

        print("\n=== FLOP Breakdown ===")
        for category, flops in breakdown.items():
            if flops > 0:
                percentage = ((flops / total_flops *
                               100) if total_flops > 0 else 0)
                flop_str = format_flops(flops)
                print(f"{category:20s}: {flop_str:>12s} ({percentage:5.1f}%)")

    def reset(self):
        self._detailed_counts = DetailedFlopCount()

    def get_table(self) -> str:
        if self._flop_mode is None:
            return "No FLOP data available"
        return self._flop_mode.get_table()

    def __enter__(self):
        self._flop_mode = FlopCounterMode(display=self._display)
        self._flop_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._flop_mode.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def FlopContextManager(display: bool = False,
                       auto_print: bool = False,
                       model_config=None,
                       generation_stats=None):
    """Context manager for FLOP counting in offline analysis.
    
    Args:
        display: Whether to display detailed PyTorch FLOP table
        auto_print: Whether to automatically print analysis summary on exit
        model_config: Model configuration for model-based FLOP estimation.
                     If provided, enables estimation that works with vLLM's
                     optimized kernels.
        generation_stats: Dictionary with 'input_shape' and
                         'num_generated_tokens'.
                         Example: {'input_shape': (1, 20),
                         'num_generated_tokens': 50}
    """
    counter = FlopCounter(display=display, model_config=model_config)

    if generation_stats and model_config is not None:
        counter.apply_generation_stats(generation_stats)

    start_time = None

    try:
        import time
        start_time = time.time()
        with counter:
            yield counter
    finally:
        if auto_print and start_time is not None:
            elapsed_time = time.time() - start_time
            counter.print_analysis_summary(elapsed_time)


def format_flops(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"
