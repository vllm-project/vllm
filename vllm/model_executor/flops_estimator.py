"""
Model Configuration P-Value Estimator
Estimate non-attention and attention calculation ratio for different model architectures
"""

import json
import argparse
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import logging

# Global logger: prefer vllm logger if available, else fallback to std logging
try:
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore
    logger = _vllm_init_logger(__name__)
except Exception:  # noqa: BLE001
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class BaseModelEstimator(ABC):
    """Abstract base class for model p-value estimators"""
    
    def __init__(self, config: Dict):
        """
        Initialize estimator with model config
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        # Predefine attributes that may be populated by subclasses
        self.effective_kv_dim = None
        self.effective_q_dim = None
        self._parse_common_config()
        self._parse_architecture_specific_config()
        self._validate_config()
        # Baseline and attention-equivalent tokens configured from scheduler
        self.attn_equiv_baseline_c: int = 0
        self.attn_equiv_tokens: int = 0
    
    def _parse_common_config(self):
        """Parse common configuration parameters"""
        self.model_type = self.config.get('model_type', 'unknown')
        self.hidden_size = self.config.get('hidden_size')
        # Some architectures may override or refine this later.
        self.intermediate_size = self.config.get('intermediate_size')
        self.num_attention_heads = self.config.get('num_attention_heads')
        self.num_key_value_heads = self.config.get('num_key_value_heads', self.num_attention_heads)
        self.num_hidden_layers = self.config.get('num_hidden_layers')
        self.vocab_size = self.config.get('vocab_size', 32000)
        # Optional calibration factor to account for softmax/RoPE and kernel overheads
        self.attn_flops_scale = float(self.config.get('attn_flops_scale', 1.0))
        
        # Calculate common derived values
        if self.hidden_size and self.num_attention_heads:
            self.head_dim = self.hidden_size // self.num_attention_heads
            self.gqa_ratio = self.num_attention_heads // self.num_key_value_heads
    
    @abstractmethod
    def _parse_architecture_specific_config(self):
        """Parse architecture-specific configuration parameters"""
        raise NotImplementedError
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        if not all([self.hidden_size, self.num_attention_heads, self.num_hidden_layers]):
            raise ValueError(f"Missing required parameters for {self.model_type} model")
    
    @abstractmethod
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """
        Calculate attention FLOPs
        
        Args:
            chunk_size: Input chunk size
            seq_len: Total sequence length
            
        Returns:
            (projection_flops, attention_computation_flops)
        """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """
        Calculate FFN FLOPs
        
        Args:
            chunk_size: Input chunk size  
            layer_idx: Layer index (for models with varying layer types)
            
        Returns:
            FFN FLOPs
        """
        raise NotImplementedError
    
    def calculate_other_flops(self, chunk_size: int) -> int:
        """Calculate other FLOPs (LayerNorm, residual connections, etc.)"""
        # Standard: 2 LayerNorms per layer
        norm_flops = chunk_size * self.hidden_size * 2
        
        # Residual connections
        residual_flops = chunk_size * self.hidden_size * 2
        
        return norm_flops + residual_flops

    # -------------------------
    # Shared FLOPs helpers
    # -------------------------
    def _standard_attention_flops(self, 
                                  chunk_size: int, 
                                  seq_len: int, 
                                  *, 
                                  kv_dim: Optional[int] = None, 
                                  effective_seq_len: Optional[int] = None
                                  ) -> Tuple[int, int]:
        """
        Standard scaled dot-product attention FLOPs with GQA.

        Args:
            chunk_size: Number of tokens processed this step.
            seq_len: Total attention context length (KV cache + chunk).
            kv_dim: Dimension used for K/V projections per token.
                    Defaults to hidden_size // gqa_ratio.
            effective_seq_len: Effective attention length if using sliding
                               window or other sparsity; defaults to seq_len.
        Returns:
            (projection_flops, attention_compute_flops)
        """
        kv_dim = kv_dim if kv_dim is not None else (self.hidden_size // self.gqa_ratio)
        eff_len = effective_seq_len if effective_seq_len is not None else seq_len

        q_flops = chunk_size * self.hidden_size * self.hidden_size
        kv_flops = chunk_size * self.hidden_size * kv_dim * 2
        score_flops = chunk_size * eff_len * self.hidden_size
        value_flops = chunk_size * eff_len * self.hidden_size
        out_flops = chunk_size * self.hidden_size * self.hidden_size

        return q_flops + kv_flops + out_flops, score_flops + value_flops

    def _ffn_swiglu_like_flops(self, chunk_size: int, *, intermediate_size: Optional[int] = None) -> int:
        """
        FLOPs for FFN with gate+up and down projections (SwiGLU/GeGLU).
        The activation doesn't change matmul FLOPs.
        """
        inter_size = intermediate_size if intermediate_size is not None else self.intermediate_size
        # Support configs that use `ffn_dim` (e.g., OPT) without guessing.
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None or self.hidden_size is None:
            raise ValueError("Missing FFN dimension: expected 'intermediate_size' or 'ffn_dim' in config")
        up_gate_flops = chunk_size * self.hidden_size * inter_size * 2
        down_flops = chunk_size * inter_size * self.hidden_size
        return up_gate_flops + down_flops

    # -------------------------
    # Dynamic chunk sizing helper (DCPP-style)
    # -------------------------
    def configure_baseline_from_scheduler(self, scheduler_config) -> None:
        """Configure baseline chunk and attention-equivalent tokens using scheduler config."""
        baseline_c = getattr(scheduler_config, 'long_prefill_token_threshold', None)
        if baseline_c is None or baseline_c <= 0:
            baseline_c = 2048
        self.attn_equiv_baseline_c = int(baseline_c)
        self.attn_equiv_tokens = int(self.estimate_flops_ratio_scaled(self.attn_equiv_baseline_c, 0))

    def compute_chunk_size_with_overhead(self,
                                         hist_seq_len: int,
                                         seq_len: int,
                                         chunk_size: int,
                                         block_size: int) -> tuple[int, int]:
        """
        Compute shortened chunk to offset long-KV attention overhead.

        Returns (dcpp_chunk, scheduled_chunk):
        - dcpp_chunk: actual reduced chunk length to run this step (block-aligned)
        - scheduled_chunk: the baseline chunk length used for budgeting/accounting
        """
        # Must be configured once by the scheduler
        assert self.attn_equiv_tokens > 0, "Call configure_baseline_from_scheduler() before chunk computation."

        target_time = chunk_size * (self.attn_equiv_tokens + chunk_size)
        discriminant = (self.attn_equiv_tokens + hist_seq_len) ** 2 + 4 * target_time
        dcpp_chunk = int((-(self.attn_equiv_tokens + hist_seq_len) + discriminant ** 0.5) / 2)
        # align to block size
        dcpp_chunk = (dcpp_chunk + block_size - 1) // block_size * block_size
        scheduled_chunk = dcpp_chunk
        # do not exceed remaining tokens
        dcpp_chunk = min(dcpp_chunk, seq_len - hist_seq_len)
        return dcpp_chunk, scheduled_chunk
    
    def estimate_flops_ratio(self, chunk_size: int, kv_cache_len: int, layer_idx: Optional[int] = None) -> float:
        """
        Estimate FLOPs ratio (non-attention / attention)
        
        Args:
            chunk_size: Current chunk size
            kv_cache_len: KV cache length (number of processed tokens)
            layer_idx: Specific layer index
        
        Returns:
            non-attention FLOPs / attention FLOPs
        """
        seq_len = kv_cache_len + chunk_size
        
        proj_flops, attn_flops = self.calculate_attention_flops(chunk_size, seq_len)
        # Apply global attention calibration to better match wall-clock
        attn_flops = attn_flops * self.attn_flops_scale
        ffn_flops = self.calculate_ffn_flops(chunk_size, layer_idx)
        other_flops = self.calculate_other_flops(chunk_size)
        
        non_attn_flops = ffn_flops + other_flops + proj_flops
        
        if attn_flops == 0:
            return float('inf')
        
        return non_attn_flops / attn_flops
    
    def estimate_flops_ratio_average(self, chunk_size: int, kv_cache_len: int) -> float:
        """Average FLOPs ratio across layers (default: uniform layers)."""
        return self.estimate_flops_ratio(chunk_size, kv_cache_len)
    
    def estimate_flops_ratio_scaled(self, chunk_size: int, kv_cache_len: int) -> float:
        """Estimate FLOPs ratio scaled by chunk size (ratio * chunk_size)."""
        return self.estimate_flops_ratio_average(chunk_size, kv_cache_len) * chunk_size

    # Backward-compatible aliases
    def estimate_p(self, chunk_size: int, kv_cache_len: int, layer_idx: Optional[int] = None) -> float:
        return self.estimate_flops_ratio(chunk_size, kv_cache_len, layer_idx)

    def estimate_p_average(self, chunk_size: int, kv_cache_len: int) -> float:
        return self.estimate_flops_ratio_average(chunk_size, kv_cache_len)

    def estimate_p_times_c(self, chunk_size: int, kv_cache_len: int) -> float:
        return self.estimate_flops_ratio_scaled(chunk_size, kv_cache_len)
    
    def print_model_info(self):
        """Print model information (common + subclass extras)."""
        logger.info("Model Type: %s", self.model_type)
        logger.info("Hidden Size: %s", self.hidden_size)
        logger.info("Attention Heads: %s", self.num_attention_heads)
        logger.info("KV Heads: %s (GQA ratio: %s:1)", self.num_key_value_heads, self.gqa_ratio)
        logger.info("Layers: %s", self.num_hidden_layers)
        logger.info("Head Dimension: %s", self.head_dim)
        # Subclass-provided extra details
        for line in self._extra_model_info():
            logger.info("%s", line)
    
    def get_flops_ratio_table(self, chunk_sizes: List[int] = None, 
                              kv_cache_lens: List[int] = None) -> Dict:
        """Generate FLOPs ratio table for different chunk sizes and cache lengths"""
        if chunk_sizes is None:
            chunk_sizes = [512, 1024, 2048, 4096]
        
        if kv_cache_lens is None:
            kv_cache_lens = [0, 2048, 8192, 16384, 32768, 65536, 98304]
        
        table = {}
        for chunk_size in chunk_sizes:
            table[chunk_size] = {}
            for kv_len in kv_cache_lens:
                ratio = self.estimate_flops_ratio_average(chunk_size, kv_len)
                table[chunk_size][kv_len] = ratio
        
        return table
    
    def print_flops_ratio_table(self, chunk_sizes: List[int] = None, 
                                kv_cache_lens: List[int] = None):
        """Print FLOPs ratio table"""
        table = self.get_flops_ratio_table(chunk_sizes, kv_cache_lens)
        
        logger.info("\nFLOPs Ratio Table (non-attention / attention)")
        logger.info("%s", "=" * 80)
        header = ["KV Cache"] + [f"C={cs:<6}" for cs in sorted(table.keys())]
        logger.info("%s", "".join(f"{h:<10}" for h in header))
        logger.info("%s", "-" * 80)
        
        all_kv_lens = sorted(next(iter(table.values())).keys())
        for kv_len in all_kv_lens:
            row = [f"{kv_len:<10}"]
            for chunk_size in sorted(table.keys()):
                ratio = table[chunk_size][kv_len]
                row.append(f"{ratio:8.2f}")
            logger.info("%s", "".join(row))
        # Allow subclasses to append extra table-related info
        self._extra_p_table_info()

    # Backward-compatible aliases
    def get_p_table(self, chunk_sizes: List[int] = None, 
                    kv_cache_lens: List[int] = None) -> Dict:
        return self.get_flops_ratio_table(chunk_sizes, kv_cache_lens)

    def print_p_table(self, chunk_sizes: List[int] = None, 
                      kv_cache_lens: List[int] = None):
        return self.print_flops_ratio_table(chunk_sizes, kv_cache_lens)

    # -------------------------
    # Subclass hooks
    # -------------------------
    def _extra_model_info(self) -> List[str]:
        """Override to append subclass-specific info lines."""
        return []

    def _extra_p_table_info(self) -> None:
        """Override to print additional info after p-table (e.g., layer analysis)."""
        return None


class LlamaEstimator(BaseModelEstimator):
    """FLOP estimator for Llama models (standard transformer architecture)"""
    
    def _parse_architecture_specific_config(self):
        """Parse Llama-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')  # e.g., OPT family
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.hidden_size else 0
        
        # Llama uses SwiGLU activation (gate + up projections)
        self.has_gate_proj = True
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Standard transformer attention FLOPs (Llama)."""
        return self._standard_attention_flops(chunk_size, seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Llama (SwiGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        return [
            "Architecture: Standard Transformer (Llama)",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            "Activation: SwiGLU",
        ]


class QwenEstimator(BaseModelEstimator):
    """FLOP estimator for Qwen models"""
    
    def _parse_architecture_specific_config(self):
        """Parse Qwen-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
        
        # Qwen also uses SwiGLU but may have different parameter names
        self.has_gate_proj = True
        
        # Qwen-specific parameters
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        self.use_sliding_window = self.config.get('use_sliding_window', False)
        self.sliding_window = self.config.get('sliding_window')
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Qwen attention FLOPs with optional sliding window sparsity."""
        effective_seq_len = (min(seq_len, self.sliding_window)
                             if self.use_sliding_window and self.sliding_window
                             else seq_len)
        return self._standard_attention_flops(
            chunk_size, seq_len, effective_seq_len=effective_seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Qwen (SwiGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        lines = [
            "Architecture: Qwen Transformer",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            f"Max Position Embeddings: {self.max_position_embeddings}",
        ]
        if self.use_sliding_window:
            lines.append(f"Sliding Window: {self.sliding_window}")
        return lines


class DeepSeekEstimator(BaseModelEstimator):
    """FLOP estimator for DeepSeek models (supports MLA + MoE)"""
    
    def _parse_architecture_specific_config(self):
        """Parse DeepSeek-specific configuration"""
        # Standard FFN parameters
        self.intermediate_size = self.config.get('intermediate_size')
        
        # MLA (Multi-head Latent Attention) parameters
        self.kv_lora_rank = self.config.get('kv_lora_rank')
        self.q_lora_rank = self.config.get('q_lora_rank')
        self.qk_rope_head_dim = self.config.get('qk_rope_head_dim', 64)
        
        # MoE parameters
        self.moe_intermediate_size = self.config.get('moe_intermediate_size')
        self.first_k_dense_replace = self.config.get('first_k_dense_replace', 0)
        self.n_routed_experts = self.config.get('n_routed_experts', 0)
        self.num_experts_per_tok = self.config.get('num_experts_per_tok', 1)
        self.n_shared_experts = self.config.get('n_shared_experts', 0)
        
        # Architecture flags
        self.is_mla = self._is_mla_architecture()
        self.is_moe = self._is_moe_architecture()
        
        # Calculate effective dimensions
        self._calculate_effective_dims()
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
    
    def _is_mla_architecture(self) -> bool:
        """Check if model uses MLA (Multi-head Latent Attention)"""
        kv = self.kv_lora_rank
        q = self.q_lora_rank
        return (isinstance(kv, int) and kv > 0 and
                isinstance(q, int) and q > 0)
    
    def _is_moe_architecture(self) -> bool:
        """Check if model uses MoE (Mixture of Experts)"""
        return (self.moe_intermediate_size is not None and 
                self.n_routed_experts > 0)
    
    def _calculate_effective_dims(self):
        """Calculate effective dimensions for MLA"""
        if self.is_mla:
            # MLA uses compressed K/V representations. Do not add RoPE dim here
            # to avoid double-counting; RoPE cost is captured separately.
            self.effective_kv_dim = self.kv_lora_rank
            self.effective_q_dim = self.q_lora_rank
        else:
            # Standard dimensions
            self.effective_kv_dim = self.hidden_size // self.gqa_ratio
            self.effective_q_dim = self.hidden_size
    
    def _is_dense_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is dense (not MoE)"""
        return layer_idx < self.first_k_dense_replace
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate attention FLOPs with MLA support"""
        if self.is_mla:
            return self._calculate_mla_attention_flops(chunk_size, seq_len)
        else:
            return self._calculate_standard_attention_flops(chunk_size, seq_len)
    
    def _calculate_mla_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate FLOPs for MLA (Multi-head Latent Attention)"""
        # Q projections: q_a (to LoRA rank) + q_b (LoRA to heads)
        q_a_flops = chunk_size * self.hidden_size * self.q_lora_rank
        q_b_flops = chunk_size * self.q_lora_rank * (self.num_attention_heads * self.head_dim)
        
        # KV projections: kv_a (to LoRA rank) + kv_b (LoRA to compressed KV)
        kv_a_flops = chunk_size * self.hidden_size * self.effective_kv_dim
        kv_b_flops = chunk_size * self.effective_kv_dim * (self.num_key_value_heads * self.head_dim * 2)
        
        # Attention computation using compressed dimensions
        # MLA uses two components: compressed KV and RoPE
        # Score computation: Q^T @ K^C (compressed) + Q^TR @ K^R (RoPE)
        # The effective computation is with compressed dimensions, not full hidden_size
        compressed_score_flops = chunk_size * seq_len * self.effective_kv_dim
        rope_score_flops = chunk_size * seq_len * self.qk_rope_head_dim * self.num_attention_heads
        
        # Value computation: attention_weights @ V^C (compressed)
        value_flops = chunk_size * seq_len * self.effective_kv_dim
        
        # Output projection
        out_flops = chunk_size * self.hidden_size * self.hidden_size
        
        proj_flops = q_a_flops + q_b_flops + kv_a_flops + kv_b_flops + out_flops
        attn_flops = compressed_score_flops + rope_score_flops + value_flops
        
        return proj_flops, attn_flops
    
    def _calculate_standard_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate FLOPs for standard transformer attention"""
        return self._standard_attention_flops(chunk_size, seq_len, kv_dim=self.effective_kv_dim)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """Calculate FFN FLOPs with MoE support"""
        if self.is_moe and (layer_idx is None or not self._is_dense_layer(layer_idx)):
            return self._calculate_moe_ffn_flops(chunk_size)
        else:
            return self._calculate_dense_ffn_flops(chunk_size)
    
    def _calculate_dense_ffn_flops(self, chunk_size: int) -> int:
        """Calculate dense FFN FLOPs"""
        # Gate and Up projections (SwiGLU)
        up_gate_flops = chunk_size * self.hidden_size * self.intermediate_size * 2
        
        # Down projection
        down_flops = chunk_size * self.intermediate_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def _calculate_moe_ffn_flops(self, chunk_size: int) -> int:
        """Calculate MoE FFN FLOPs"""
        # Shared experts (if exist)
        shared_flops = 0
        if self.n_shared_experts > 0:
            shared_flops = self._calculate_single_expert_flops(chunk_size, self.moe_intermediate_size)
        
        # Routed experts (only active ones)
        routed_flops = (self._calculate_single_expert_flops(chunk_size, self.moe_intermediate_size) * 
                       self.num_experts_per_tok)
        
        # Gating network
        gate_flops = chunk_size * self.hidden_size * self.n_routed_experts
        
        return shared_flops + routed_flops + gate_flops
    
    def _calculate_single_expert_flops(self, chunk_size: int, expert_size: int) -> int:
        """Calculate FLOPs for a single expert"""
        # Gate and Up projections
        up_gate_flops = chunk_size * self.hidden_size * expert_size * 2
        
        # Down projection  
        down_flops = chunk_size * expert_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def calculate_other_flops(self, chunk_size: int) -> int:
        """Calculate other FLOPs with MLA-specific normalization"""
        # MLA has additional normalization layers
        norm_multiplier = 6 if self.is_mla else 2
        norm_flops = chunk_size * self.hidden_size * norm_multiplier
        
        # Residual connections
        residual_flops = chunk_size * self.hidden_size * 2
        
        return norm_flops + residual_flops
    
    def estimate_flops_ratio_average(self, chunk_size: int, kv_cache_len: int) -> float:
        """Average FLOPs ratio across layers (MoE uses per-layer averaging)."""
        if not self.is_moe:
            return super().estimate_flops_ratio_average(chunk_size, kv_cache_len)
        total_ratio = 0.0
        for layer_idx in range(self.num_hidden_layers):
            layer_ratio = self.estimate_flops_ratio(chunk_size, kv_cache_len, layer_idx)
            total_ratio += layer_ratio
        return total_ratio / self.num_hidden_layers
    
    def get_layer_analysis(self, chunk_size: int = 2048, kv_cache_len: int = 16384) -> Dict:
        """Analyze p values for each layer type (useful for MoE models)"""
        analysis = {
            'dense_layers': [],
            'moe_layers': [],
            'layer_p_values': {}
        }
        
        for layer_idx in range(self.num_hidden_layers):
            is_dense = self._is_dense_layer(layer_idx)
            p_value = self.estimate_p(chunk_size, kv_cache_len, layer_idx)
            
            analysis['layer_p_values'][layer_idx] = {
                'p_value': p_value,
                'is_dense': is_dense,
                'layer_type': 'dense' if is_dense else 'moe'
            }
            
            if is_dense:
                analysis['dense_layers'].append(layer_idx)
            else:
                analysis['moe_layers'].append(layer_idx)
        
        return analysis
    
    def _extra_model_info(self) -> List[str]:
        architecture_type = f"{'MLA' if self.is_mla else 'Standard'} + {'MoE' if self.is_moe else 'Dense'}"
        lines: List[str] = [f"Architecture: {architecture_type}"]
        if self.is_mla:
            lines += [
                f"Q LoRA Rank: {self.q_lora_rank}",
                f"KV LoRA Rank: {self.kv_lora_rank}",
                f"Effective KV Dim: {self.effective_kv_dim}",
            ]
        if self.is_moe:
            lines += [
                f"MoE Intermediate Size: {self.moe_intermediate_size}",
                f"Routed Experts: {self.n_routed_experts}",
                f"Experts per Token: {self.num_experts_per_tok}",
                f"First K Dense Layers: {self.first_k_dense_replace}",
            ]
            if self.n_shared_experts:
                lines.append(f"Shared Experts: {self.n_shared_experts}")
        else:
            lines.append(
                f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)")
        return lines
    
    def _extra_p_table_info(self) -> None:
        if self.is_moe:
            logger.info("\nLayer Analysis:")
            logger.info("%s", "-" * 50)
            analysis = self.get_layer_analysis()
            logger.info("Dense layers (0-%d): %d", self.first_k_dense_replace - 1, len(analysis['dense_layers']))
            logger.info("MoE layers (%d-%d): %d", self.first_k_dense_replace, self.num_hidden_layers - 1, len(analysis['moe_layers']))


class MixtralEstimator(BaseModelEstimator):
    """FLOP estimator for Mixtral models (MoE architecture)"""
    
    def _parse_architecture_specific_config(self):
        """Parse Mixtral-specific configuration"""
        self.intermediate_size = self.config.get('intermediate_size')
        self.num_local_experts = self.config.get('num_local_experts', 8)
        self.num_experts_per_tok = self.config.get('num_experts_per_tok', 2)
        
        # Mixtral specific parameters
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        self.sliding_window = self.config.get('sliding_window')
        
        # Architecture flags
        self.is_moe = True
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Attention FLOPs for Mixtral with optional sliding window."""
        effective_seq_len = min(seq_len, self.sliding_window) if self.sliding_window else seq_len
        return self._standard_attention_flops(
            chunk_size, seq_len, effective_seq_len=effective_seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """Calculate MoE FFN FLOPs for Mixtral"""
        # Only active experts are computed
        active_expert_flops = self._calculate_single_expert_flops(chunk_size) * self.num_experts_per_tok
        
        # Gating network
        gate_flops = chunk_size * self.hidden_size * self.num_local_experts
        
        return active_expert_flops + gate_flops
    
    def _calculate_single_expert_flops(self, chunk_size: int) -> int:
        """Calculate FLOPs for a single Mixtral expert"""
        # Gate, Up and Down projections (SwiGLU)
        up_gate_flops = chunk_size * self.hidden_size * self.intermediate_size * 2
        down_flops = chunk_size * self.intermediate_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def _extra_model_info(self) -> List[str]:
        lines = [
            "Architecture: Mixtral (MoE)",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            f"Local Experts: {self.num_local_experts}",
            f"Experts per Token: {self.num_experts_per_tok}",
        ]
        if self.sliding_window:
            lines.append(f"Sliding Window: {self.sliding_window}")
        return lines


class GemmaEstimator(BaseModelEstimator):
    """FLOP estimator for Gemma models"""
    
    def _parse_architecture_specific_config(self):
        """Parse Gemma-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
        
        # Gemma specific parameters
        self.head_dim = self.config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 8192)
        
        # Gemma uses GeGLU activation instead of SwiGLU
        self.activation_type = 'gelu'
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate attention FLOPs for Gemma"""
        return self._standard_attention_flops(chunk_size, seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Gemma (GeGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        return [
            "Architecture: Gemma Transformer",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            "Activation: GeGLU",
            f"Max Position Embeddings: {self.max_position_embeddings}",
        ]


# Registry of model estimators
MODEL_ESTIMATOR_REGISTRY = {
    # Llama family
    'llama': LlamaEstimator,
    'llama2': LlamaEstimator,
    'llama3': LlamaEstimator,
    'llama4': LlamaEstimator,
    'code_llama': LlamaEstimator,
    
    # Qwen family  
    'qwen': QwenEstimator,
    'qwen2': QwenEstimator,
    'qwen2_moe': QwenEstimator,
    'qwen2.5': QwenEstimator,
    
    # DeepSeek family
    'deepseek': DeepSeekEstimator,
    'deepseek_v2': DeepSeekEstimator,
    'deepseek_v3': DeepSeekEstimator,
    'deepseek_mtp': DeepSeekEstimator,  # Unified model type for DeepSeek
    
    # Mixtral family
    'mixtral': MixtralEstimator,
    'mixtral_8x7b': MixtralEstimator,
    'mixtral_8x22b': MixtralEstimator,
    
    # Gemma family
    'gemma': GemmaEstimator,
    'gemma2': GemmaEstimator,
    
    # Add other models as needed
    # 'phi': PhiEstimator,
    # 'chatglm': ChatGLMEstimator,
    # etc.
}


def create_estimator(config_path: str = None, config_dict: Dict = None) -> BaseModelEstimator:
    """
    Factory function to create appropriate model estimator based on model type
    
    Args:
        config_path: Path to config file
        config_dict: Config dictionary
        
    Returns:
        Appropriate model estimator instance
    """
    # Load config
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Must provide either config_path or config_dict")
    
    # Determine model type
    model_type = config.get('model_type', 'unknown').lower()
    
    # Handle special cases and mappings
    if model_type == 'deepseek_v3':
        model_type = 'deepseek_mtp'  # Use unified DeepSeek estimator
    
    # Find appropriate estimator class
    estimator_class = MODEL_ESTIMATOR_REGISTRY.get(model_type)
    
    if estimator_class is None:
        # Fallback to Llama estimator for unknown models (most are transformer-based)
        logger.warning("Unknown model type '%s', falling back to Llama estimator",
                       model_type)
        estimator_class = LlamaEstimator
    
    return estimator_class(config)


# Backward compatibility: keep the original ModelPEstimator as an alias
class ModelPEstimator:
    """Backward compatibility wrapper - use create_estimator() for new code"""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        # Create appropriate estimator and delegate to it
        self._estimator = create_estimator(config_path, config_dict)
        
        # Copy attributes for backward compatibility
        self.__dict__.update(self._estimator.__dict__)
    
    def __getattr__(self, name):
        """Delegate any missing methods to the underlying estimator"""
        return getattr(self._estimator, name)


def create_estimator_safely(
    hf_config_obj: object | None = None,
    config_path: str | None = None,
    scheduler_config: object | None = None,
):
    """Best-effort estimator factory.

    - Prefer building from an in-memory HF config object (with to_dict()).
    - Fallback to reading a JSON config from disk.
    - If provided, configure baseline from scheduler config.

    Returns an estimator instance or None on failure.
    """
    estimator = None
    # Try HF config object first
    if hf_config_obj is not None:
        try:
            to_dict_fn = getattr(hf_config_obj, "to_dict", None)
            cfg_dict = to_dict_fn() if callable(to_dict_fn) else dict(
                getattr(hf_config_obj, "__dict__", {}))
            estimator = create_estimator(config_dict=cfg_dict)
        except (AttributeError, TypeError, ValueError):
            estimator = None

    # Fallback to disk json
    if estimator is None and config_path:
        try:
            estimator = create_estimator(config_path=config_path)
        except (OSError, json.JSONDecodeError, ValueError):
            estimator = None

    # Configure baseline if possible
    if estimator is not None and scheduler_config is not None:
        try:
            estimator.configure_baseline_from_scheduler(scheduler_config)
        except AttributeError:
            # Non-fatal; leave estimator as-is
            pass

    return estimator


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Estimate p-values from model config')
    parser.add_argument('--config-path', type=str, help='Path to model config.json')
    parser.add_argument('--chunk-size', type=int, default=2048, 
                       help='Chunk size for estimation')
    parser.add_argument('--kv-cache-len', type=int, default=0,
                       help='KV cache length')
    parser.add_argument('--table', action='store_true',
                       help='Print p-value table')
    parser.add_argument('--estimator-type', type=str, 
                       help='Force specific estimator type (llama, qwen, deepseek)')
    
    args = parser.parse_args()
    
    # Create estimator
    if args.estimator_type:
        # Override model type for testing
        config = json.load(open(args.config_path, 'r', encoding='utf-8'))
        config['model_type'] = args.estimator_type
        estimator = create_estimator(config_dict=config)
    else:
        # Auto-detect from config
        estimator = create_estimator(config_path=args.config_path)
    
    # Print info and calculate ratios
    estimator.print_model_info()
    
    ratio = estimator.estimate_flops_ratio(args.chunk_size, args.kv_cache_len)
    ratio_scaled = estimator.estimate_flops_ratio_scaled(args.chunk_size, args.kv_cache_len)
    
    print("\nEstimated values:")
    print(f"flops_ratio = {ratio:.4f}")
    print(f"flops_ratio × chunk_size = {ratio_scaled:.4f}")
    
    # Print ratio table if requested
    if args.table:
        estimator.print_flops_ratio_table()
    
    estimator.configure_baseline_from_scheduler(None)
    seq_len = 128000
    chunk_size = 2048
    hist_seq_len = 0
    block_size = 16
    while(hist_seq_len < seq_len):
        chunk_size_with_overhead = estimator.compute_chunk_size_with_overhead(
            hist_seq_len, seq_len, chunk_size, block_size)
        print(f"chunk_size_with_overhead = {chunk_size_with_overhead}")
        hist_seq_len += chunk_size_with_overhead[0]

if __name__ == "__main__":
    main()