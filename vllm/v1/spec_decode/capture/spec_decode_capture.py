# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Main capture orchestrator for speculative decoding distillation."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.capture.config import (
    CaptureConfig,
    ConfigurationManager,
)
from vllm.v1.spec_decode.capture.safetensors_writer import (
    AsyncSafetensorsWriter,
)
from vllm.v1.spec_decode.capture.percentile_tracker import (
    PercentileTracker,
)
from vllm.v1.spec_decode.capture.rate_limiter import RateLimiter
from vllm.v1.spec_decode.capture.transfer_handler import (
    AsyncTransferHandler,
)

logger = init_logger(__name__)


class SpecDecodeCapture:
    """Orchestrates capture workflow for speculative decoding distillation.
    
    Uses percentile-based tracking to capture only the worst X% of acceptance
    rates, focusing on the most problematic cases for knowledge distillation.
    """
    
    def __init__(self, speculative_config: SpeculativeConfig):
        """Initialize capture from speculative config.
        
        Args:
            speculative_config: Speculative decoding configuration.
        """
        # Auto-detect tensor parallel config
        try:
            from vllm.distributed import get_tp_group
            tp_group = get_tp_group()
            self.tp_size = tp_group.world_size
            self.tp_rank = tp_group.rank_in_group
        except Exception:
            # Not in distributed mode
            self.tp_size = 1
            self.tp_rank = 0
        
        # Create configuration
        config = CaptureConfig(
            enabled=speculative_config.capture_enabled,
            top_k=speculative_config.capture_top_k,
            output_dir=speculative_config.capture_dir
            or "./capture_data",
            max_capture_percentage=100.0,
            write_queue_size=1000,
        )
        self.config = ConfigurationManager(config)

        # Create components
        self.rate_limiter = RateLimiter(max_percentage=100.0)
        self.percentile_tracker = PercentileTracker(
            percentile=speculative_config.capture_percentile,
            window_size=speculative_config.capture_window_size,
            min_samples=100,
        )
        writer = AsyncSafetensorsWriter(
            output_dir=speculative_config.capture_dir
            or "./capture_data",
            queue_size=1000,
            batch_size=10,
            batch_timeout=5.0,
            use_compression=True,
        )
        self.transfer_handler = AsyncTransferHandler(
            writer,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

        # Track whether we're using percentile mode (always true)
        self.use_percentile = True
        
        # Stats tracking for periodic logging
        self._step_count = 0
        self._stats_interval = 100  # Log stats every 100 steps
    
    def log_spec_decode_step(
        self,
        logits: torch.Tensor,
        spec_decode_metadata,
        sampled_token_ids: torch.Tensor,
        model_name: Optional[str] = None,
        sample_hidden_states: Optional[torch.Tensor] = None,
        aux_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Log a speculative decoding step for distillation capture.
        
        This is the main entry point called from gpu_model_runner. It handles
        extracting target logits/hidden states and periodic stats logging.
        
        Args:
            logits: Full logits tensor from target model.
            spec_decode_metadata: SpecDecodeMetadata with target_logits_indices.
            sampled_token_ids: Sampled token IDs from rejection sampler.
            model_name: Name of the target model.
            sample_hidden_states: Hidden states tensor (optional).
            aux_hidden_states: List of auxiliary hidden states for EAGLE3 (optional).
        """
        try:
            # Extract target logits
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            
            # Extract corresponding hidden states if available
            target_hidden_states = None
            if aux_hidden_states is not None and len(aux_hidden_states) > 0:
                # Concatenate aux hidden states from multiple layers (EAGLE3)
                concat_hidden_states = torch.cat(aux_hidden_states, dim=-1)
                target_hidden_states = concat_hidden_states[spec_decode_metadata.target_logits_indices]
            elif sample_hidden_states is not None:
                target_hidden_states = sample_hidden_states[spec_decode_metadata.target_logits_indices]
            
            # Call the core capture method
            self.maybe_capture(
                target_logits,
                spec_decode_metadata,
                sampled_token_ids,
                model_name,
                target_hidden_states,
            )
            
            # Periodic stats logging
            self._step_count += 1
            if self._step_count % self._stats_interval == 0:
                self._log_periodic_stats()
                
        except Exception as e:
            logger.error(f"Error in log_spec_decode_step: {e}")
    
    def _log_periodic_stats(self) -> None:
        """Log periodic statistics for observability."""
        stats = self.get_stats()
        threshold = stats.get('percentile_tracker', {}).get('percentile_threshold')
        threshold_str = f'{threshold:.3f}' if threshold is not None else 'N/A'
        writer_stats = stats.get('writer', {})
        
        logger.info(
            f"Distillation capture stats (step {self._step_count}): "
            f"threshold={threshold_str}, "
            f"writes={writer_stats.get('total_writes', 0)}, "
            f"drops={writer_stats.get('total_drops', 0)}"
        )
    
    def maybe_capture(
        self,
        logits: torch.Tensor,
        spec_decode_metadata,
        output_token_ids: torch.Tensor,
        model_name: Optional[str] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Check if capture should occur and initiate async capture.
        
        This method must return quickly to avoid blocking inference.
        
        Args:
            logits: Logits tensor from target model [batch_size, seq_len, vocab_size].
            spec_decode_metadata: SpecDecodeMetadata with draft token info.
            output_token_ids: Output token IDs after rejection sampling.
            model_name: Name of the target model (optional).
            hidden_states: Hidden states from target model [batch_size, seq_len, hidden_size] (optional).
        """
        try:
            # Early exit if disabled
            if not self.config.is_enabled():
                return
            
            # Calculate acceptance stats from metadata
            # NOTE: For all spec decode methods (EAGLE, Medusa, ngram):
            # - target_logits_indices contains indices of draft tokens to verify
            # - output_token_ids contains accepted tokens + 1 bonus token
            num_draft_tokens = len(spec_decode_metadata.target_logits_indices)
            num_accepted_tokens = len(output_token_ids) - 1  # Exclude bonus token
            acceptance_length = self._calculate_acceptance_length({
                'num_accepted_tokens': num_accepted_tokens,
                'num_draft_tokens': num_draft_tokens,
            })
            
            logger.debug(
                f"Checking if should capture. Acceptance length: {acceptance_length:.2f}, "
                f"num_accepted: {num_accepted_tokens}, num_drafts: {num_draft_tokens}"
            )
            
            # Check if this draft should be captured
            should_capture = self.percentile_tracker.observe_and_check_capture(acceptance_length)
            if should_capture:
                logger.debug(
                    f"Acceptance length {acceptance_length:.2f} is in worst "
                    f"{self.percentile_tracker.percentile}% percentile"
                )
            
            if not should_capture:
                return
            
            # Check rate limit
            if not self.rate_limiter.should_capture():
                logger.debug("Rate limit exceeded, skipping capture")
                return
            
            # Record that we're capturing this draft
            self.rate_limiter.record_captured()
            
            # Handle logits dimensions
            # Handle both 2D [batch_size, vocab_size] and 3D [batch_size, seq_len, vocab_size]
            if logits.dim() == 2:
                batch_size, vocab_size = logits.shape
                seq_len = 1
                # Reshape to 3D for consistency
                logits = logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
            elif logits.dim() == 3:
                batch_size, seq_len, vocab_size = logits.shape
            else:
                logger.warning(f"Unexpected logits shape: {logits.shape}, skipping capture")
                return
            
            # Extract top-k (fast operation on GPU) - after dimension normalization
            top_k_probs, top_k_indices = self._extract_top_k(
                logits, self.config.get_top_k()
            )
            
            # Extract target token IDs from output_token_ids
            # output_token_ids contains accepted tokens + 1 bonus token
            # We want the target tokens that correspond to the draft positions
            # For EAGLE3 training, we need the actual token IDs that were sampled
            if output_token_ids.dim() == 1:
                # Reshape to [batch_size, seq_len] format
                # Each position in target_logits_indices corresponds to a draft token
                # The output_token_ids are the accepted tokens
                num_tokens = min(len(output_token_ids), batch_size * seq_len)
                input_ids = output_token_ids[:num_tokens].reshape(-1, seq_len)
                # Pad if needed
                if input_ids.shape[0] < batch_size:
                    padding = torch.zeros(
                        (batch_size - input_ids.shape[0], seq_len),
                        dtype=torch.long,
                        device=logits.device
                    )
                    input_ids = torch.cat([input_ids, padding], dim=0)
            else:
                input_ids = output_token_ids[:batch_size, :seq_len]
            
            # Ensure correct dtype
            input_ids = input_ids.to(dtype=torch.long, device=logits.device)
            logger.debug(f"Extracted input_ids with shape {input_ids.shape}, values: {input_ids.flatten()[:5].tolist()}")
            
            logger.debug(
                f"Initiating async transfer for capture. "
                f"Shape: {logits.shape}, top_k: {self.config.get_top_k()}"
            )
            
            # Initiate async transfer (non-blocking)
            self.transfer_handler.transfer_and_write(
                top_k_probs,
                top_k_indices,
                input_ids,
                acceptance_length,
                num_accepted_tokens,
                num_draft_tokens,
                model_name,
                None,  # prompt
                hidden_states,  # Pass hidden states for EAGLE3 training
            )
            
        except Exception as e:
            # Log error but don't propagate to avoid blocking inference
            logger.error(f"Error in distillation capture: {e}")
    
    # Backward compatibility alias
    def maybe_log(
        self,
        logits: torch.Tensor,
        spec_decode_metadata,
        output_token_ids: torch.Tensor,
        model_name: Optional[str] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Check if capture should occur (backward compat alias)."""
        self.maybe_capture(logits, spec_decode_metadata, output_token_ids, model_name, hidden_states)
    
    def _calculate_acceptance_length(self, acceptance_stats: Dict) -> float:
        """Calculate acceptance length from stats.
        
        Args:
            acceptance_stats: Dictionary with num_accepted_tokens and num_draft_tokens.
        
        Returns:
            Acceptance length (1.0 + num_accepted / num_drafts).
        """
        num_accepted = acceptance_stats.get('num_accepted_tokens', 0)
        num_drafts = acceptance_stats.get('num_draft_tokens', 0)
        
        if num_drafts > 0:
            return 1.0 + (num_accepted / num_drafts)
        else:
            return 1.0
    
    def _extract_top_k(
        self,
        logits: torch.Tensor,
        k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract top-k probabilities and indices (GPU operation).
        
        Optimized to do topk on logits first, then softmax only on top-k.
        This is O(k) instead of O(vocab_size) for the softmax.
        
        Args:
            logits: Logits tensor [batch_size, seq_len, vocab_size].
            k: Number of top probabilities to extract.
        
        Returns:
            Tuple of (top_k_probs, top_k_indices), each with shape
            [batch_size, seq_len, k].
        """
        # Extract top-k logits first (O(vocab_size * log k))
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Only softmax over top-k values (O(k) instead of O(vocab_size))
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        return top_k_probs, top_k_indices
    
    def get_stats(self) -> dict:
        """Get current logging statistics for observability.
        
        Returns:
            Dictionary with percentile tracker stats, rate limiter stats,
            and writer stats.
        """
        stats = {
            'enabled': self.config.is_enabled(),
            'use_percentile': self.use_percentile,
        }
        
        # Add percentile tracker stats
        if self.percentile_tracker:
            stats['percentile_tracker'] = self.percentile_tracker.get_stats()
        
        # Add rate limiter stats
        rate_limiter_stats = self.rate_limiter.get_stats()
        stats['rate_limiter'] = {
            'max_logging_percentage': self.rate_limiter.max_percentage,
            **rate_limiter_stats,
        }
        
        # Add writer stats
        stats['writer'] = self.transfer_handler.writer.get_stats()
        
        return stats
