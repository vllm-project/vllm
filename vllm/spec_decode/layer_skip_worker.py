"""Layer skip proposer using early exit from target model."""
import weakref
from typing import List, Set, Tuple, Optional
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.sequence import ExecuteModelRequest
from vllm.logger import init_logger
from vllm.worker.worker_base import DelegateWorkerBase
from vllm.model_executor.model_loader import get_model
from vllm.forward_context import set_forward_context

logger = init_logger(__name__)

class LayerSkipProposer(ProposerWorkerBase, DelegateWorkerBase):
    """Use early exit from target model as draft model.
    
    This proposer runs the first N layers of the target model to generate
    draft tokens, using LSQ (Least Squares Quantization) heads for projection
    to vocabulary space.
    """
    
    def __init__(self, *, vllm_config, **kwargs):
        # Initialize DelegateWorkerBase to get access to underlying worker
        DelegateWorkerBase.__init__(self, vllm_config=vllm_config, **kwargs)
        self.model = None
        # Configuration will be loaded in init_device() when model is available
        self.layer_skip = None
        self.entropy_threshold = 2.0
        self.num_speculative_tokens = 4
        self._proposer: Top1Proposer


    def init_device(self):
        """Initialize the worker on GPU device."""
        # Initialize the underlying worker (this loads the actual model onto GPU)
        self.worker.init_device()
        
        # Initialize Top1Proposer to handle proposal conversion
        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )
        
        # Set defaults - config will be loaded lazily when first needed
        self.layer_skip = 4
        self.entropy_threshold = 2.0  
        self.num_speculative_tokens = 4
        self._config_loaded = False
    
    def _ensure_config_loaded(self):
        """Lazy load configuration from model when first needed.
        
        We do this lazily because:
        1. Model might not be fully ready in init_device()
        2. Config loading can be expensive 
        3. We only need config when actually doing inference
        """
        if self._config_loaded:
            return
        
        # Now the model should be fully loaded and ready
        self.model = self.get_model()
        
        # Try to load speculative config
        spec_config = getattr(self.vllm_config.model_config, "speculative_config", None)
        if spec_config is not None:
            self.layer_skip = getattr(spec_config, 'layer_skip', 4)
            self.entropy_threshold = getattr(spec_config, 'draft_entropy_threshold', 2.0)
            self.num_speculative_tokens = getattr(spec_config, 'num_speculative_tokens', 4)
            
            # Load LSQ head into the model if available
            if (hasattr(self.model, 'load_lsq_head') and 
                hasattr(spec_config, 'lsq_head_path') and 
                spec_config.lsq_head_path):
                
                self.model.load_lsq_head(spec_config.lsq_head_path, self.layer_skip)
                logger.info(f"[LayerSkipProposer] Loaded LSQ head from {spec_config.lsq_head_path}")
                
            logger.info(
                "[LayerSkipProposer] Loaded config: layer_skip=%d, "
                "entropy_threshold=%.2f, num_speculative_tokens=%d",
                self.layer_skip, self.entropy_threshold, self.num_speculative_tokens
            )
        else:
            logger.info("[LayerSkipProposer] No speculative config found, using defaults")
        
        self._config_loaded = True
    
    def _layer_skip_logits(self, execute_model_req) -> torch.Tensor:
        """Perform layer skip forward pass to get logits."""
        self._ensure_config_loaded()
        
        # Let vLLM build a proper BroadcastableModelInput ✅
        model_input, _, _ = self.worker.prepare_input(execute_model_req)
        
        # Use the model's early exit method
        num_tokens = model_input.input_tokens.numel()
        with set_forward_context(model_input.attn_metadata, self.vllm_config, num_tokens=num_tokens):
            hidden_states = self.model.forward_with_early_exit(
                model_input.input_tokens,
                model_input.input_positions,
                self.layer_skip,
            )
        
        # Get LSQ head
        head = self._get_lsq_head(hidden_states)
        
        # Handle decode step: hidden_states is (B, D) for single token processing
        if hidden_states.ndim == 2:
            # Decode step: (B, D) -> project -> (B, vocab_size)
            B, D = hidden_states.shape
            logits = hidden_states @ head.T  # (B, D) @ (vocab_size, D).T = (B, vocab_size)
            return logits.unsqueeze(1)  # (B, 1, vocab_size) for consistency with expected output
        else:
            # Prefill step: (B, S, D) -> project -> (B, S, vocab_size)
            B, S, D = hidden_states.shape
            return (hidden_states.view(-1, D) @ head.T).view(B, S, -1)
    
    def _get_lsq_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get LSQ head tensor, caching GPU copy."""
        if self.layer_skip not in self.model.lsq_heads:
            raise ValueError(f"No LSQ head for layer {self.layer_skip}")
        
        # Get LSQ head (cache GPU copy)
        if self.layer_skip not in self.model._lsq_heads_gpu:
            self.model._lsq_heads_gpu[self.layer_skip] = self.model.lsq_heads[self.layer_skip].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        return self.model._lsq_heads_gpu[self.layer_skip]

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],  # Unused for simple layer skip
    ) -> Tuple[List[SamplerOutput], bool]:
        """Generate draft tokens using early exit from target model."""
        self._raise_if_unsupported(execute_model_req)
        
        # Get logits from layer skip forward pass
        logits = self._layer_skip_logits(execute_model_req)
        
        # Convert logits to SamplerOutput objects
        # Note: We need to generate sample_len future tokens, not process sequence length
        sampler_outputs = []
        
        # Debug: Log shapes to understand what we're produced
        logger.info(f"[LAYER_SKIP_DEBUG] logits.shape: {logits.shape}, sample_len: {sample_len}")
        
        # Use the last position logits (most recent token) for next token prediction
        current_logits = logits[:, -1, :]  # [B, vocab_size]
        B = current_logits.shape[0]
        
        # Create one SamplerOutput per sequence (not per time step!)
        for seq_idx in range(B):
            # Get logits for this sequence
            seq_logits = current_logits[seq_idx:seq_idx+1]  # [1, vocab_size]
            
            # Collect sample_len tokens for this sequence
            seq_token_ids = []
            seq_full_probs = []  # Full vocab distributions [sample_len, vocab_size]
            seq_full_logprobs = []  # Full vocab log distributions [sample_len, vocab_size]
            
            for step in range(sample_len):
                probs = F.softmax(seq_logits, dim=-1)  # [1, vocab_size]
                step_logprobs = F.log_softmax(seq_logits, dim=-1)  # [1, vocab_size]
                
                # Greedy sampling for this step
                sampled_token_id = torch.argmax(seq_logits, dim=-1)  # [1]
                
                seq_token_ids.append(sampled_token_id[0].item())
                seq_full_probs.append(probs[0])  # [vocab_size]
                seq_full_logprobs.append(step_logprobs[0])  # [vocab_size]
                
                # For subsequent steps, reuse same logits (TODO: proper multi-step)
            
            # Convert to tensors with correct shape for this sequence
            seq_sampled_token_ids = torch.tensor(seq_token_ids, device=current_logits.device)  # [sample_len]
            seq_sampled_token_probs = torch.stack(seq_full_probs, dim=0)  # [sample_len, vocab_size]
            seq_sampled_logprobs = torch.stack(seq_full_logprobs, dim=0)  # [sample_len, vocab_size]
            
            logger.info(f"[LAYER_SKIP_DEBUG] Sequence {seq_idx}: token_ids.shape={seq_sampled_token_ids.shape}, "
                       f"token_probs.shape={seq_sampled_token_probs.shape}, logprobs.shape={seq_sampled_logprobs.shape}")
            
            sampler_outputs.append(SamplerOutput(
                outputs=[],  # Empty list - will be filled by Top1Proposer
                sampled_token_ids=seq_sampled_token_ids,
                sampled_token_probs=seq_sampled_token_probs,  # [sample_len, vocab_size] - full distributions
                logprobs=seq_sampled_logprobs,  # [sample_len, vocab_size] - full distributions
            ))
        
        logger.info(f"[LAYER_SKIP_DEBUG] Returning {len(sampler_outputs)} SamplerOutput objects (one per sequence)")
        return sampler_outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Generate speculative proposals by delegating to Top1Proposer.
        
        Top1Proposer will call our sampler_output() method and convert
        the SamplerOutput objects to SpeculativeProposals format.
        """
        logger.info(f"[LAYER_SKIP_DEBUG] Calling Top1Proposer.get_spec_proposals()")
        proposals = self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step
        )
        
        logger.info(f"[LAYER_SKIP_DEBUG] Got proposals: token_ids.shape={proposals.proposal_token_ids.shape}, "
                   f"probs.shape={proposals.proposal_probs.shape}, lens.shape={proposals.proposal_lens.shape}")
        logger.info(f"[LAYER_SKIP_DEBUG] proposal_lens values: {proposals.proposal_lens}")
        
        return proposals

    def _raise_if_unsupported(self, execute_model_req: ExecuteModelRequest) -> None:
        """Layer skip doesn't support certain operations yet."""
        if any([
            execute_model_req.blocks_to_swap_in,
            execute_model_req.blocks_to_swap_out,
            execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "LayerSkipProposer does not support cache operations yet")

        if any(
            len(seq_group_metadata.seq_data.keys()) != 1
            for seq_group_metadata in execute_model_req.seq_group_metadata_list
        ):
            raise NotImplementedError(
                "LayerSkipProposer does not support beam search")