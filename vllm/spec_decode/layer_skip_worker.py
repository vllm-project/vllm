"""Minimal layer skip proposer."""
from typing import List
import torch

from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.data_structures import SpeculativeProposals
from vllm.sequence import ExecuteModelRequest

class LayerSkipProposer(ProposerWorkerBase):
    """Use early model layers as draft model."""
    
    def init_device(self):
        super().init_device()
        # Access config through model runner
        if hasattr(self.model_runner.model, 'config'):
            spec_config = self.model_runner.model.config.speculative_config
            if spec_config and spec_config.layer_skip:
                self.layer_skip = spec_config.layer_skip
                self.entropy_threshold = spec_config.draft_entropy_threshold
                # FIXED: Define num_speculative_tokens
                self.num_speculative_tokens = spec_config.num_speculative_tokens or 4
                
                # Load LSQ head into model
                if hasattr(self.model_runner.model, 'load_lsq_head'):
                    self.model_runner.model.load_lsq_head(
                        spec_config.lsq_head_path, 
                        self.layer_skip
                    )
            else:
                # Fallback defaults
                self.layer_skip = 4
                self.entropy_threshold = 2.0
                self.num_speculative_tokens = 4
    
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: set,
    ) -> SpeculativeProposals:
        """Generate proposals via early exit."""
        
        # Use base class helper
        model_input = self._build_model_runner_input(execute_model_req)
        
        # Handle None KV caches
        kv_caches = model_input.kv_caches
        if kv_caches is not None:
            kv_caches = kv_caches[:self.layer_skip]
        
        # Run early exit forward
        with torch.no_grad():
            logits, entropy = self.model_runner.model.forward_with_early_exit(
                model_input.input_tokens,
                model_input.input_positions,
                kv_caches,
                model_input.attn_metadata,
                stop_layer=self.layer_skip,
            )
        
        # Sample tokens (greedy for now)
        tokens = torch.argmax(logits, dim=-1)
        
        # Build proposals
        proposal_lens = []
        proposal_tokens = []
        total_draft_tokens = 0
        
        for batch_idx in range(tokens.shape[0]):
            seq_tokens = []
            for pos_idx in range(min(tokens.shape[1], self.num_speculative_tokens)):
                if entropy[batch_idx, pos_idx] < self.entropy_threshold:
                    seq_tokens.append(tokens[batch_idx, pos_idx].item())
                    total_draft_tokens += 1
                else:
                    break  # Stop at high entropy
            
            proposal_tokens.append(seq_tokens)
            proposal_lens.append(len(seq_tokens))
        
        # Track metrics
        self.num_draft_tokens = total_draft_tokens
        
        return SpeculativeProposals(
            proposal_token_ids=proposal_tokens,
            proposal_probs=[[1.0] * len(tokens) for tokens in proposal_tokens],
            proposal_lens=proposal_lens,
        )