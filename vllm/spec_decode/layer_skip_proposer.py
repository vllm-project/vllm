"""Layer skip proposer using single-model toggle pattern."""

import weakref
from typing import List, Set, Tuple

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import DelegateWorkerBase

logger = init_logger(__name__)


class LayerSkipProposer(ProposerWorkerBase, DelegateWorkerBase):
    """Layer skip proposer using single-model toggle pattern.
    
    Uses the target model directly, toggling draft_mode for early exit.
    No model duplication, no memory overhead, no layer name conflicts.
    """
    
    def __init__(self, target_worker, layer_skip: int = 4, **kwargs):
        # Store the target worker and configuration
        self.target_worker = target_worker
        self.layer_skip = layer_skip
        
        # Initialize DelegateWorkerBase (it will create a worker, but we'll override)
        DelegateWorkerBase.__init__(self, **kwargs)
        
        # Override with our target worker
        self.worker = target_worker
        
        logger.info(f"[LayerSkipProposer] Created with layer_skip={layer_skip}")
    
    def init_device(self) -> None:
        """Initialize device - target worker is already initialized."""
        # Target worker is already initialized by SpecDecodeWorker
        # Just create our Top1Proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )
        
        logger.info("[LayerSkipProposer] Top1Proposer initialized")
    
    def load_model(self) -> None:
        """Load model - target worker already loaded, just validate and setup LSQ heads."""
        # Target worker is already loaded by SpecDecodeWorker
        model = self.worker.get_model()
        
        # Validate layer_skip bounds
        num_layers = model.config.num_hidden_layers
        if self.layer_skip >= num_layers:
            raise ValueError(f"layer_skip={self.layer_skip} "
                           f">= num_layers={num_layers}")
        
        # Load LSQ heads if configured  
        print(f"[DEBUG] Starting LSQ head loading check...")
        print(f"[DEBUG] vllm_config: {self.worker.vllm_config}")
        print(f"[DEBUG] model_config: {self.worker.vllm_config.model_config}")
        
        spec_config = getattr(self.worker.vllm_config.model_config, "speculative_config", None)
        print(f"[DEBUG] spec_config from model_config: {spec_config}")
        
        # Also try the engine config location
        engine_config = getattr(self.worker.vllm_config, "speculative_config", None)  
        print(f"[DEBUG] spec_config from engine config: {engine_config}")
        
        # Try both locations
        final_spec_config = spec_config or engine_config
        print(f"[DEBUG] final_spec_config: {final_spec_config}")
        
        if final_spec_config:
            print(f"[DEBUG] spec_config type: {type(final_spec_config)}")
            print(f"[DEBUG] spec_config dir: {dir(final_spec_config)}")
            lsq_path = getattr(final_spec_config, 'lsq_head_path', None)
            print(f"[DEBUG] lsq_head_path: {lsq_path}")
            print(f"[DEBUG] model type: {type(model)}")
            print(f"[DEBUG] model has load_lsq_head: {hasattr(model, 'load_lsq_head')}")
            
            if (hasattr(final_spec_config, 'lsq_head_path') and 
                final_spec_config.lsq_head_path and hasattr(model, 'load_lsq_head')):
                
                print(f"[DEBUG] About to load LSQ head from: {final_spec_config.lsq_head_path}")
                model.load_lsq_head(final_spec_config.lsq_head_path, self.layer_skip)
                print(f"[DEBUG] LSQ head loaded successfully!")
                logger.info(f"[LayerSkipProposer] Loaded LSQ head from {final_spec_config.lsq_head_path}")
            else:
                print(f"[DEBUG] LSQ head loading conditions not met")
                print(f"[DEBUG] - has lsq_head_path attr: {hasattr(final_spec_config, 'lsq_head_path')}")
                print(f"[DEBUG] - lsq_head_path value: {getattr(final_spec_config, 'lsq_head_path', 'NOT_FOUND')}")
                print(f"[DEBUG] - model has load_lsq_head: {hasattr(model, 'load_lsq_head')}")
        else:
            print(f"[DEBUG] No spec config found")
        
        logger.info("[LayerSkipProposer] Model setup complete")
    
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations using draft mode toggle."""
        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)
    
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run model in draft mode for the specified sample length."""
        
        # Get the model and toggle to draft mode
        model = self.worker.get_model()
        
        with model.draft_mode_ctx(self.layer_skip):
            # Delegate to MultiStepWorker logic (we inherit its sampler_output via ProposerWorkerBase)
            # For now, implement simple multi-step logic directly
            return self._multi_step_draft_sampling(
                execute_model_req, sample_len, seq_ids_with_bonus_token_in_last_step)
    
    def _multi_step_draft_sampling(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Simple multi-step autoregressive draft sampling."""
        
        # For now, implement a simple version
        # TODO: Use MultiStepWorker logic if needed
        model_outputs: List[SamplerOutput] = []
        
        for step in range(sample_len):
            # Execute one step in draft mode
            step_outputs = self.worker.execute_model(execute_model_req)
            assert len(step_outputs) == 1, "Expected single output per step"
            step_output = step_outputs[0]
            
            # Append tokens to sequences for next step
            self._append_new_tokens(step_output, execute_model_req.seq_group_metadata_list)
            model_outputs.append(step_output)
        
        return model_outputs, True  # Indicate transposition needed
    
    def _append_new_tokens(self, sampler_output: SamplerOutput, seq_group_metadata_list):
        """Append sampled tokens to sequences for next step."""
        for seq_group_metadata, sequence_output in zip(seq_group_metadata_list, sampler_output.outputs):
            for seq_output in sequence_output.samples:
                # Get the sequence and append the token
                seq_data = seq_group_metadata.seq_data[sequence_output.parent_seq_id]
                seq_data.append_token_id(
                    seq_output.output_token, 
                    seq_output.logprobs.get(seq_output.output_token, 0.0) if seq_output.logprobs else 0.0
                )
    
    # DelegateWorkerBase provides delegation to self.worker automatically
    # No manual delegation needed