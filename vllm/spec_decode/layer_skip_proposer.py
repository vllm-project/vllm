"""Layer skip proposer using single-model toggle pattern."""

import weakref
from typing import List, Set, Tuple
import torch

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
        
        model_outputs: List[SamplerOutput] = []
        
        # Enable hidden states if needed for next step
        if execute_model_req.previous_hidden_states is not None:
            self.worker.model_runner.return_hidden_states = True
        
        for step in range(sample_len):
            logger.info(f"[LayerSkipProposer] Draft step {step}/{sample_len}")
            
            # Debug: Check sequence state before execution
            logger.info(f"[DEBUG] Step {step}: checking {len(execute_model_req.seq_group_metadata_list)} groups")
            for i, seq_group in enumerate(execute_model_req.seq_group_metadata_list):
                logger.info(f"  Group {i}: {len(seq_group.seq_data)} sequences")
                for seq_id, seq_data in seq_group.seq_data.items():
                    logger.info(f"  Seq {seq_id}: len={seq_data.get_len()}, "
                              f"computed_tokens={seq_data.get_num_computed_tokens()}, "
                              f"is_prompt={seq_group.is_prompt}")
            
            # CRITICAL FIX for H1: Force single-step execution to prevent logit caching
            # The GPU multi-step path caches a [batch, k, vocab] tensor and slices it
            # We need to ensure each step actually runs the model
            execute_model_req.spec_step_idx = step
            
            # Force num_steps=1 to prevent GPU multi-step optimization
            execute_model_req.num_steps = 1
            
            # BREAKPOINT 1: Draft token generation - watch model.draft_mode, layer_skip value
            step_outputs = self.worker.execute_model(execute_model_req)
            assert len(step_outputs) == 1, "Expected single output per step"
            step_output = step_outputs[0]
            
            # ELEGANT FIX: Force GPU synchronization to break computation graph caching
            # This prevents stale cached values from polluting subsequent draft steps
            # while maintaining most of the performance benefits of early exit
            torch.cuda.synchronize()
            
            # Debug: Check what tokens were generated
            for i, output in enumerate(step_output.outputs):
                for sample in output.samples:
                    logger.info(f"  Generated token: {sample.output_token}")
            
            # Update hidden states if available
            if hasattr(step_output, 'hidden_states') and step_output.hidden_states is not None:
                from vllm.sequence import HiddenStates
                execute_model_req.previous_hidden_states = HiddenStates(
                    step_output.hidden_states,
                    execute_model_req.seq_group_metadata_list
                )
            
            # Append tokens to sequences for next step
            # BREAKPOINT 1b: Sequence update after draft generation - watch seq_data changes
            self._append_new_tokens(step_output, execute_model_req.seq_group_metadata_list)
            model_outputs.append(step_output)
        
        return model_outputs, True  # Indicate transposition needed
    
    def _append_new_tokens(self, sampler_output: SamplerOutput, seq_group_metadata_list):
        """Append sampled tokens to sequences for next step.
        
        When include_gpu_probs_tensor=True (for spec decode), the sampler
        defers CPU pythonization and only provides sampled_token_ids tensor.
        """
        # Since we're in spec decode mode with include_gpu_probs_tensor=True,
        # we need to use the GPU tensor directly
        if hasattr(sampler_output, 'sampled_token_ids') and sampler_output.sampled_token_ids is not None:
            # sampled_token_ids shape: [num_query_tokens, 1]
            # When processing multiple tokens (e.g., step 1 processes ["paris", "famed"]),
            # we only want the LAST token's output (the newly generated one)
            last_token_output = sampler_output.sampled_token_ids[-1]  # Get last row
            
            # For each sequence group (usually just 1 in our case)
            for seq_group_metadata in seq_group_metadata_list:
                seq_group_metadata.is_prompt = False
                
                # Get the first (and usually only) sequence in the group
                seq_id = list(seq_group_metadata.seq_data.keys())[0]
                seq_data = seq_group_metadata.seq_data[seq_id]
                
                # Convert tensor to int
                token_id_int = last_token_output.item()
                
                logger.info(f"[APPEND] seq {seq_id}: appending token {token_id_int} "
                           f"to sequence with len={seq_data.get_len()}")
                
                # Append with default logprob since we're using GPU tensors
                seq_data.append_token_id(token_id_int, 0.0)
        else:
            # This path shouldn't be reached in spec decode mode
            logger.warning("Unexpected: sampler output without GPU tensors in spec decode mode")
    
    # DelegateWorkerBase provides delegation to self.worker automatically
    # No manual delegation needed