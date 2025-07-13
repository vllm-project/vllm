"""Early exit proposer worker using monkey-patching for Pattern C.

This implementation uses the monkey-patch approach to avoid compilation
context mismatches while maintaining clean two-worker separation.
"""
import copy
import logging
import os
import types
import weakref
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import DelegateWorkerBase

if TYPE_CHECKING:
    from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata, SequenceData
    from vllm.config import VllmConfig
    import torch.nn as nn
else:
    from vllm.sequence import SequenceGroupMetadata, SequenceData

logger = init_logger(__name__)


def create_early_exit_forward(exit_layer: int):
    """Create an early exit forward function for the given exit layer."""
    def early_exit_forward(self, input_ids, positions, 
                          intermediate_tensors=None, inputs_embeds=None):
        """Forward pass that exits early at specified layer."""
        # Get embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        
        # Access layers
        if hasattr(self, 'model'):
            layers = self.model.layers
            norm = self.model.norm
        else:
            layers = self.layers
            norm = self.norm
        
        # Process layers up to exit point
        residual = None
        for i in range(min(exit_layer, len(layers))):
            layer = layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        
        # Apply final norm
        if residual is not None:
            hidden_states, _ = norm(hidden_states, residual)
        else:
            hidden_states = norm(hidden_states)
        
        return hidden_states
    
    return early_exit_forward


def create_early_exit_compute_logits(lsq_head):
    """Create a compute_logits function that uses LSQ head."""
    def compute_logits(self, hidden_states, sampling_metadata):
        """Compute logits using LSQ head if available."""
        if lsq_head is not None:
            return self.logits_processor(lsq_head, hidden_states, sampling_metadata)
        else:
            # Fallback to original lm_head
            return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
    
    return compute_logits


class EarlyExitProposerWorker(ProposerWorkerBase, DelegateWorkerBase):
    """Proposer worker using monkey-patched early exit model.
    
    This worker creates its own execution context but shares model weights
    with the target by using a shallow clone with patched methods.
    """
    
    def __init__(self, 
                 target_worker,
                 exit_layer: int,
                 lsq_head_path: Optional[str] = None,
                 **kwargs):
        """Initialize early exit proposer worker.
        
        Args:
            target_worker: The target worker whose model to share
            exit_layer: Layer index to exit at (0-indexed)
            lsq_head_path: Optional path to LSQ projection heads
            **kwargs: Additional arguments for worker initialization
        """
        self.target_worker = target_worker
        self.exit_layer = exit_layer
        self.lsq_head_path = lsq_head_path
        self._target_model = None
        self._proposer_model = None
        self._lsq_head = None
        
        # Create optimized config for proposer
        base_config = kwargs.get('vllm_config', target_worker.vllm_config)
        proposer_config = self._create_proposer_config(base_config)
        kwargs['vllm_config'] = proposer_config
        
        # Initialize DelegateWorkerBase which creates self.worker
        DelegateWorkerBase.__init__(self, **kwargs)
        
        # Copy necessary attributes for ProposerWorkerBase
        self.vllm_config = proposer_config
        self.device = getattr(self.worker, 'device', None)
        self.device_config = self.worker.device_config
        self.model_config = self.worker.model_config
        self.max_model_len = self.worker.model_config.max_model_len
        
        logger.info(f"Created EarlyExitProposerWorker with exit_layer={exit_layer}")
    
    def _create_proposer_config(self, base_config: "VllmConfig") -> "VllmConfig":
        """Create optimized config for proposer with minimal resources."""
        config = copy.deepcopy(base_config)
        
        # Proposer needs minimal KV cache
        if hasattr(config.cache_config, 'num_gpu_blocks') and config.cache_config.num_gpu_blocks:
            original_blocks = config.cache_config.num_gpu_blocks
            config.cache_config.num_gpu_blocks = max(10, min(50, original_blocks // 20))
            logger.info(f"Reduced proposer GPU blocks from {original_blocks} to "
                       f"{config.cache_config.num_gpu_blocks}")
        
        # Single sequence speculation
        config.scheduler_config.max_num_seqs = 1
        
        return config
    
    def init_device(self) -> None:
        """Initialize device and create Top1 proposer."""
        # Let the worker initialize its device
        self.worker.init_device()
        
        # Update device attribute
        self.device = self.worker.device
        
        # Create Top1 proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),
            self.device,
            self.worker.model_config.get_vocab_size(),
            max_proposal_len=self.max_model_len,
        )
        
        # Enable GPU probs tensor for multi-step
        if hasattr(self.worker, 'set_include_gpu_probs_tensor'):
            self.worker.set_include_gpu_probs_tensor()
        elif hasattr(self.worker.model_runner, 'sampler'):
            self.worker.model_runner.sampler.include_gpu_probs_tensor = True
        
        logger.info("Initialized early exit proposer device and Top1 proposer")
    
    def load_model(self) -> None:
        """Load model and apply monkey-patch for early exit behavior."""
        # First, let the worker load its model normally
        # This ensures compilation happens with the correct model structure
        self.worker.load_model()
        
        # Get the target model for weight sharing
        self._target_model = self.target_worker.model_runner.model
        if self._target_model is None:
            raise RuntimeError("Target model must be loaded before proposer")
        
        # Create a shallow clone of the target model
        # This shares all tensors but is a distinct Python object
        self._proposer_model = copy.copy(self._target_model)
        self._proposer_model.__dict__ = self._target_model.__dict__.copy()
        
        # Load LSQ head if configured
        if self.lsq_head_path:
            self._load_lsq_head()
        
        # Monkey-patch the forward and compute_logits methods
        # Since this is a different object than target_model, changes don't affect scorer
        self._proposer_model.forward = types.MethodType(
            create_early_exit_forward(self.exit_layer), 
            self._proposer_model
        )
        self._proposer_model.compute_logits = types.MethodType(
            create_early_exit_compute_logits(self._lsq_head),
            self._proposer_model
        )
        
        # Replace the worker's model with our patched version
        self.worker.model_runner.model = self._proposer_model
        
        logger.info(f"Monkey-patched model for early exit at layer {self.exit_layer}")
    
    def _load_lsq_head(self) -> None:
        """Load LSQ head and attach to the model."""
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.distributed import get_pp_group
        
        # Only load on last PP rank
        if not get_pp_group().is_last_rank:
            from vllm.model_executor.models.utils import PPMissingLayer
            self._lsq_head = PPMissingLayer()
            return
        
        # Load full head weight
        head_file = os.path.join(self.lsq_head_path, f"h{self.exit_layer}.pt")
        if not os.path.exists(head_file):
            raise FileNotFoundError(f"LSQ head file not found: {head_file}")
        
        try:
            full_weight = torch.load(head_file, map_location="cpu")
        except Exception as e:
            raise ValueError(f"Failed to load LSQ head from {head_file}: {e}")
        
        # Validate shape
        expected_shape = (self._target_model.config.vocab_size, 
                         self._target_model.config.hidden_size)
        if full_weight.shape != expected_shape:
            raise ValueError(f"LSQ head shape {full_weight.shape} does not match "
                           f"expected {expected_shape}")
        
        # Get dtype and device from target model
        target_dtype = next(self._target_model.parameters()).dtype
        target_device = next(self._target_model.parameters()).device
        
        # Create TP-sharded head if needed
        if hasattr(self._target_model, 'lm_head') and hasattr(self._target_model.lm_head, 'shard_indices'):
            # Get sharding info from main lm_head
            shard_indices = self._target_model.lm_head.shard_indices
            v_start = shard_indices.org_vocab_start_index
            v_end = shard_indices.org_vocab_end_index
            
            # Create LSQ head with same config as lm_head
            self._lsq_head = ParallelLMHead(
                num_embeddings=self._target_model.config.vocab_size,
                embedding_dim=self._target_model.config.hidden_size,
                quant_config=getattr(self._target_model, 'quant_config', None),
                prefix=f"lsq_head_{self.exit_layer}"
            )
            
            # Load sharded weights
            self._lsq_head.weight.data = full_weight[v_start:v_end].to(target_dtype)
            self._lsq_head = self._lsq_head.to(target_device)
            
            logger.info(f"Loaded TP-sharded LSQ head for layer {self.exit_layer} "
                       f"(vocab shard: {v_start}:{v_end})")
        else:
            # Non-sharded case
            logger.warning("Loading non-sharded LSQ head")
            self._lsq_head = torch.nn.Linear(
                self._target_model.config.hidden_size, 
                self._target_model.config.vocab_size,
                bias=False
            )
            self._lsq_head.weight.data = full_weight.to(target_dtype)
            self._lsq_head = self._lsq_head.to(target_device)
        
        logger.info(f"Successfully loaded LSQ head from {self.lsq_head_path}")
    
    def get_spec_proposals(
        self,
        execute_model_req: "ExecuteModelRequest",
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculative proposals using early exit."""
        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step
        )
    
    def _keep_last_row(self, out: SamplerOutput, num_seqs: int) -> None:
        """Keep the newest token for every live sequence.
        
        Args:
            out: SamplerOutput to modify
            num_seqs: Number of sequences in the batch
        """
        row_slice = slice(-num_seqs, None)  # last <num_seqs> rows
        
        # Tensors that have shape [num_tokens, vocab] or [num_tokens, hidden]
        for name in ("sampled_token_probs", "sampled_token_logprobs",
                     "logprobs", "hidden_states"):
            t = getattr(out, name, None)
            if t is not None and t.size(0) > num_seqs:
                logger.debug(f"Trimming {name}: {t.shape} -> {t[row_slice].shape}")
                setattr(out, name, t[row_slice].contiguous())
        
        # sampled_token_ids must keep the same rows so indexing matches
        if hasattr(out, "sampled_token_ids") and out.sampled_token_ids is not None:
            if out.sampled_token_ids.size(0) > num_seqs:
                logger.debug(f"Trimming sampled_token_ids: {out.sampled_token_ids.shape} -> "
                           f"{out.sampled_token_ids[row_slice].shape}")
                out.sampled_token_ids = out.sampled_token_ids[row_slice].contiguous()
    
    def sampler_output(
        self,
        execute_model_req: "ExecuteModelRequest", 
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Generate draft tokens using early exit model."""
        # TODO: Handle bonus tokens for KV cache consistency
        if seq_ids_with_bonus_token_in_last_step:
            logger.debug(f"Bonus tokens present but not yet handled: "
                        f"{len(seq_ids_with_bonus_token_in_last_step)} sequences")
        
        # Create shallow copies of sequence metadata for local mutations
        # This allows us to append tokens without affecting the original
        local_seq_group_metadata_list = [
            MultiStepWorker._shallow_copy_seq_group_metadata(sgm)
            for sgm in execute_model_req.seq_group_metadata_list
        ]
        
        model_outputs: List[SamplerOutput] = []
        
        # Generate tokens autoregressively using our patched model
        for step in range(sample_len):
            # Create local request with our shallow-copied metadata
            local_execute_model_req = copy.copy(execute_model_req)
            local_execute_model_req.seq_group_metadata_list = local_seq_group_metadata_list
            local_execute_model_req.num_steps = 1
            
            # Execute model with local sequences
            step_outputs = self.worker.execute_model(local_execute_model_req)
            
            assert len(step_outputs) == 1
            step_output = step_outputs[0]
            
            # Debug: Log tokens before any processing
            if hasattr(step_output, 'sampled_token_ids') and step_output.sampled_token_ids is not None:
                logger.debug(f"Step {step} - Raw sampled_token_ids: shape={step_output.sampled_token_ids.shape}, "
                           f"values={step_output.sampled_token_ids.tolist()}")
            
            # CRITICAL: Keep only the last num_seqs rows (new tokens) for consistent shapes
            # This ensures torch.stack in sampler_output_to_torch succeeds
            num_seqs = len(local_seq_group_metadata_list)
            self._keep_last_row(step_output, num_seqs)
            
            # Debug: Log tokens after trimming
            if hasattr(step_output, 'sampled_token_ids') and step_output.sampled_token_ids is not None:
                logger.debug(f"Step {step} - After trim sampled_token_ids: shape={step_output.sampled_token_ids.shape}, "
                           f"values={step_output.sampled_token_ids.tolist()}")
            
            # Force GPU synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Update hidden states if available
            if hasattr(step_output, 'hidden_states') and step_output.hidden_states is not None:
                from vllm.sequence import HiddenStates
                local_execute_model_req.previous_hidden_states = HiddenStates(
                    step_output.hidden_states,
                    local_seq_group_metadata_list
                )
            
            # Append tokens to LOCAL copy only - won't affect original
            self._append_local_tokens(step_output, local_seq_group_metadata_list)
            
            # Debug: Verify what we're storing matches what we appended
            if hasattr(step_output, 'sampled_token_ids') and step_output.sampled_token_ids is not None:
                logger.debug(f"Step {step} - Storing sampled_token_ids for output: {step_output.sampled_token_ids.tolist()}")
            
            model_outputs.append(step_output)
            
            logger.debug(f"Early exit step {step}: generated "
                        f"{len(step_output.outputs)} outputs")
        
        # Debug: Log what we're returning for conversion
        logger.debug(f"Returning {len(model_outputs)} SamplerOutputs for proposal building")
        for i, output in enumerate(model_outputs):
            if hasattr(output, 'sampled_token_ids') and output.sampled_token_ids is not None:
                logger.debug(f"  Output {i}: sampled_token_ids = {output.sampled_token_ids.tolist()}")
        
        return model_outputs, True
    
    def _append_local_tokens(self, sampler_output: SamplerOutput, seq_group_metadata_list):
        """Append tokens to LOCAL shallow-copied sequences.
        
        This only affects our local copy used for multi-step generation.
        The original sequences remain untouched for MQAScorer.
        """
        if sampler_output.sampled_token_ids is not None:
            token_ids = sampler_output.sampled_token_ids.flatten()
            
            # Append tokens to each local sequence
            for i, seq_group_metadata in enumerate(seq_group_metadata_list):
                seq_group_metadata.is_prompt = False
                seq_id, seq_data = next(iter(seq_group_metadata.seq_data.items()))
                
                if i < len(token_ids):
                    token_id_int = token_ids[i].item()
                    # Append to local copy - original is untouched
                    seq_data.append_token_id(token_id_int, 0.0)
                    seq_data.update_num_computed_tokens(1)
                    logger.debug(f"Appended token {token_id_int} to local seq {seq_id}")
    
    def execute_model(self, execute_model_req: "ExecuteModelRequest") -> List[SamplerOutput]:
        """Execute the early exit model using our worker."""
        return self.worker.execute_model(execute_model_req)
    
    def get_model(self):
        """Get the model from our worker."""
        return self.worker.get_model()
    
    def get_cache_block_size_bytes(self):
        """Get cache block size from our worker."""
        return self.worker.get_cache_block_size_bytes()
    
    def profile_run(self):
        """Profile run - use our worker."""
        return self.worker.profile_run()
    
    def remove_lora(self, lora_id: int):
        """Remove LoRA - delegate to our worker."""
        return self.worker.remove_lora(lora_id)
    
    def pin_lora(self, lora_id: int):
        """Pin LoRA - delegate to our worker."""
        return self.worker.pin_lora(lora_id)
    
    def add_lora(self, lora_request):
        """Add LoRA - delegate to our worker."""
        return self.worker.add_lora(lora_request)
    
    def list_loras(self):
        """List LoRAs - delegate to our worker."""
        return self.worker.list_loras()
    
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize cache with our reduced allocation."""
        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine available blocks - use our worker's calculation."""
        return self.worker.determine_num_available_blocks()