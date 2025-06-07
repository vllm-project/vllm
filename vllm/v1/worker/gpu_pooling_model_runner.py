# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.distributed

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class GPUPoolingModelRunner(GPUModelRunner):
    """
    GPU model runner for pooling/embedding models in V1 architecture.
    
    This runner extends the base GPUModelRunner but skips sampling logic
    and instead performs pooling to produce embeddings, classifications,
    or other pooled outputs.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Pooling models don't use sampling - set these to None after super init
        self.sampler = None  # type: ignore
        self.rejection_sampler = None  # type: ignore

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """Execute model for pooling tasks (embedding, classification, etc.)."""

        # Update internal state from scheduler output
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Use the same input preparation as the base class
        attn_metadata, logits_indices, _ = self._prepare_inputs(scheduler_output)
        
        # Run the model forward pass like in the base class
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens  # Simplified for pooling

        # Handle multimodal inputs if needed
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Prepare inputs similar to base class
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Forward pass through the model  
        from vllm.forward_context import set_forward_context
        with set_forward_context(attn_metadata, self.vllm_config, 
                                 num_tokens=num_input_tokens):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # For pipeline parallel, handle intermediate tensors
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Get pooling metadata for last rank
        pooling_metadata = self._get_pooling_metadata()

        # Perform pooling instead of sampling
        pooled_output = self.model.pooler(hidden_states, pooling_metadata)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=None,  # No sampling for pooling models
            spec_token_ids=None,
            logprobs=None,  # No logprobs for pooling models
            prompt_logprobs_dict={},
            pooled_outputs=pooled_output,  # Add pooled outputs
        )

    def _get_pooling_metadata(self) -> PoolingMetadata:
        """Create pooling metadata from current batch state."""
        seq_groups = []
        seq_lens = []
        prompt_lens = []

        for req_id in self.input_batch.req_ids:
            # For pooling models, we typically process the entire sequence
            req_state = self.requests[req_id]
            seq_len = req_state.num_tokens

            seq_groups.append([0])  # Single sequence per group for embedding
            seq_lens.append(seq_len)
            prompt_lens.append(seq_len)  # For pooling, entire input is "prompt"

        return PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_lens,
            prompt_lens=prompt_lens,
        )

    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        """Override dummy sampler run since pooling models don't use sampling."""
        # For pooling models, we need to test pooling instead of sampling
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Create dummy pooling metadata
        dummy_pooling_metadata = PoolingMetadata(
            seq_groups=[[0] for _ in range(batch_size)],
            seq_data=[seq_len for _ in range(batch_size)],
            prompt_lens=[seq_len for _ in range(batch_size)],
        )

        # Test pooling operation
        if hasattr(self.model, 'pooler'):
            self.model.pooler(hidden_states, dummy_pooling_metadata)
