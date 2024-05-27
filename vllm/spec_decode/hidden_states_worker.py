from typing import List, Optional

from vllm.sequence import SequenceGroupMetadata, ExecuteModelRequest, SamplerOutput
from vllm.worker.worker import Worker
import torch

class HiddenStatesWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speculator = None
        self.prev_request_context_lengths = {}

    def _get_hidden_states(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ):

        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.model_runner.prepare_input_tensors(seq_group_metadata_list)

        if self.model_runner.lora_config:
            self.model_runner.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.model_runner.graph_runners[graph_batch_size]
        else:
            model_executable = self.model_runner.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        # save the previous hidden states for later use
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model_runner.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.model_runner.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model_runner.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # we only need to pass hidden states of most recent token
        if seq_group_metadata_list[0].is_prompt:
            hidden_states = hidden_states.index_select(0, sampling_metadata.selected_token_indices)

        return output, hidden_states


    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:

        sampler_output, hidden_states = self._get_hidden_states(execute_model_req.seq_group_metadata_list, self.gpu_cache)

        # if we are executing the prompt, we need to flag the first decode step since pruning is handled differently
        if execute_model_req.seq_group_metadata_list[0].is_prompt:
            self.speculator.first_decode_step = True

        self.speculator.previous_hidden_state = hidden_states
        return [sampler_output]
