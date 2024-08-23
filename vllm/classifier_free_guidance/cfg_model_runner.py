from typing import List, Optional, Union

import torch

from vllm.distributed import get_pp_group
from vllm.multimodal import MultiModalInputs
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.worker.model_runner import (ModelRunner, ModelInputForGPUWithSamplingMetadata,
                                      FLASHINFER_WORKSPACE_BUFFER_SIZE, BatchDecodeWithPagedKVCacheWrapper,
                                      BatchPrefillWithPagedKVCacheWrapper)


class CFGModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def model_execute(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        if self.attn_backend.get_name() == "flashinfer":
            assert model_input.attn_metadata is not None
            assert model_input.input_tokens is not None
            if self.flashinfer_decode_workspace_buffer is None:
                self.flashinfer_decode_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_decode_wrapper = \
                    BatchDecodeWithPagedKVCacheWrapper(
                    self.flashinfer_decode_workspace_buffer, "NHD")
                self.flashinfer_prefill_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_prefill_wrapper = \
                    BatchPrefillWithPagedKVCacheWrapper(
                    self.flashinfer_prefill_workspace_buffer, "NHD")

            model_input.attn_metadata.prefill_wrapper = \
                self.flashinfer_prefill_wrapper
            if model_input.attn_metadata.use_cuda_graph:
                batch_size = model_input.input_tokens.shape[0]
                model_input.attn_metadata.decode_wrapper = self.graph_runners[
                    model_input.
                    virtual_engine][batch_size].flashinfer_decode_wrapper
            else:
                model_input.attn_metadata.decode_wrapper = \
                    self.flashinfer_decode_wrapper
            model_input.attn_metadata.begin_forward()

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            raise NotImplementedError("")

        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                         device=self.device),
            **seqlen_agnostic_kwargs)

        return hidden_or_intermediate_states

    @torch.inference_mode()
    def get_logits(
        self,
        hidden_or_intermediate_states: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model._get_logits(hidden_or_intermediate_states, 
                                      model_input.sampling_metadata)

    @torch.inference_mode()
    def compute_logits(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model.compute_logits(logits,
                                         model_input.sampling_metadata)

    @torch.inference_mode()
    def do_sample(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ):
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        if self.return_hidden_states:
            raise NotImplementedError("return_hidden_states is not supported in CFGModelRunner")

        return [output]

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:

        hidden_or_intermediate_states = self.model_execute(model_input, kv_caches, intermediate_tensors, num_steps)

        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        hidden_or_intermediate_states = self.get_logits(hidden_or_intermediate_states, model_input)
        logits = self.compute_logits(hidden_or_intermediate_states, model_input)

        return self.do_sample(logits, model_input)
