import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from transformers import TopPLogitsWarper

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                         ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.model_executor.models import supports_multimodal
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata, Logprob, SequenceOutput, CompletionSequenceGroupOutput, SequenceGroup
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.transformers_utils.config import uses_mrope
from vllm.utils import make_tensor_with_pad

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.
    """
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    prompt_lens: Optional[torch.Tensor] = None
    seq_groups: Optional[List[int]] = None
    block_tables: Optional[torch.Tensor] = None
    unpadded_batch_size: Optional[int] = None
    tt_sampling_params: Optional[TTSamplingParams] = None
    multi_modal_kwargs: Optional[List[Dict[str, Any]]] = None
    cross_block_tables: Optional[torch.Tensor] = None
    is_first_multi_step: bool = True
    is_last_step: bool = True
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
            "seq_groups": self.seq_groups,
            "block_tables": self.block_tables,
            "unpadded_batch_size": self.unpadded_batch_size,
            "tt_sampling_params": self.tt_sampling_params,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "cross_block_tables": self.cross_block_tables,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }
        
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
            cls: Type["TTModelInput"],
            tensor_dict: Dict[str, Any],
    ) -> "TTModelInput":
        return cls(**tensor_dict)
    

def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    if k == -1:  # no top-k sampling
        top_k_values, top_k_indices = logits, torch.arange(logits.shape[-1]).unsqueeze(0).repeat(logits.shape[0],1)
    else:
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = TopPLogitsWarper(top_p=p)(None, top_k_values)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(probs)  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


class TTModelRunner(ModelRunnerBase[TTModelInput]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        trace_mode: bool = True,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # Currently, TT worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        self.trace_mode = trace_mode  # whether to use ttnn tracing for model execution
        override_tt_config = model_config.override_tt_config
        if override_tt_config is not None and "sample_on_device_mode" in override_tt_config:
            self.sample_on_device_mode = override_tt_config["sample_on_device_mode"]
            assert self.sample_on_device_mode in ["all", "decode_only"], f"Invalid sample_on_device_mode: {self.sample_on_device_mode}"
        else:
            self.sample_on_device_mode = None  # whether to sample on device
        logger.info(f"TTModelRunner: trace_mode={self.trace_mode}, sample_on_device_mode={self.sample_on_device_mode}")

        self.cached_step_outputs: List[torch.Tensor] = []  # Only used for multi-step execution
        
        if self.model_config.is_encoder_decoder_model:
            self.cached_enc_dec_data: Optional[Dict[int, Dict[str, Any]]] = None  # seq_id -> enc_dec_data

        if self.model_is_mrope:
            assert "TTModelRunner does not currently support models with mrope rope_scaling"
        
    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        return uses_mrope(self.model_config.hf_config)

    def load_model(self) -> None:
        # Note: using custom TT loader instead of selecting from default vllm loaders
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(model_config=self.model_config,
            device_config=self.device_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config
        )
        if self.model_config.is_encoder_decoder_model:
            self.max_cross_blocks = self.model.max_cross_attn_tokens // self.cache_config.block_size

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> TTModelInput:
        return TTModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
        )

    def validate_seq_group(self, seq_group: SequenceGroup) -> None:
        '''
        Validate a sequence group before it is scheduled for execution.
        Called by TTExecutor::validate_seq_group before the sequence group 
        is scheduled for execution in LLMEngine::_add_processed_request.
        '''
        
        sampling_params = seq_group.sampling_params
        
        if seq_group.num_seqs() != 1:
            raise ValueError("Currently only supporting one sequence per request group")
        if sampling_params.n != 1:
            raise ValueError("Currently only supporting n=1")
        if sampling_params.best_of is not None:
            raise ValueError("Currently not supporting best_of")
        if sampling_params.logprobs is not None:
            raise ValueError("Currently not supporting logprobs")
        if sampling_params.prompt_logprobs is not None:
            raise ValueError("Currently not supporting prompt_logprobs")

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> TTModelInput:
        
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt  # prefill if True, otherwise decode
        assert all(x.is_prompt == is_prompt for x in seq_group_metadata_list), "Currently only supporting all prefills or all decodes in seq group"
        
        unpadded_batch_size = len(seq_group_metadata_list)
        assert unpadded_batch_size > 0
        
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prompt_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_groups: List[int] = []
        top_pk_sampling_params = {}
        multi_modal_kwargs: Dict[str, Any] = {}
        if supports_multimodal(self.model) and is_prompt:
            multi_modal_kwargs = {"images": []}
        cross_block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1, "Currently only supporting one sequence per request group"
            seq_id = seq_ids[0]
            seq_groups.append(seq_id)

            multi_modal_data = seq_group_metadata.multi_modal_data
            seq_data = seq_group_metadata.seq_data[seq_id]
            
            if is_prompt:
                # tokens
                prompt_tokens = seq_data.get_token_ids()
                input_tokens.append(prompt_tokens)
                
                # prompt lengths
                prompt_lens.append(len(prompt_tokens))
            else:
                # tokens
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)
                
                # positions
                position = seq_data.get_len() - 1
                input_positions.append(position)
                
            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables.append(block_table)
            
            # Multi-modal data
            # TODO: Replace with multi_modal_input_mapper (used by CPU/GPU model runners) once TT models no longer require raw PIL images
            if supports_multimodal(self.model) and is_prompt:
                if (multi_modal_data := seq_group_metadata.multi_modal_data):
                    assert "image" in multi_modal_data, "Currently only supporting image multi-modal inputs"
                    image = multi_modal_data["image"]  # this is of type PIL.Image.Image
                    multi_modal_kwargs["images"].append(image)
                else:
                    multi_modal_kwargs["images"].append(None)
            
            # Encoder-decoder data (currently only supporting cross attention metadata and not additional encoder data)
            if self.model_config.is_encoder_decoder_model:
                cross_block_table = seq_group_metadata.cross_block_table
                cross_block_tables.append(cross_block_table)
            
            # Sampling params
            # TODO: Add support for different sampling params in the same batch
            sampling_params = seq_group_metadata.sampling_params
            if len(top_pk_sampling_params) == 0:
                top_pk_sampling_params["temperature"] = sampling_params.temperature
                top_pk_sampling_params["top_k"] = sampling_params.top_k
                top_pk_sampling_params["top_p"] = sampling_params.top_p
            else:
                if top_pk_sampling_params["temperature"] != sampling_params.temperature:
                    logger.warning(f"Currently only supporting same temperature for all sequences in batch, falling back to first sequence's temperature ({top_pk_sampling_params['temperature']})")
                if top_pk_sampling_params["top_k"] != sampling_params.top_k:
                    logger.warning(f"Currently only supporting same top_k for all sequences in batch, falling back to first sequence's top_k ({top_pk_sampling_params['top_k']})")
                if top_pk_sampling_params["top_p"] != sampling_params.top_p:
                    logger.warning(f"Currently only supporting same top_p for all sequences in batch, falling back to first sequence's top_p ({top_pk_sampling_params['top_p']})")
        
        tt_sampling_params = TTSamplingParams(
            temperature=top_pk_sampling_params["temperature"],
            top_k=top_pk_sampling_params["top_k"],
            top_p=top_pk_sampling_params["top_p"]
        )
        
        # Remove cached encoder-decoder data for any seq ids that are not in the current batch (assume they were either finished or preempted)
        if self.model_config.is_encoder_decoder_model and not is_prompt and self.cached_enc_dec_data:
            seq_ids_to_del = []
            for seq_id in self.cached_enc_dec_data:
                if seq_id not in seq_groups:
                    seq_ids_to_del.append(seq_id)
            for seq_id in seq_ids_to_del:
                del self.cached_enc_dec_data[seq_id]
        
        # Convert lists to tensors and add padding
        
        block_tables = make_tensor_with_pad(
            block_tables,
            dtype=torch.int32,
            device="cpu",
            pad=0
        )
        if self.model_config.is_encoder_decoder_model:
            cross_block_tables = make_tensor_with_pad(
                cross_block_tables,
                dtype=torch.int32,
                device="cpu",
                pad=0
            )
        else:
            cross_block_tables = None
        if is_prompt:
            input_tokens = make_tensor_with_pad(
                input_tokens, 
                dtype=torch.int32, 
                device="cpu", 
                pad=0
            )
            input_positions = 0
            prompt_lens = torch.tensor(
                prompt_lens,
                dtype=torch.int32,
                device="cpu"
            )
        else:
            input_tokens = torch.tensor(input_tokens, dtype=torch.int32, device="cpu").view(-1, 1)
            input_positions = torch.tensor(input_positions, dtype=torch.int32, device="cpu")
            prompt_lens = None
            
            # TODO: Remove once TT models can support arbitrary batch sizes
            # Pad batch to max_num_seqs
            if input_tokens.shape[0] < self.scheduler_config.max_num_seqs:
                batch_pad_len = self.scheduler_config.max_num_seqs - input_tokens.shape[0]
                input_tokens = torch.cat([
                    input_tokens,
                    torch.zeros(batch_pad_len, 1, dtype=torch.int32, device="cpu")
                ])
                input_positions = torch.cat([
                    input_positions,
                    torch.ones(batch_pad_len, dtype=torch.int32, device="cpu") * -1  # Pad with -1 to indicate no position
                ])
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(batch_pad_len, block_tables.shape[1], dtype=torch.int32, device="cpu")
                ])
                if self.model_config.is_encoder_decoder_model:
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(batch_pad_len, cross_block_tables.shape[1], dtype=torch.int32, device="cpu")
                    ])
            
            # Pad block_tables to max num blocks so ttnn tracing can work (requires constant shape)
            if self.trace_mode:
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(block_tables.shape[0], self.cache_config.num_gpu_blocks - block_tables.shape[1], dtype=torch.int32, device="cpu")
                ], dim=1)
                if self.model_config.is_encoder_decoder_model:
                    # Note for vision models: the number of cross blocks may change if the number of image tiles changes or if prompts are text-only
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(cross_block_tables.shape[0], self.max_cross_blocks - cross_block_tables.shape[1], dtype=torch.int32, device="cpu")
                    ], dim=1)
        
        return TTModelInput(input_tokens, input_positions, prompt_lens, seq_groups, block_tables, unpadded_batch_size, tt_sampling_params, multi_modal_kwargs, cross_block_tables)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: TTModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        is_decode = model_input.prompt_lens is None
        
        # Note on async_out_proc + multi-step: for gpu/tpu, the N steps are enqueued on device and the last step
        # will trigger the output processor for all outputs but the last. Currently for TT, the inputs/outputs of each step
        # are transferred between host/device, so async_out_proc does not help unless using async_out_proc_per_trace
        # which triggers the output processor for step (i) on host while device is executing step (i+1).
        use_async_out_proc = model_input.async_callback is not None
        async_out_proc_per_trace = self.trace_mode and self.scheduler_config.is_multi_step and use_async_out_proc
        
        if not is_decode:
            assert num_steps == 1, "Num steps must be 1 for prefill"

        if model_input.is_first_multi_step:  # always true if not using multi-step
            self.cached_step_outputs = []
            for i in range(num_steps):
                next_token_ids = self._execute_model_single_step(model_input, kv_caches, is_decode, async_out_proc_per_trace, step_idx=i)
                self.cached_step_outputs.append(next_token_ids)
                
                if i < num_steps - 1:
                    # Prepare the inputs for the next step
                    new_input_tokens = next_token_ids.unsqueeze(dim=1).int()
                    if new_input_tokens.shape[0] < self.scheduler_config.max_num_seqs:
                        # Pad batch to max_num_seqs
                        batch_pad_len = model_input.input_tokens.shape[0] - new_input_tokens.shape[0]
                        new_input_tokens = torch.cat([
                            new_input_tokens,
                            torch.zeros(batch_pad_len, 1, dtype=torch.int32, device="cpu")
                        ])
                    
                    # Update input positions for all except those that are -1 (padding)
                    new_input_positions = torch.where(
                        model_input.input_positions == -1,
                        model_input.input_positions,
                        model_input.input_positions + 1
                    )

                    model_input = dataclasses.replace(
                        model_input,
                        input_tokens=new_input_tokens,
                        input_positions=new_input_positions
                    )
                    
            if use_async_out_proc:
                model_input.async_callback()  # trigger output processor

        sampler_outputs = []  # no outputs unless last step
        if model_input.is_last_step:  # always true if not using multi-step
            num_outputs = len(self.cached_step_outputs)
            if async_out_proc_per_trace:
                assert num_outputs == 1, "Last step should only have one output"
            for i in range(num_outputs):
                next_token_ids = self.cached_step_outputs.pop(0)
                # TODO: add read back from device once model can keep executing steps on device
                sampler_output = self._make_sampler_output(
                    next_token_ids,
                    model_input.seq_groups
                )
                sampler_outputs.append(sampler_output)
                if i < num_outputs - 1 and use_async_out_proc:
                    self._send_async_out(sampler_output, model_input.async_callback, is_first_step_output=i == 0)
            if use_async_out_proc:
                return [sampler_outputs[-1]]  # only return the last output for async output processor
        
        return sampler_outputs
    
    def _send_async_out(self, sampler_output, async_callback, is_first_step_output):
        ctx = async_callback.keywords["ctx"]
        ctx.append_output(
            outputs=[sampler_output],
            seq_group_metadata_list=ctx.seq_group_metadata_list,
            scheduler_outputs=ctx.scheduler_outputs,
            is_async=False,
            is_last_step=False,
            is_first_step_output=is_first_step_output)
        async_callback()  # trigger output processor
    
    def _make_sampler_output(
        self,
        next_token_ids: List[int],
        seq_groups: List[int],
    ) -> SamplerOutput:
        # Minimal code to construct the sampler outputs, based on tpu_model_runner.py
        # TT backend does not support the advanced sampling parameters such as logprobs.
        zero_logprob = Logprob(0.0)
        sampler_outputs = []
        for batch_idx, seq_id in enumerate(seq_groups):
            next_token_id = int(next_token_ids[batch_idx])
            seq_outputs = [SequenceOutput(seq_id, next_token_id,
                                {next_token_id: zero_logprob})]
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
        return SamplerOutput(sampler_outputs)
    
    def _send_prev_step_async_out(self, model_input: TTModelInput, step_idx):
        if step_idx > 0:
            next_token_ids = self.cached_step_outputs.pop(0)
            sampler_output = self._make_sampler_output(
                next_token_ids,
                model_input.seq_groups
            )
            self._send_async_out(sampler_output, model_input.async_callback, is_first_step_output=(step_idx == 1))
    
    def _execute_model_single_step(self, model_input: TTModelInput, kv_caches: List[torch.Tensor], is_decode, async_out_proc_per_trace=False, step_idx=0):
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": kv_caches,
            **(model_input.multi_modal_kwargs or {}),
        }
        if not is_decode:
            execute_model_kwargs["prompt_lens"] = model_input.prompt_lens
        else:
            execute_model_kwargs["start_pos"] = model_input.input_positions
        if self.sample_on_device_mode == "all" or (self.sample_on_device_mode == "decode_only" and is_decode):
            execute_model_kwargs["sampling_params"] = model_input.tt_sampling_params
        if model_input.cross_block_tables is not None:
            execute_model_kwargs["cross_page_table"] = model_input.cross_block_tables
        
        if not is_decode:
            outputs = self.model.prefill_forward(**execute_model_kwargs)
            
            if self.model_config.is_encoder_decoder_model:
                # Save encoder-decoder data for use in subsequent decode steps (may need to be updated for future models)
                tt_out, cross_attention_masks, full_text_row_masked_out_mask = outputs
                if self.cached_enc_dec_data is None:
                    self.cached_enc_dec_data = {}
                for i, seq_id in enumerate(model_input.seq_groups):
                    enc_dec_data = {"cross_attention_masks": cross_attention_masks[i], 
                                        "full_text_row_masked_out_mask": full_text_row_masked_out_mask[i]}
                    self.cached_enc_dec_data[seq_id] = enc_dec_data
            else:
                tt_out = outputs  # [batch_size, seq_len, vocab_size]
        else:
            if self.model_config.is_encoder_decoder_model:
                # Use encoder-decoder data from prefill step
                cross_attention_masks = [self.cached_enc_dec_data[seq_id]["cross_attention_masks"] for seq_id in model_input.seq_groups]
                full_text_row_masked_out_mask = [self.cached_enc_dec_data[seq_id]["full_text_row_masked_out_mask"] for seq_id in model_input.seq_groups]
                enc_dec_kwargs = {"cross_attention_masks": cross_attention_masks,
                                        "full_text_row_masked_out_mask": full_text_row_masked_out_mask}
            else:
                enc_dec_kwargs = {}
            
            tt_out = self.model.decode_forward(
                **execute_model_kwargs, **enc_dec_kwargs, enable_trace=self.trace_mode, read_from_device=False
            )
            if async_out_proc_per_trace:
                # trigger output processor on host while device is executing next step
                self._send_prev_step_async_out(model_input, step_idx)
            tt_out = self.model.read_decode_output(tt_out, model_input.unpadded_batch_size, is_tokens=(self.sample_on_device_mode is not None))

        # Note: for other devices, vLLM applies vllm.model_executor.layers.logits_processor::LogitsProcessor::_apply_logits_processors on logits, we don't use this
        # Note: for other devices, vLLM applies vllm.model_executor.layers.sampler::Sampler for sampling tokens, we don't use this
        if not self.sample_on_device_mode or (self.sample_on_device_mode == "decode_only" and not is_decode):
            next_logits = tt_out[:model_input.unpadded_batch_size, -1, :]  # unpadded batch, vocab of last token
            next_token_ids = self._sample_tokens(next_logits, model_input.tt_sampling_params)
        else:
            next_token_ids = tt_out
        
        return next_token_ids

    def _sample_tokens(self, logits, tt_sampling_params : TTSamplingParams):
        if tt_sampling_params.temperature == 0:  # greedy decoding
            return torch.argmax(logits, dim=-1)
        else:  # top-k top-p sampling
            return top_pk_logits_efficient(
                logits,
                p=tt_sampling_params.top_p,
                k=tt_sampling_params.top_k,
                temperature=tt_sampling_params.temperature
            )