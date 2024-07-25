import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, cast

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.sequence import (IntermediateTensors, PoolerOutput, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE, _PAD_SLOT_ID,
                                      LORA_WARMUP_RANK, GPUModelRunnerBase,
                                      ModelInputForGPUBuilder,
                                      ModelInputForGPUWithSamplingMetadata)

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchDecodeWithPagedKVCacheWrapper = None
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

from vllm.inputs import INPUT_REGISTRY
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.models.interfaces import supports_vision
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class EncoderDecoderModelInput(ModelInputForGPUWithSamplingMetadata):
    """
    Used by the EncoderDecoderModelRunner.
    """
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "encoder_input_tokens": self.encoder_input_tokens,
            "encoder_input_positions": self.encoder_input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "EncoderDecoderModelInput":
        return cast(
            EncoderDecoderModelInput,
            super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


class EncoderDecoderModelRunner(GPUModelRunnerBase[EncoderDecoderModelInput]):
    _model_input_cls: Type[EncoderDecoderModelInput] = (
        EncoderDecoderModelInput)
    _builder_cls: Type[ModelInputForGPUBuilder] = (ModelInputForGPUBuilder)

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
    ):
        super().__init__(model_config,
                         parallel_config,
                         scheduler_config,
                         device_config,
                         cache_config,
                         load_config,
                         lora_config=lora_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker,
                         prompt_adapter_config=prompt_adapter_config,
                         multimodal_config=multimodal_config)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: EncoderDecoderModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[PoolerOutput]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in "
                             "EncoderDecoderModelRunner")

        # if self.lora_config:
        #     assert model_input.lora_requests is not None
        #     assert model_input.lora_mapping is not None
        #     self.set_active_loras(model_input.lora_requests,
        #                           model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        if self.attn_backend.get_name() == "flashinfer":
            raise NotImplementedError("FlashInfer is currently not supported "
                                      "for encoder/decoder models.")

        # Currently cuda graph is not supported for encoder/decoder models
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            raise NotImplementedError("CUDAGraph is currently not supported "
                                      "for encoder/decoder models.")
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        # virtual_engine = model_input.virtual_engine
        # if prefill_meta is None and decode_meta.use_cuda_graph:
        #     assert model_input.input_tokens is not None
        #     graph_batch_size = model_input.input_tokens.shape[0]
        #     model_executable = self.graph_runners[virtual_engine][
        #         graph_batch_size]
        # else:
        model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            encoder_input_ids=model_input.encoder_input_tokens,
            encoder_positions=model_input.encoder_input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **multi_modal_kwargs,
            **seqlen_agnostic_kwargs)

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
            # elif decode_meta.use_cuda_graph:
            #     hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> EncoderDecoderModelInput:
        return EncoderDecoderModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> EncoderDecoderModelInput:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)

        model_input = self._prepare_encoder_model_input_tensors(
            seq_group_metadata_list, model_input)

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     self.pin_memory)
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        model_config = self.model_config

        if supports_vision(self.model):
            max_mm_tokens = MULTIMODAL_REGISTRY \
                .get_max_multimodal_tokens(model_config)
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            seq_data, dummy_multi_modal_data = INPUT_REGISTRY \
                .dummy_data_for_profiling(model_config, seq_len)

            # Having more tokens is over-conservative but otherwise fine
            assert len(seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(seq_data.prompt_token_ids)}")

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                encoder_seq_data=seq_data,
                cross_block_table=None,
                multi_modal_data=dummy_multi_modal_data,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def _prepare_encoder_model_input_tensors(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            model_input: EncoderDecoderModelInput) -> EncoderDecoderModelInput:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()
        prompt_adapter_index_mapping: List[int] = []
        prompt_adapter_prompt_mapping: List[int] = []
        prompt_adapter_requests: Set[PromptAdapterRequest] = set()

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        multi_modal_inputs_list: List[MultiModalInputs] = []
        decode_only = True
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        if len(seq_group_metadata_list) == 0:
            # Leave the encoder/cross-attention input
            # fields at default values if the seq group
            # metadata list arg is an empty list
            return model_input

        if self.sliding_window is not None:
            raise NotImplementedError()
            # sliding_window_blocks = (self.sliding_window + self.block_size -
            #                          1) // self.block_size
            # block_aligned_sliding_window = \
            #     sliding_window_blocks * self.block_size

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            computed_block_nums = None
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            seq_data = seq_group_metadata.encoder_seq_data
            cross_block_table = seq_group_metadata.cross_block_table
            if is_prompt:
                context_len = seq_data.get_num_computed_tokens()
            else:
                # get_num_computed_tokens is incorrect for spec decoding.
                # So, we should have a special logic here.
                # TODO(sang): Fix it.
                context_len = seq_data.get_len()

            seq_len = seq_data.get_len()

            if is_prompt:
                tokens = seq_data.get_token_ids()[context_len:seq_len]
            else:
                # Optimization. get_token_ids requires the entire copy of
                # tokens.
                tokens = [seq_data.get_last_token_id()]

            # Prefix cache was hit.
            # Prefix is not supported with sliding_window
            prefix_cache_hit = (computed_block_nums is not None
                                and len(computed_block_nums) > 0
                                and self.sliding_window is None and is_prompt)

            # These are seq_len/context_len capped to the sliding window.
            # They are passed to decode kernel.
            # We still need original seq_len/context_len to compute slot
            # mapping (and input position) below.
            curr_sliding_window_blocks = None
            sliding_seq_len = seq_len
            sliding_context_len = context_len

            # TODO(sang): This is a hack to make sliding window work with
            # paged attn. We can remove it if we make paged attn kernel
            # to properly handle slinding window attn.
            # if (self.sliding_window is not None and not is_prompt):
            #     curr_sliding_window_blocks = sliding_window_blocks
            #     if self.scheduler_config.use_v2_block_manager:
            #         # number of elements in last block
            #         suff_len = seq_len % self.block_size
            #         sliding_seq_len = min(
            #             seq_len, block_aligned_sliding_window + suff_len)
            #         if suff_len > 0:
            #             curr_sliding_window_blocks += 1
            #     else:
            #         sliding_seq_len = min(seq_len, self.sliding_window)
            #     sliding_context_len = sliding_seq_len - 1

            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            if prefix_cache_hit:
                assert computed_block_nums is not None
                context_len = len(computed_block_nums) * self.block_size
                tokens = tokens[context_len:]

                # need to think what to set it to when we have both sliding
                # window and prefix caching...
                assert self.sliding_window is None, \
                    "Prefix caching is not supported with sliding window"
                sliding_context_len = context_len

                if self.attn_backend.get_name() == "flash-attn":
                    # NOTE(woosuk): For flash-attn, the block table should
                    # include the entries for the incoming prefill tokens.
                    # TODO(woosuk): This is a temporary fix. We should
                    # provide a unified interface for different backends.
                    block_table = cross_block_table
                else:
                    block_table = computed_block_nums
            elif (self.scheduler_config.chunked_prefill_enabled
                  or not is_prompt):
                if cross_block_table is not None:
                    # chunked prefill or decode
                    block_table = cross_block_table
                    if curr_sliding_window_blocks is not None:
                        block_table = block_table[-curr_sliding_window_blocks:]
                else:
                    # Only happens when memory profiling runs.
                    block_table = []
            else:
                # Prefill without chunked prefill or memory profiling.
                block_table = []
            block_tables.append(block_table)

            seq_lens.append(sliding_seq_len)
            context_lens.append(sliding_context_len)
            query_len = sliding_seq_len - sliding_context_len
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            input_positions.extend(list(range(context_len, seq_len)))
            lora_id = seq_group_metadata.lora_int_id
            prompt_adapter_id = seq_group_metadata.prompt_adapter_id

            if is_prompt:
                assert len(seq_ids) == 1
                num_prefills += 1
                num_prefill_tokens += len(tokens)
                decode_only = False
                prefill_seq_lens.append(seq_len)
            else:
                # assert is_encoder_seq or query_len == 1, (
                #     "seq_len: {}, context_len: {}, query_len: {}".format(
                #         seq_len, context_len, query_len))
                num_decode_tokens += query_len
                decode_seq_lens.append(sliding_seq_len)

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * query_len
            lora_prompt_mapping.extend(
                [lora_id] *
                (query_len if seq_group_metadata.sampling_params and
                 seq_group_metadata.sampling_params.prompt_logprobs is not None
                 else 1))

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                # Process multi-modal data
                mm_kwargs = self.multi_modal_input_mapper(mm_data)
                multi_modal_inputs_list.append(mm_kwargs)

            if prompt_adapter_id > 0 and is_prompt:
                prompt_adapter_requests.add(
                    seq_group_metadata.prompt_adapter_request)

                num_tokens = seq_group_metadata.\
                                        prompt_adapter_num_virtual_tokens
                pm = [prompt_adapter_id
                      ] * num_tokens + [0] * (query_len - num_tokens)
                prompt_adapter_index_mapping += pm
                prompt_adapter_prompt_mapping.extend(
                    [prompt_adapter_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     else 1))

            is_profile_run = _is_single_block_table_empty(
                seq_group_metadata.block_tables)
            if is_profile_run:
                # During memory profiling, the block tables are not
                # initialized yet. In this case, we just use a dummy
                # slot mapping.
                # In embeddings, the block tables are {seq_id: None}.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = cross_block_table

            # Mask the [0, start_idx) tokens of the prompt with
            # _PAD_SLOT_ID, where start_idx is max(0, seq_len -
            # sliding_window). For example, if the prompt len is 10,
            # sliding window is 8, and block size is 4, the first two
            # tokens are masked and the slot mapping will be
            # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                if is_prompt:
                    assert self.scheduler_config.use_v2_block_manager \
                        or context_len == 0, (
                        "Prefix caching is currently not supported with "
                        "sliding window attention in V1 block manager")
                # It is an optimization. When it is decoding, it is always
                # 0. When prefill, we use it to not write slots to kv cache
                # to save memory.
                start_idx = max(0, query_len - self.sliding_window)

            for i in range(context_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

            # # Prepare input tensors for flashinfer
            # if self.attn_backend.get_name() == "flashinfer":
            #     assert False

        batch_size = len(input_tokens)
        max_query_len = max(query_lens)
        max_seq_len = (max(prefill_seq_lens, default=0)
                       if is_prompt else max(decode_seq_lens, default=0))

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        use_captured_graph = (decode_only
                              and not self.model_config.enforce_eager
                              and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
                              and max_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            raise NotImplementedError("CUDAGraph is currently not supported "
                                      "for encoder/decoder models.")

        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )
        assert (not is_prompt) or max_query_len > 0, (
            "Decode-phase query_lens: {}".format(query_lens))

        # context_lens_tensor = torch.tensor(context_lens,
        #                                    dtype=torch.int,
        #                                    device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)

        # Set encoder-oriented attention metadata fields
        attn_metadata.num_encoder_tokens = sum(seq_lens)
        attn_metadata.encoder_seq_lens = seq_lens
        attn_metadata.encoder_seq_lens_tensor = seq_lens_tensor
        attn_metadata.max_encoder_seq_len = max_seq_len
        attn_metadata.cross_slot_mapping = slot_mapping_tensor
        attn_metadata.cross_block_tables = block_tables

        if seq_group_metadata.is_prompt:

            input_tokens_tensor = torch.tensor(input_tokens,
                                               dtype=torch.long,
                                               device=self.device)
            input_positions_tensor = torch.tensor(input_positions,
                                                  dtype=torch.long,
                                                  device=self.device)

        else:

            input_tokens_tensor = torch.tensor([],
                                               dtype=torch.long,
                                               device=self.device)
            input_positions_tensor = torch.tensor([],
                                                  dtype=torch.long,
                                                  device=self.device)

        # Inject attn_metadata encoder/cross-attention fields &
        # encoder input tokens/positions into model_input.
        # Frozen dataclass fields cannot be modified, so use
        # dataclasses.replace to construct a new model input
        # instance.
        model_input = dataclasses.replace(
            model_input,
            attn_metadata=attn_metadata,
            encoder_input_tokens=input_tokens_tensor,
            encoder_input_positions=input_positions_tensor,
        )

        logits_soft_cap = getattr(self.model_config.hf_config,
                                  'attn_logit_softcapping', None)
        if logits_soft_cap is not None and self.attn_backend.get_name(
        ) != "flashinfer":
            raise ValueError("Models with logits_soft_cap (i.e., Gemma-2)"
                             " require FlashInfer backend, however vLLM"
                             " currently only supports xFormers backend"
                             " for encoder/decoder models.")

        return model_input


def _is_single_block_table_empty(block_table: Optional[List[int]]):
    """
    Check if a single block table has not been constructed
    """
    if block_table is None:
        return True
    return False
