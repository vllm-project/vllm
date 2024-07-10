from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch

from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
# from vllm.distributed.communication_op import graph_capture
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import (LIST_ENC_DEC_SUPPORTED_BACKENDS,
                        STR_NOT_IMPL_ENC_DEC_BACKEND,
                        STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL,
                        STR_NOT_IMPL_ENC_DEC_CUDAGRAPH,
                        STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
                        STR_NOT_IMPL_ENC_DEC_SWA, make_tensor_with_pad)
from vllm.worker.model_runner import LORA_WARMUP_RANK, ModelInput, ModelRunner

logger = init_logger(__name__)

# Error message if EncoderDecoderModelRunner is used with
# a non-encoder/decoder model (i.e. decoder-only)
STR_ENCDECMR_ENCODER_DECODER_REQUIRED = \
    "Only encoder/decoder models may be executed " + \
        "using EncoderDecoderModelRunner"


class EncoderInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor

    @classmethod
    def empty(cls, device):
        return EncoderInput(
            input_tokens=torch.empty(0, device=device),
            input_positions=torch.empty(0, device=device),
        )


class EncoderDecoderModelRunner(ModelRunner):

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
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        super().__init__(model_config, parallel_config, scheduler_config,
                         device_config, cache_config, load_config, lora_config,
                         kv_cache_dtype, is_driver_worker,
                         vision_language_config)

        self._check_encoder_decoder_unsupported_scenarios()

    def _check_encoder_decoder_unsupported_scenarios(self):
        '''
        Catch and raise NotImplemented errors if features unsupported
        for encoder/decoder models are enabled, or if an otherwise
        unsupported scenario is configured.
        '''

        if not self._is_encoder_decoder_model():
            # Fail if EncoderDecoderModelRunner is constructed for a
            # non-encoder/decoder model i.e. decoder-only
            raise AttributeError(STR_ENCDECMR_ENCODER_DECODER_REQUIRED)

        if self.scheduler_config.chunked_prefill_enabled:
            # Fail if chunked prefill is enabled
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL)

        if self.cache_config.enable_prefix_caching:
            # Fail if prefix caching is enabled
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE)

        if self.sliding_window is not None:
            # Fail if sliding window is enabled
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_SWA)

        if not self.model_config.enforce_eager:
            # Fail if CUDA graph is enabled
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_CUDAGRAPH)

        backend_name = self.attn_backend.get_name()
        caps_backend_name = backend_name.upper()
        if caps_backend_name not in LIST_ENC_DEC_SUPPORTED_BACKENDS:
            # Fail if the selected backend is not supported for
            # encoder decoder models.
            msg = STR_NOT_IMPL_ENC_DEC_BACKEND + \
                f" {backend_name}; supported backends: " + \
                    "{str(LIST_ENC_DEC_SUPPORTED_BACKENDS)}"

            raise NotImplementedError(msg)

    def _prepare_encoder_model_input(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            attn_metadata: AttentionMetadata) -> EncoderInput:
        """Prepare the encoder input based on a given sequence group.

        Encoder attention is an entirely prefill-phase operation.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        multi_modal_kwargs_list: Dict[str,
                                      List[torch.Tensor]] = defaultdict(list)
        decode_only = True
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        sliding_window_blocks = 0
        block_aligned_sliding_window = 0

        # The following fields are only for flashinfer
        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        paged_kv_last_page_len: List[int] = []

        if len(seq_group_metadata_list) == 0:
            return ModelInput.empty(self.device)

        if self.sliding_window is not None:
            sliding_window_blocks = (self.sliding_window + self.block_size -
                                     1) // self.block_size
            block_aligned_sliding_window = \
                sliding_window_blocks * self.block_size

        for seq_group_metadata in seq_group_metadata_list:
            computed_block_nums = None  #seq_group_metadata.computed_block_nums

            is_prompt = seq_group_metadata.is_prompt

            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            seq_data = seq_group_metadata.encoder_seq_data
            block_table = seq_group_metadata.cross_block_table
            decode_only, \
            num_prefills, \
            num_prefill_tokens, \
            num_decode_tokens = self._prepare_seq_model_input(
                is_prompt, decode_only, num_prefills, num_prefill_tokens,
                num_decode_tokens, block_tables, seq_lens, slot_mapping,
                context_lens, query_lens, input_tokens, input_positions,
                prefill_seq_lens, decode_seq_lens, seq_group_metadata,
                seq_data, computed_block_nums, block_table,
                paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
                sliding_window_blocks, block_aligned_sliding_window,
                lora_index_mapping, lora_prompt_mapping, lora_requests,
                multi_modal_kwargs_list, is_encoder_seq=True)

        max_query_len = max(query_lens)

        max_seq_len = max(prefill_seq_lens, default=0) if is_prompt else \
                        max(decode_seq_lens, default=0)

        # Assume Eager Mode
        # TODO: CUDA Graph support
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
            "query_lens: {}".format(query_lens))

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=self.device)

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

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

            return EncoderInput(input_tokens=input_tokens_tensor,
                                input_positions=input_positions_tensor)

        else:

            input_tokens_tensor = torch.tensor([],
                                               dtype=torch.long,
                                               device=self.device)
            input_positions_tensor = torch.tensor([],
                                                  dtype=torch.long,
                                                  device=self.device)

        return EncoderInput(input_tokens=input_tokens_tensor,
                            input_positions=input_positions_tensor)

    def prepare_input_tensors_encoder_decoder(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               AttentionMetadata, SamplingMetadata, Set[LoRARequest],
               LoRAMapping, Dict[str, torch.Tensor]]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            # Prepare input tensors.
            (
                input_tokens,
                input_positions,
                attn_metadata,
                seq_lens,
                query_lens,
                lora_mapping,
                lora_requests,
                multi_modal_kwargs,
                slot_mapping,
                num_prefill_tokens,
                num_decode_tokens,
                num_prefills,
            ) = self._prepare_model_input(seq_group_metadata_list)
            (encoder_input_tokens,
             encoder_input_positions) = self._prepare_encoder_model_input(
                 seq_group_metadata_list, attn_metadata)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_kwargs": multi_modal_kwargs,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
                "encoder_input_tokens": encoder_input_tokens,
                "encoder_input_positions": encoder_input_positions
            }
            if attn_metadata:
                metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            encoder_input_tokens = metadata_dict.pop("encoder_input_tokens")
            encoder_input_positions = metadata_dict.pop(
                "encoder_input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_kwargs = metadata_dict.pop("multi_modal_kwargs")
            if metadata_dict:
                attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                attn_metadata = None
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

        return (input_tokens, input_positions, encoder_input_tokens,
                encoder_input_positions, attn_metadata, sampling_metadata,
                lora_requests, lora_mapping, multi_modal_kwargs)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, encoder_input_tokens,
         encoder_input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_kwargs
         ) = \
            self.prepare_input_tensors_encoder_decoder(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            encoder_input_ids=encoder_input_tokens,
            encoder_positions=encoder_input_positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **multi_modal_kwargs,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output

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
        vlm_config = self.vision_language_config

        if vlm_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens / vlm_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))

            if vlm_config is None:
                seq_data = SequenceData([0] * seq_len)
                dummy_multi_modal_data = None
            else:
                seq_data, dummy_multi_modal_data = MULTIMODAL_REGISTRY \
                    .dummy_data_for_profiling(seq_len, model_config, vlm_config)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
                encoder_seq_data=seq_data,
                cross_block_table=None)
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def capture_model(self, _: List[torch.Tensor]) -> None:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_CUDAGRAPH)
