import gc
import time
import warnings
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.communication_op import graph_capture
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import (CudaMemoryProfiler, get_kv_cache_torch_dtype, is_hip,
                        is_pin_memory_available, make_tensor_with_pad)
from vllm.worker.model_runner import (
    _PAD_SLOT_ID, LORA_WARMUP_RANK, _BATCH_SIZE_ALIGNMENT,
    _BATCH_SIZES_TO_CAPTURE, _NUM_WARMUP_ITERS, ModelInput, ModelRunner,
    _is_block_tables_empty, _get_graph_batch_size, CUDAGraphRunner,
    _is_encoder_decoder_model)
from vllm.core.block.utils import (STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
                                   STR_NOT_IMPL_ENC_DEC_SWA)
from vllm.attention.backends.utils import STR

logger = init_logger(__name__)

# Error message if EncoderDecoderModelRunner is used with
# a non-encoder/decoder model (i.e. decoder-only)
STR_ENCDECMR_ENCODER_DECODER_REQUIRED = "Only encoder/decoder models may be executed using EncoderDecoderModelRunner"

# Error message if EncoderDecoderModelRunner is used with
# CUDAGraph
STR_ENCDECMR_CUDAGRAPH_UNSUPPORTED = "Currently CUDAGraph is not supported for encoder/decoder models"


class EncoderDecoderModelInput(ModelInput):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_mapping: Optional[LoRAMapping]
    lora_requests: Set[LoRARequest]
    multi_modal_kwargs: Dict[str, torch.Tensor]
    slot_mapping: torch.Tensor
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefills: int

    @classmethod
    def empty(cls, device):
        return ModelInput(
            input_tokens=torch.empty(0, device=device),
            input_positions=torch.empty(0, device=device),
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            lora_mapping=None,
            lora_requests=set(),
            multi_modal_kwargs={},
            slot_mapping=torch.empty(0, device=device),
            num_prefill_tokens=0,
            num_decode_tokens=0,
            num_prefills=0,
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

        if not self._is_encoder_decoder_model():
            # Fail if EncoderDecoderModelRunner is constructed for a
            # non-encoder/decoder model i.e. decoder-only
            raise AttributeError(STR_ENCDECMR_ENCODER_DECODER_REQUIRED)

        if self.scheduler_config.chunked_prefill_enabled:
            raise NotImplementedError()

    def _prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> ModelInput:
        """Prepare the model input based on a given sequence group.

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
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:
                computed_block_nums = seq_group_metadata.computed_block_nums
                if (self.scheduler_config is not None
                        and self.scheduler_config.chunked_prefill_enabled
                        and not (computed_block_nums is None
                                 or computed_block_nums == [])):
                    raise RuntimeError(
                        "chunked prefill cannot be used with prefix caching "
                        "now.")

                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                else:
                    # get_num_computed_tokens is incorrect for spec decoding.
                    # So, we should have a special logic here.
                    # TODO(sang): Fix it.
                    context_len = seq_data.get_len() - 1

                seq_len = min(
                    seq_data.get_len(),
                    context_len + seq_group_metadata.token_chunk_size)
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
                                    and self.sliding_window is None
                                    and is_prompt)

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
                if (self.sliding_window is not None and not is_prompt):
                    curr_sliding_window_blocks = sliding_window_blocks
                    if self.scheduler_config.use_v2_block_manager:
                        # number of elements in last block
                        suff_len = seq_len % self.block_size
                        sliding_seq_len = min(
                            seq_len, block_aligned_sliding_window + suff_len)
                        if suff_len > 0:
                            curr_sliding_window_blocks += 1
                    else:
                        sliding_seq_len = min(seq_len, self.sliding_window)
                    sliding_context_len = sliding_seq_len - 1

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
                        block_table = seq_group_metadata.block_tables[seq_id]
                    else:
                        block_table = computed_block_nums
                elif (self.scheduler_config.chunked_prefill_enabled
                      or not is_prompt):
                    if seq_group_metadata.block_tables is not None:
                        # chunked prefill or decode
                        block_table = seq_group_metadata.block_tables[seq_id]
                        if curr_sliding_window_blocks is not None:
                            block_table = block_table[
                                -curr_sliding_window_blocks:]
                        if self.attn_backend.get_name() == "flashinfer":
                            paged_kv_indices.extend(block_table)
                            paged_kv_indptr.append(paged_kv_indptr[-1] +
                                                   len(block_table))
                            last_page_len = seq_data.get_len(
                            ) % self.block_size
                            if last_page_len == 0:
                                last_page_len = self.block_size
                            paged_kv_last_page_len.append(last_page_len)
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

                if is_prompt:
                    assert len(seq_ids) == 1
                    num_prefills += 1
                    num_prefill_tokens += len(tokens)
                    decode_only = False
                    prefill_seq_lens.append(seq_len)
                else:
                    assert query_len == 1, (
                        "seq_len: {}, context_len: {}, query_len: {}".format(
                            seq_len, context_len, query_len))
                    num_decode_tokens += query_len
                    decode_seq_lens.append(sliding_seq_len)

                if lora_id > 0:
                    lora_requests.add(seq_group_metadata.lora_request)

                lora_index_mapping += [lora_id] * query_len
                lora_prompt_mapping.extend(
                    [lora_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     is not None else 1))

                mm_data = seq_group_metadata.multi_modal_data
                if mm_data is not None:
                    # Process multi-modal data
                    if self.multi_modal_input_processor is None:
                        raise ValueError(
                            "Multi-modal inputs are only supported by "
                            "vision language models.")

                    mm_kwargs = self.multi_modal_input_processor(mm_data)
                    for k, v in mm_kwargs.items():
                        multi_modal_kwargs_list[k].append(v)

                if _is_block_tables_empty(seq_group_metadata.block_tables):
                    # During memory profiling, the block tables are not
                    # initialized yet. In this case, we just use a dummy
                    # slot mapping.
                    # In embeddings, the block tables are {seq_id: None}.
                    slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                    continue

                # Compute the slot mapping.
                block_table = seq_group_metadata.block_tables[seq_id]

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

        batch_size = len(input_tokens)
        max_query_len = max(query_lens)
        max_prefill_seq_len = max(prefill_seq_lens, default=0)
        max_decode_seq_len = max(decode_seq_lens, default=0)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        use_captured_graph = (
            decode_only and not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_decode_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                seq_lens.append(1)
                block_tables.append([])
                lora_index_mapping.append(0)
            batch_size = graph_batch_size
            num_decode_tokens = batch_size

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

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

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device=self.device)
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)

        if self.attn_backend.get_name() == "flashinfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr,
                                                  dtype=torch.int,
                                                  device=self.device)
            paged_kv_indices_tensor = torch.tensor(paged_kv_indices,
                                                   dtype=torch.int,
                                                   device=self.device)
            paged_kv_last_page_len_tensor = torch.tensor(
                paged_kv_last_page_len, dtype=torch.int, device=self.device)
            kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                      self.model_config.dtype)
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                use_cuda_graph=False,
                max_prefill_seq_len=max_prefill_seq_len,
                block_tables=block_tables,
                workspace_buffer=self.flashinfer_workspace_buffer,
                paged_kv_indptr=paged_kv_indptr_tensor,
                paged_kv_indices=paged_kv_indices_tensor,
                paged_kv_last_page_len=paged_kv_last_page_len_tensor,
                num_qo_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config),
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_dim=self.model_config.get_head_size(),
                page_size=16,
                seq_start_loc=seq_start_loc,
                data_type=kv_cache_dtype)
        else:
            context_lens_tensor = torch.tensor(context_lens,
                                               dtype=torch.int,
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

            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max_query_len,
                max_prefill_seq_len=max_prefill_seq_len,
                max_decode_seq_len=max_decode_seq_len,
                query_start_loc=query_start_loc,
                seq_start_loc=seq_start_loc,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=use_captured_graph,
            )

        if self.lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        multi_modal_kwargs = {
            k: torch.cat(v, dim=0).to(self.device)
            for k, v in multi_modal_kwargs_list.items()
        }

        return ModelInput(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, Dict[str, torch.Tensor]]:
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
            }
            if attn_metadata:
                metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
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

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_kwargs)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_kwargs
         ) = self.prepare_input_tensors(seq_group_metadata_list)

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
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
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
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def capture_model(self, _: List[torch.Tensor]) -> None:
        raise NotImplementedError(STR_ENCDECMR_CUDAGRAPH_UNSUPPORTED)
