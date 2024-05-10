import time
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import (AttentionMetadata, AttentionMetadataPerStage,
                            get_attn_backend)
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.communication_op import graph_capture_mode
from vllm.distributed.device_communicators import custom_all_reduce
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams
from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import (CudaMemoryProfiler, get_kv_cache_torch_dtype, is_hip,
                        is_pin_memory_available, make_tensor_with_pad)

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]


class PreparePromptMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions: List[int]
    attn_metadata: Optional[AttentionMetadataPerStage]
    seq_lens: List[int]
    query_lens: List[int]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    multi_modal_input: Optional[torch.Tensor]
    slot_mapping: List[int]

    @classmethod
    def empty(cls):
        return PreparePromptMetadata(
            input_tokens=[],
            input_positions=[],
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            multi_modal_input=None,
            slot_mapping=[],
        )


class PrepareDecodeMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions: List[int]
    attn_metadata: Optional[AttentionMetadata]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    slot_mapping: List[int]

    @classmethod
    def empty(cls):
        return PrepareDecodeMetadata(
            input_tokens=[],
            input_positions=[],
            attn_metadata=None,
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            slot_mapping=[],
        )


# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


class ModelRunner:

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
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.vision_language_config = vision_language_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)
        self.attn_backend = get_attn_backend(self.model_config.dtype)

        # Lazy initialization
        self.model: torch.nn.Module  # Set after load_model
        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                lora_config=self.lora_config,
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
            )

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules"
                           ) and self.model.supported_lora_modules, (
                               "Model does not support LoRA")
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently scaled KV cache is only enabled on ROCm
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")
        elif self.model_config.quantization_param_path is not None:
            logger.warning("KV cache scaling factors provided, "
                           "but the KV cache data type is not FP8. "
                           "KV cache scaling factors will not be used.")

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PreparePromptMetadata:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        multi_modal_input_list: List[torch.Tensor] = []

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            context_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            seq_len = min(seq_data.get_len(), context_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[context_len:seq_len]
            seq_lens.append(seq_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                context_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[context_len:]
                prefix_block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.block_tables[seq_id]
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert context_len == 0

            # actual prompt lens
            context_lens.append(context_len)
            query_lens.append(seq_len - context_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(context_len, seq_len)))
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (seq_len - context_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (seq_len - context_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]

            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert context_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(context_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_query_len = max(query_lens)
        max_seq_len = max(seq_lens)
        assert max_query_len > 0

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        subquery_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        if self.attn_backend.get_name() == "flashinfer":
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=True,
                use_cuda_graph=False,
                seq_start_loc=seq_start_loc,
                max_seq_len=max_seq_len,
                block_tables=block_tables)
        else:
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=True,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                subquery_start_loc=subquery_start_loc,
                seq_start_loc=seq_start_loc,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=False,
            )

        return PreparePromptMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping=slot_mapping,
        )

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PrepareDecodeMetadata:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

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
            return PrepareDecodeMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
                lora_index_mapping.append(lora_id)
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

                paged_kv_indices.extend(block_table)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(block_table))
                last_page_len = seq_data.get_len() % self.block_size
                if last_page_len == 0:
                    last_page_len = self.block_size
                paged_kv_last_page_len.append(last_page_len)

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        # For decoding requests, batch_size == input_tokens.
        batch_size = len(input_tokens)
        max_seq_len = max(seq_lens)
        use_captured_graph = (not self.model_config.enforce_eager
                              and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
                              and max_seq_len <= self.max_seq_len_to_capture)
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

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            assert seq_lens_tensor.shape[0] == len(input_tokens)
            assert seq_lens_tensor.shape[0] == len(input_positions)
            assert seq_lens_tensor.shape[0] == len(slot_mapping)

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

        if self.attn_backend.get_name() == "flashinfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            paged_kv_indptr = torch.tensor(paged_kv_indptr,
                                           dtype=torch.int,
                                           device=self.device)
            paged_kv_indices = torch.tensor(paged_kv_indices,
                                            dtype=torch.int,
                                            device=self.device)
            paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len,
                                                  dtype=torch.int,
                                                  device=self.device)
            kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                      self.model_config.dtype)

            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=False,
                use_cuda_graph=False,
                workspace_buffer=self.flashinfer_workspace_buffer,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config),
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_dim=self.model_config.get_head_size(),
                page_size=self.block_size,
                data_type=kv_cache_dtype)
        else:
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=False,
                seq_lens=None,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=None,
                max_seq_len=max_seq_len,
                subquery_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=block_tables,
                use_cuda_graph=use_captured_graph,
            )
        return PrepareDecodeMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            slot_mapping=slot_mapping,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, torch.Tensor]:
        if self.is_driver_worker:
            prefill_reqs = []
            decode_reqs = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_reqs.append(seq_group_meta)
                else:
                    decode_reqs.append(seq_group_meta)

            # Prepare input tensors.
            (
                input_tokens,
                input_positions,
                prefill_attn_metadata,
                seq_lens,
                query_lens,
                lora_index_mapping,
                lora_prompt_mapping,
                lora_requests,
                multi_modal_input,
                slot_mapping,
            ) = self._prepare_prompt(prefill_reqs)
            (
                decode_input_tokens,
                decode_input_positions,
                decode_attn_metadata,
                decode_lora_index_mapping,
                decode_lora_prompt_mapping,
                decode_lora_requests,
                decode_slot_mapping,
            ) = self._prepare_decode(decode_reqs)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            if not self.scheduler_config.chunked_prefill_enabled:
                assert (len(prefill_reqs) and len(decode_reqs)) == 0

            num_prefills = len(seq_lens)
            num_prefill_tokens = len(input_tokens)
            num_decode_tokens = len(decode_input_tokens)

            # Coalesce tensors. Note that attn_metadata is currently not
            # coalesced for simplicity.
            input_tokens.extend(decode_input_tokens)
            input_positions.extend(decode_input_positions)
            slot_mapping.extend(decode_slot_mapping)
            lora_index_mapping.extend(decode_lora_index_mapping)
            lora_prompt_mapping.extend(decode_lora_prompt_mapping)
            lora_requests.update(decode_lora_requests)

            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device=self.device)
            input_positions = torch.tensor(input_positions,
                                           dtype=torch.long,
                                           device=self.device)
            slot_mapping = torch.tensor(slot_mapping,
                                        dtype=torch.long,
                                        device=self.device)

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None

            # Broadcast the metadata.
            # If batch contains both prefill and decode, it sends 2 broadcasts.
            # If it only contains 1 type, it triggers a single broadcast.
            if (prefill_attn_metadata is not None
                    and decode_attn_metadata is not None):
                batch_type = BatchType.MIXED
            elif prefill_attn_metadata is not None:
                batch_type = BatchType.PREFILL
            else:
                batch_type = BatchType.DECODE

            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_input": multi_modal_input,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
                "batch_type": batch_type,
            }
            if prefill_attn_metadata is not None:
                metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
            else:
                assert decode_attn_metadata is not None
                metadata_dict.update(decode_attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)

            # Broadcast decode attn metadata for mixed batch type.
            # The additional broadcast costs 300us overhead on 4 A10 GPUs.
            # We can potentially reduce the overhead by coelescing tensors.
            if batch_type == BatchType.MIXED:
                assert decode_attn_metadata is not None
                metadata_dict = decode_attn_metadata.asdict_zerocopy()
                broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            slot_mapping = metadata_dict.pop("slot_mapping")
            num_prefills = metadata_dict.pop("num_prefills")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            num_prefill_tokens = metadata_dict.pop("num_prefill_tokens")
            num_decode_tokens = metadata_dict.pop("num_decode_tokens")
            batch_type = metadata_dict.pop("batch_type")

            # Create an attention metadata.
            prefill_attn_metadata = None
            decode_attn_metadata = None
            if batch_type == BatchType.PREFILL or batch_type == BatchType.MIXED:
                prefill_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

            # if it is a mixed batch, decode attn_metadata is broadcasted
            # separately.
            if batch_type == BatchType.MIXED:
                metadata_dict = broadcast_tensor_dict(src=0)
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_attn_metadata,
            decode_metadata=decode_attn_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
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
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})
        hidden_states = model_executable(**execute_model_kwargs)

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
        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_loras()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_loras()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
        # kernel, pynccl, and PyTorch NCCL. When using CUDA graph, we use
        # either custom all-reduce kernel or pynccl. When not using CUDA
        # graph, we use either custom all-reduce kernel or PyTorch NCCL.
        # We always prioritize using custom all-reduce kernel but fall back
        # to PyTorch or pynccl if it is disabled or not supported.
        with custom_all_reduce.capture():
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                # Create dummy attn_metadata.
                decode_metadata = self.attn_backend.make_metadata(
                    is_prompt=False,
                    seq_lens=None,
                    seq_lens_tensor=seq_lens[:batch_size],
                    max_query_len=None,
                    max_seq_len=self.max_seq_len_to_capture,
                    subquery_start_loc=None,
                    seq_start_loc=None,
                    context_lens_tensor=None,
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                )
                attn_metadata = AttentionMetadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size,
                    slot_mapping=slot_mapping[:batch_size],
                    prefill_metadata=None,
                    decode_metadata=decode_metadata,
                    kv_cache_dtype=self.kv_cache_dtype,
                )

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size,
                        [0] * batch_size,
                    )
                    self.set_active_loras(set(), lora_mapping)

                graph_runner = CUDAGraphRunner(self.model)
                graph_runner.capture(
                    input_tokens[:batch_size],
                    input_positions[:batch_size],
                    kv_caches,
                    attn_metadata,
                    memory_pool=self.graph_memory_pool,
                )
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the pynccl communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        # TODO(youkaichao): when we get enough user feedback that pynccl is
        # more stable than cupy, we can remove this, e.g. in v0.4.1.
        self.graph_runners.clear()
        self.pynccl_backend = None

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool,
        **kwargs,
    ) -> None:
        assert self._graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        with graph_capture_mode():
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        # NOTE(woosuk): Python 3.8 does not support multi-line with statements.
        # https://stackoverflow.com/questions/31039022/python-multi-line-with-statement
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool):  # noqa: SIM117
            with graph_capture_mode():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    kv_caches,
                    attn_metadata,
                    **kwargs,
                )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _prepare_fake_inputs(
        seq_len: int, vision_language_config: Optional[VisionLanguageConfig]):
    """Prepare fake inputs for profile run."""
    if vision_language_config:
        prompt_tokens = [
            vision_language_config.image_token_id
        ] * vision_language_config.image_feature_size + [0] * (
            seq_len - vision_language_config.image_feature_size)
        fake_image_input = MultiModalData(
            type=MultiModalData.Type.IMAGE,
            data=torch.zeros(vision_language_config.image_input_shape,
                             dtype=torch.float16))
    else:
        prompt_tokens = [0] * seq_len
        fake_image_input = None
    return SequenceData(prompt_tokens), fake_image_input
