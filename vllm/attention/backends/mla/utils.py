# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple,
                    Type)

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, MLAAttentionImpl,
                                              T)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           get_flash_attn_version,
                                           is_block_tables_empty)
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_fp8_linear_generic, current_platform_fp8_dtype, is_fp8)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_dequantize, scaled_quantize)
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.v1.attention.backends.flash_attn import merge_attn_states
from vllm.vllm_flash_attn import flash_attn_varlen_func

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


class MLACommonBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return MLACommonMetadata

    @staticmethod
    def get_builder_cls() -> Type["MLACommonMetadataBuilder"]:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["MLACommonState"]:
        return MLACommonState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        ops.copy_blocks_mla(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]


class MLACommonState(AttentionState):

    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False

        scheduler_config = runner.scheduler_config
        model_config = runner.model_config
        cache_config = runner.cache_config

        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled

        if self.chunked_prefill_enabled:
            workspace_size = min(
                # Max sure there is enough for 1 full length request or at least
                # 2 pages of cache per request
                max(
                    model_config.max_model_len, 2 *
                    scheduler_config.max_num_seqs * cache_config.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens
                64 * 1024)

            self.chunked_prefill_workspace = torch.empty(
                (workspace_size, model_config.get_head_size()),
                dtype=model_config.dtype,
                device=runner.device,
            )

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._is_graph_capturing = True

        self._graph_slot_mapping = torch.full((max_batch_size, ),
                                              PAD_SLOT_ID,
                                              dtype=torch.long,
                                              device=self.runner.device)
        self._graph_seq_lens = torch.ones(max_batch_size,
                                          dtype=torch.int32,
                                          device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(
            self.runner.graph_block_tables).to(device=self.runner.device)

        self._positions = torch.zeros((max_batch_size, ),
                                      dtype=torch.long,
                                      device=self.runner.device)

        yield

        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables
        del self._positions

    def graph_clone(self, batch_size: int):
        assert self._is_graph_capturing
        return self.__class__(self.runner)

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing

        attn_metadata = self.runner.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=self._graph_slot_mapping[:batch_size],
            seq_lens=None,
            seq_lens_tensor=self._graph_seq_lens[:batch_size],
            max_query_len=1,
            max_decode_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.runner.max_seq_len_to_capture,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self._graph_block_tables[:batch_size],
            input_positions=self._positions[:batch_size],
            head_dim=self.runner.model_config.get_head_size())

        if is_encoder_decoder_model:
            raise NotImplementedError(
                "MLACommonState does not support encoder/decoder yet")

        return attn_metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = {
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
            "input_positions": attn_metadata.decode_metadata.input_positions,
        }
        if is_encoder_decoder_model:
            raise NotImplementedError(
                "MLACommonState does not support encoder/decoder yet")

        return input_buffers

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        input_positions = attn_metadata.input_positions
        num_positions = input_positions.shape[0]
        input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # CUDA graph buffer is padded so only perform a partial copy based on
        # num_positions
        input_buffers["input_positions"][:num_positions].copy_(
            input_positions, non_blocking=True)
        if is_encoder_decoder_model:
            raise NotImplementedError(
                "TritonMLAState does not support encoder/decoder yet")

    def begin_forward(self, model_input):
        return


@dataclass
class MLACommonMetadata(AttentionMetadata):
    """Metadata for MLACommon.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # Smuggle the state to the impl via meta-data, we need this for the
    # `chunked_prefill_workspace` but passing that directly will result in
    # it unnecessarily being broadcasted to all workers.
    attn_state: MLACommonState

    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["MLACommonMetadata"] = None
    _cached_decode_metadata: Optional["MLACommonMetadata"] = None

    num_prefill_tokens: int

    # The dimension of the attention heads
    head_dim: Optional[int] = None

    # For chunked prefill
    chunk_cu_seq_lens: Optional[torch.Tensor] = None
    chunk_seq_starts: Optional[torch.Tensor] = None
    chunk_iter_toks: Optional[List[int]] = None
    chunk_max_seq_lens: Optional[List[int]] = None

    def __post_init__(self):
        supported_head_sizes = MLACommonBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.")

    @property
    def prefill_metadata(self) -> Optional["MLACommonMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])
        input_positions = (None if self.input_positions is None else
                           self.input_positions[:self.num_prefill_tokens])

        self._cached_prefill_metadata = MLACommonMetadata(
            # Required by ModelRunner
            use_cuda_graph=False,  # Not Attention Related
            # Required by Attention Metadata
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            # Required by Attention Metadata (not used)
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            attn_state=self.attn_state,
            input_positions=input_positions,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            head_dim=self.head_dim,
            # MLACommonMetadata Chunk prefill specific
            chunk_cu_seq_lens=self.chunk_cu_seq_lens,
            chunk_seq_starts=self.chunk_seq_starts,
            chunk_iter_toks=self.chunk_iter_toks,
            chunk_max_seq_lens=self.chunk_max_seq_lens,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["MLACommonMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.seq_lens_tensor is not None

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        input_positions = (None if self.input_positions is None else
                           self.input_positions[self.num_prefill_tokens:])

        self._cached_decode_metadata = MLACommonMetadata(
            # Required by ModelRunner
            use_cuda_graph=self.use_cuda_graph,  # Not Attention Related
            # Required by Attention Metadata
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            # Required by Attention Metadata (not used)
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            attn_state=self.attn_state,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=block_tables,
            input_positions=input_positions,
            head_dim=self.head_dim)
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class MLACommonMetadataBuilder(AttentionMetadataBuilder[MLACommonMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.chunked_prefill_enabled = \
            self.runner.scheduler_config.chunked_prefill_enabled
        self.attn_state = self.input_builder.runner.attn_state

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.input_positions: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block, input_positions) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks,
                 inter_data.input_positions):
            self.input_positions.extend(input_positions)
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])

        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        input_positions = async_tensor_h2d(self.input_positions, torch.long,
                                           device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)

        chunk_cu_seq_lens = None
        chunk_seq_starts = None
        chunk_iter_toks = None
        chunk_max_seq_lens = None

        if self.chunked_prefill_enabled and self.num_prefills > 0 \
            and context_lens_tensor is not None:
            chunked_prefill_workspace_size = \
                self.attn_state.chunked_prefill_workspace.shape[0]
            page_size = self.runner.block_size
            seq_chunk_size = chunked_prefill_workspace_size // self.num_prefills

            print(self.attn_state.chunked_prefill_workspace.shape,
                  self.num_prefills, page_size, seq_chunk_size)

            # align seq_chunk_size to page_size by rounding down
            seq_chunk_size = seq_chunk_size - (seq_chunk_size % page_size)
            assert seq_chunk_size > 0
            num_chunks = (context_lens_tensor.max() + seq_chunk_size -
                          1) // seq_chunk_size

            # if `seq_chunk_size = 256`, `num_chunks = 3`, and
            #   `num_prefills = 4`, create a tensor that looks like
            #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
            # Note: we only chunk the current context so we can separate the
            #       causal masked and un-masked portions of the computation
            chunk_seq_starts = \
                torch.arange(num_chunks, device=device, dtype=torch.int32)\
                .unsqueeze(1).expand(-1, self.num_prefills)\
                * seq_chunk_size
            chunk_ends = torch.min(context_lens_tensor[:self.num_prefills]\
                .unsqueeze(0), chunk_seq_starts + seq_chunk_size)
            chunk_seq_lens = (chunk_ends - chunk_seq_starts).clamp(min=0)
            _chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device)\
                .unsqueeze(-1)
            chunk_cu_seq_lens = torch.cat([zero, _chunk_cu_seq_lens], dim=1)
            chunk_max_seq_lens = chunk_seq_lens.max(dim=1).values.tolist()
            chunk_iter_toks = chunk_seq_lens.sum(dim=1).tolist()

        return MLACommonMetadata(
            # Required by ModelRunner
            use_cuda_graph=use_captured_graph,  # Not Attention Related
            # Required by Attention Metadata
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            # Required by Attention Metadata (not used)
            multi_modal_placeholder_index_maps=None,  # Not Attention Related
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            attn_state=self.attn_state,
            input_positions=input_positions,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            head_dim=self.runner.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            chunk_cu_seq_lens=chunk_cu_seq_lens,
            chunk_seq_starts=chunk_seq_starts,
            chunk_iter_toks=chunk_iter_toks,
            chunk_max_seq_lens=chunk_max_seq_lens,
        )


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    Common class for implementing repeated parts

    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    Deepseek's MLA attention works the following way:
    * Use a single latent vector to represent the entire KV cache.
    * The attention "simulates" a multi-head attention, while the compute is
      similar to multi-query attention.
    * The dataflow is as follows,

        * B: batch/sequence length
        * H: hidden size
        * N: number of attention heads
        * Lq: latent dimension for Q
        * Lkv: latent dimension for K/V
        * P: nope dimension, P+R is the actual head_dim in common attention.
        * R: rope dimension, this slide of the head_dim goes through rope.
        * V: V head dim.
        * kv_c: latent/compressed KV
        * q_c: latent/compressed Q

        #
        # Outside the MLA attention backend
        #

        1. The hidden states (B, H) are projected down into cq (B, Lq) and
           kv_c_k_pe (B, Lkv+R).
        2. The kv_c_k_pe is split into kv_c (B, Lkv) and k_pe (B, R). cq
           and kv_c are normalized.

        #
        # Inside the MLA attention backend
        #

        * if prefill:

        3. The q_c is then projected up into the multi-head version.
           * q_c goes from (B, Lq) to (B, N, (P+R)), which is split into q_nope
             (B, N, P) and q_pe (B, N, R).
        4. q_pe, k_pe are then passed through rotary embeddings.
        5. kv_c and k_pe are concatenated and inserted into the cache
        6. The kv_c is then projected up into the multi-head version.
           * kv_c goes from (B, Lkv) to (B, N, (P+V)) which has the nope
             dimensions for K and V, which is split into k_nope (B, N, P)
             and v (B, N, V).
        7. q (B, N, (P+R)) and k (B, N, (P+R)) matrices are assembled from
           q_nope, q_pe, k_nope, k_pe.
        8. Attention is computued with q, k, v.
        9. The attention computation returns (B, N, V), which is projected back
           to (B, H) using out projection.

        * if decode:

        3. Here's the change, we do not perform up the full up projection for
           q_c, and there is no up projection at all for kv_c. This is
           achieved by the technique of "weight absorption". The paper says
           "Fortunately, due to the associative law of matrix multiplication,
           we can absorb WUK into WUQ, and WUV into WO"
           * The q up projection turns (B, Lq) into (B, N, (P+R)), we split it
             into W_UQ (Lq, N, P) and W_QR (Lq, N, R).
           * The kv_c up projection turns (B, Lkv) into (B, N, (P+V)), we split
             it into W_UK (Lkv, N, P) and W_UV (Lkv, N, V).
           * The out projection shape W_O (N*V, H) turns (B, N, V) into (B, H).
           * We can precompute the product of W_UQ and W_UK into
             W_UQ_UK (Lq, N, Lkv), which is possible due to QK^T operation in
             attention.
           * We can precompute the product of W_UV and W_O into
             W_UV_O (N, Lkv, H), which is possible due to V@O as the
             "epilogue" of attention
        4. We still need to compute q_pe (B, N, R) by applying W_QR to q_latent.
        5. q_pe, k_pe are then passed through rotary embeddings.
        6. kv_c and k_pe are concatenated and inserted into the cache
        7. By applying W_UQ_UK to q_latent, we have the new q_nope of shape
           (B, N, Lkv).
        8. q (B, N, (Lkv+R)), k (B, (Lkv+R)) are assembled from q_nope, q_pe,
           kv_a, k_pe. v (B, Lkv) is exactly the same vector as kv_a.
        9. The attention is computed with q, k, v. Note that we just performed
           a MQA attention with (LKv+R) as our head dim.
        10. The KV cache is updated using the new entries k (B, N, (Lkv+R)),
           which included the v and rope values.
        11. The attention computation returns (B, N, Lkv), which is projected
           back to (B, H) using W_UV_O.

    From @tsu-bin's calculation, we only want to use the absorption technique
    for decode. The prefill algorithm should still use the up-projected MHA
    for less flops and memory usage.

    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        self.rotary_emb = rotary_emb
        self.use_yarn_rope = isinstance(rotary_emb,
                                        DeepseekScalingRotaryEmbedding)
        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.vllm_flash_attn_version = get_flash_attn_version()

    def _v_up_proj_and_o_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_UV_O):
                output_parallel = apply_fp8_linear_generic(
                    x.flatten(start_dim=1), self.W_UV_O, self.W_UV_O_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape)
            else:
                output_parallel = torch.matmul(x.flatten(start_dim=1),
                                               self.W_UV_O)
            if self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel
            return output
        else:
            x = torch.einsum("bnl,lnv->bnv", x, self.W_UV)
            return self.o_proj(x.reshape(-1,
                                         self.num_heads * self.v_head_dim))[0]

    def _q_proj_and_k_up_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_Q_UK):
                return apply_fp8_linear_generic(
                    x, self.W_Q_UK, self.W_Q_UK_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape).view(
                        -1, self.num_heads, self.kv_lora_rank)
            return torch.matmul(x, self.W_Q_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)
        else:
            x = torch.matmul(x, self.W_Q)\
                .view(-1, self.num_heads, self.qk_nope_head_dim)
            return torch.einsum("bnp,lnp->bnl", x, self.W_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def is_layer_fp8(layer: LinearBase) -> bool:
            return isinstance(layer.quant_method, Fp8LinearMethod) or\
                (isinstance(layer.quant_method, CompressedTensorsLinearMethod)\
                and isinstance(layer.scheme, CompressedTensorsW8A8Fp8))

        def quantization_scheme_supported(layer: LinearBase) -> bool:
            return isinstance(layer.quant_method, UnquantizedLinearMethod) or \
                is_layer_fp8(layer)

        # TODO(lucas) This is very gross, we need a more wide scale refactor of
        # all the FP8 code with a more standard way of
        # defining schemes/group-shapes, we should also potentially force
        # quant_methods to support a decompress function
        #
        # returns input_group_shape, weight_group_shape
        def get_scale_group_shapes_for_fp8(layer: LinearBase) -> \
            Tuple[Tuple[int, int], Tuple[int, int]]:
            if isinstance(layer.quant_method, Fp8LinearMethod):
                if layer.quant_method.block_quant is not None:
                    weight_block_size = \
                        layer.quant_method.quant_config.weight_block_size
                    # per-token-group (1, X), block-quantized (X, Y)
                    return (1, weight_block_size[-1]), weight_block_size
                else:
                    return (-1, -1), (-1, -1)  # per-tensor, per-tensor
            elif isinstance(layer.quant_method, CompressedTensorsLinearMethod)\
                and isinstance(layer.scheme, CompressedTensorsW8A8Fp8):
                # this is hacky but we always assume the for
                # CompressedTensorsW8A8Fp8 the input is dynamic per-token
                # we ignore if it is static-per-tensor since we are going to
                # requantize after later anyways
                strategy = layer.scheme.strategy
                if strategy == QuantizationStrategy.TENSOR:
                    return (1, -1), (-1, -1)  # per-token, per-tensor
                elif strategy == QuantizationStrategy.CHANNEL:
                    return (1, -1), (-1, 1)  # per-token, per-channel
                else:
                    raise NotImplementedError(
                        f"QuantizationStrategy.{strategy} is not supported for "
                        "fp8 MLA, please run with VLLM_MLA_DISABLE=1")
            else:
                raise NotImplementedError(
                    "Can't determine scale group shapes for "
                    f"{layer.quant_method}, please run with VLLM_MLA_DISABLE=1"
                )

        def get_scales(layer: LinearBase) -> torch.Tensor:
            if hasattr(layer, "weight_scale_inv"):
                return layer.weight_scale_inv
            return layer.weight_scale

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if is_layer_fp8(layer):
                if isinstance(layer.quant_method, \
                    CompressedTensorsLinearMethod) and \
                    isinstance(layer.scheme, CompressedTensorsW8A8Fp8):
                    # NOTE(lucas): note sure why but `CompressedTensorsW8A8Fp8`
                    # seems to store weights as (input, output) instead of
                    # (output, input) so we need to transpose
                    weight = layer.weight.T  # standardize to (output, input)
                else:
                    weight = layer.weight
                _, weight_scale_group_shape = \
                    get_scale_group_shapes_for_fp8(layer)
                scales = get_scales(layer)

                return scaled_dequantize(weight, scales,
                                         weight_scale_group_shape)
            else:
                return layer.weight

        if not (quantization_scheme_supported(self.kv_b_proj) and\
            quantization_scheme_supported(self.q_proj) and\
                quantization_scheme_supported(self.o_proj)):
            raise NotImplementedError(
                "Only FP8 and UnquantizedLinearMethod are supported for MLA"
                ", please run with VLLM_MLA_DISABLE=1")

        weight_dtype = self.kv_b_proj.weight.dtype
        assert self.o_proj.weight.dtype == weight_dtype
        assert self.q_proj.weight.dtype == weight_dtype

        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_proj_weight = get_and_maybe_dequant_weights(self.q_proj).T\
                .view(-1, self.num_heads, self.qk_head_dim)

        # can be W_Q or W_UQ depending q_lora_rank, the former if
        # q_lora_rank is None, the latter otherwise. From the Attention backend
        # perspective though we call these both W_Q and rely on the layer
        # to pass in the correct matrix
        W_Q = q_proj_weight[..., :self.qk_nope_head_dim]
        self.W_QR = q_proj_weight[..., self.qk_nope_head_dim:]\
            .flatten(start_dim=1).contiguous()

        # W_QR is small so for simplicity we dont bother requantizing it
        self.W_QR = self.W_QR.to(act_dtype)

        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            requantization_enabled = not envs.VLLM_MLA_DISABLE_REQUANTIZATION
            if is_fp8(weight_dtype) and requantization_enabled:
                # This assumes it wise to requantize using the same group shapes
                # (i.e. strategy, per-tensor, per-channel, block etc.) that the
                # weights were originally quantized
                requant_input_group_shape, requant_weight_group_shape = \
                    get_scale_group_shapes_for_fp8(self.q_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.kv_b_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.o_proj)
                self.reqaunt_input_group_shape = requant_input_group_shape
                self.reqaunt_weight_group_shape = requant_weight_group_shape

            #
            # Perform matrix-absorption following
            #     https://github.com/flashinfer-ai/flashinfer/pull/551
            # for decode, as a result we end up with absorbed weights for decode
            # and another copy of raw weights for prefill.
            #
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # We absorb `W_UK` into `W_Q` resulting in either W_Q_UK or W_UQ_UK
            # depending q_lora_rank, the former if q_lora_rank is None, the
            # latter otherwise
            # basically if q_lora_rank is none we are absorbing into q_proj
            # instead of UQ
            W_Q_UK = torch.einsum("qnd,lnd -> qnl", W_Q, W_UK)\
                .flatten(start_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_Q_UK, W_Q_UK_scales = scaled_quantize(
                    W_Q_UK,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_Q_UK = W_Q_UK.T.contiguous()
                self.W_Q_UK_scales = W_Q_UK_scales.T.contiguous()
            else:
                self.W_Q_UK = W_Q_UK.to(act_dtype)

            W_O = get_and_maybe_dequant_weights(self.o_proj)\
                .view(-1, self.num_heads, self.v_head_dim)
            W_UV_O = torch.einsum("lnd,hnd -> nlh", W_UV, W_O)\
                .flatten(start_dim=0, end_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_UV_O, W_UV_O_scales = scaled_quantize(
                    W_UV_O,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_UV_O = W_UV_O.T.contiguous()
                self.W_UV_O_scales = W_UV_O_scales.T.contiguous()
            else:
                self.W_UV_O = W_UV_O.to(act_dtype)

            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            if is_fp8(weight_dtype):
                raise NotImplementedError(
                    "Currently fp8 requires matrix absorption")

            self.W_UV = W_UV
            self.W_UK = W_UK
            self.W_Q = W_Q.flatten(start_dim=1)

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
        prefill_metadata = attn_metadata.prefill_metadata

        output = None
        iters = len(prefill_metadata.chunk_iter_toks)
        workspace = prefill_metadata.attn_state.chunked_prefill_workspace

        for i in range(iters):
            chunk_cu_seq_lens = prefill_metadata.chunk_cu_seq_lens[i]
            chunk_seq_starts = prefill_metadata.chunk_seq_starts[i]
            toks = prefill_metadata.chunk_iter_toks[i]
            max_seq_len = prefill_metadata.chunk_max_seq_lens[i]

            ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_tables,
                cu_seq_lens=chunk_cu_seq_lens,
                batch_size=prefill_metadata.num_prefills,
                seq_starts=chunk_seq_starts,
            )

            k_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank].unsqueeze(1)
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(k_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

            # For MLA the v head dim is smaller than qk head dim so we pad
            # out v with 0s to match the qk head dim
            v_padded = torch.nn.functional.pad(v,
                                               [0, q.shape[-1] - v.shape[-1]],
                                               value=0)

            attn_output, attn_softmax_lse = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=chunk_cu_seq_lens,
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.scale,
                causal=False,  # Context is unmasked
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.ones_like(output)
                output_lse_tmp = torch.ones_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    # Optional common flash-attn based prefill
    def _forward_prefill_flash(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_start_loc: torch.Tensor,
        max_query_len: int,
        max_prefill_seq_len: int,
    ) -> torch.Tensor:

        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=seq_start_loc,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.vllm_flash_attn_version,
        )
        attn_output = attn_output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(attn_output)[0]

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, MLACommonMetadata)

        prefill_metadata = attn_metadata.prefill_metadata

        has_context = prefill_metadata.context_lens_tensor is not None \
            and prefill_metadata.context_lens_tensor.max() > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_prefill_seq_len,
            max_seqlen_k=attn_metadata.max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=has_context,
            fa_version=self.vllm_flash_attn_version,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        output = output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        attn_metadata.first_layer = False

        return self.o_proj(output)[0]

    @abstractmethod
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")

        has_decode = attn_metadata.decode_metadata is not None
        has_prefill = attn_metadata.prefill_metadata is not None

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions")

        num_prefill_tokens: int = attn_metadata.num_prefill_tokens

        decode_hs_or_q_c = hidden_states_or_q_c[num_prefill_tokens:]
        decode_k_pe = k_pe[num_prefill_tokens:]
        decode_input_positions = \
            attn_metadata.input_positions[num_prefill_tokens:]

        prefill_hs_or_q_c = hidden_states_or_q_c[:num_prefill_tokens]
        prefill_k_pe = k_pe[:num_prefill_tokens]
        prefill_input_positions = \
            attn_metadata.input_positions[:num_prefill_tokens]
        prefill_k_c_normed = k_c_normed[:num_prefill_tokens]

        if has_decode:
            decode_q_nope = self._q_proj_and_k_up_proj(decode_hs_or_q_c)
            decode_q_pe = torch.matmul(decode_hs_or_q_c, self.W_QR)\
                .view(-1, self.num_heads, self.qk_rope_head_dim)
            decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
                decode_input_positions, decode_q_pe, decode_k_pe)

        if has_prefill:
            prefill_q = self.q_proj(prefill_hs_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
                prefill_input_positions, prefill_q_pe, prefill_k_pe)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        output = torch.empty(attn_metadata.num_prefill_tokens +
                             attn_metadata.num_decode_tokens,
                             self.o_proj.output_size,
                             device=hidden_states_or_q_c.device,
                             dtype=hidden_states_or_q_c.dtype)
        if has_prefill:
            output[:num_prefill_tokens] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata)

        if has_decode:
            output[num_prefill_tokens:] = self._forward_decode(
                decode_q_nope, decode_q_pe, kv_cache, attn_metadata)

        return output
