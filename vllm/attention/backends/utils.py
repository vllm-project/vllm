"""Attention backend utils"""
from collections import defaultdict
from contextlib import contextmanager
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, TypeVar, Union

import numpy as np
import torch

from vllm.attention import (AttentionMetadata, AttentionMetadataBuilder,
                            AttentionState)
from vllm.attention.backends.abstract import AttentionType
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm.worker.model_runner_base import ModelRunnerBase

# Error string(s) for encoder/decoder
# unsupported attention scenarios
STR_NOT_IMPL_ENC_DEC_ROCM_HIP = ("ROCm/HIP is not currently supported "
                                 "with encoder/decoder models.")

PAD_SLOT_ID = -1

# Switch to numpy implementation of compute_slot_mapping
# if we have at least this many elements. Could be tuned further.
_COMPUTE_SLOT_MAPPING_NUMPY_NUMEL = 256

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder


def is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    return (isinstance(block_tables, dict)
            and all(value is None for value in block_tables.values()))


def compute_slot_mapping_start_idx(is_prompt: bool, query_len: int,
                                   context_len: int, sliding_window: int):
    """
    Compute the start index of slot mapping.
    """
    start_idx = 0
    if is_prompt and sliding_window is not None:
        start_idx = max(0, query_len - sliding_window)
    return start_idx


def _compute_slot_mapping_python(slot_mapping: List[int],
                                 block_table: List[int], range_start: int,
                                 range_end: int, block_size: int):
    for i in range(range_start, range_end):
        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)


def _compute_slot_mapping_numpy(slot_mapping: List[int],
                                block_table: List[int], range_start: int,
                                range_end: int, block_size: int):
    block_table_array = np.array(block_table)
    idx = np.arange(range_start, range_end)
    block_offset = idx % block_size
    idx //= block_size
    seq_slot_mapping_array = block_table_array[idx]
    seq_slot_mapping_array *= block_size
    seq_slot_mapping_array += block_offset
    slot_mapping.extend(seq_slot_mapping_array)


def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, List[int]]):
    """
    Compute slot mapping.
    """
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        return

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    padding_mask_len = max(0, start_idx - context_len)
    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)

    range_start = max(start_idx, context_len)
    range_end = seq_len
    numel = range_end - range_start
    block_table = block_tables[seq_id]

    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
        _compute_slot_mapping_python(slot_mapping, block_table, range_start,
                                     range_end, block_size)
    else:
        _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
                                    range_end, block_size)


TAttentionMetadata = TypeVar("TAttentionMetadata", bound='AttentionMetadata')


class CommonMetadataBuilder(AttentionMetadataBuilder[TAttentionMetadata]):

    _metadata_cls: Type[TAttentionMetadata]

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
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
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return self._metadata_cls(  # type: ignore
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class CommonAttentionState(AttentionState):

    def __init__(self, runner: "ModelRunnerBase"):
        self.runner = runner
        self._is_graph_capturing = False

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
        yield
        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables

    def graph_clone(self, batch_size: int) -> "CommonAttentionState":
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
            multi_modal_placeholder_index_maps=None,
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
            use_cuda_graph=True,
        )
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in\
                ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or " \
                f"'FLASH_ATTN', but "\
                f"got '{self.runner.attn_backend.get_name()}'"
            self._update_captured_metadata_for_enc_dec_model(
                batch_size=batch_size, attn_metadata=attn_metadata)

        return attn_metadata

    def get_graph_input_buffers(
            self,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> Dict[str, Any]:
        input_buffers = {
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in\
                ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or "\
                f"'FLASH_ATTN', but "\
                f"got '{self.runner.attn_backend.get_name()}'"
            self._add_additonal_input_buffers_for_enc_dec_model(
                attn_metadata=attn_metadata, input_buffers=input_buffers)
        return input_buffers

    def prepare_graph_input_buffers(
            self,
            input_buffers,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> None:
        input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in\
                ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or "\
                f"'FLASH_ATTN', but "\
                f"got '{self.runner.attn_backend.get_name()}'"
            self._prepare_input_buffers_for_enc_dec_model(
                attn_metadata, input_buffers)

    def begin_forward(self, model_input) -> None:
        return

    def _update_captured_metadata_for_enc_dec_model(self, batch_size: int,
                                                    attn_metadata):
        """
        Updates the attention metadata parameters for CUDA graph capture in an
        encoder-decoder model.

        This method modifies attention-related tensors and metadata required
        for CUDA graph capture in encoder-decoder models. Specifically, it
        updates the cross-attention and encoder sequence tensors in the 
        AttentionMetadata object.
        """
        # During decode phase the cross_slot_mapping will be empty. Hence set
        # an empty tensor for CUDA Graph capture.
        attn_metadata.cross_slot_mapping = torch.tensor(
            [], dtype=torch.int).cuda()
        attn_metadata.cross_block_tables = torch.full(
            (batch_size, self.runner.get_max_block_per_batch()),
            1,
            dtype=torch.int).cuda()
        attn_metadata.encoder_seq_lens = torch.full((batch_size, ),
                                                    1,
                                                    dtype=torch.int).cuda()
        attn_metadata.encoder_seq_lens_tensor = torch.full(
            (batch_size, ), 1, dtype=torch.int).cuda()
        attn_metadata.max_encoder_seq_len = self.runner.max_seq_len_to_capture
        attn_metadata.num_encoder_tokens = 0

    def _add_additonal_input_buffers_for_enc_dec_model(
            self, attn_metadata, input_buffers: Dict[str, Any]):
        """
        Saves additional input buffers specific to the encoder-decoder model
        from the attention metadata.

        This method extracts and stores encoder-decoder related input buffers
        from the `attn_metadata` into the `input_buffers` dictionary. The
        buffers include encoder sequence lengths, cross-slot mappings, and
        cross-block tables, which are essential for the encoder-decoder model
        during CUDA graph replay.
        """
        input_buffers["encoder_seq_lens_tensor"] = (
            attn_metadata.decode_metadata.encoder_seq_lens_tensor)
        input_buffers["cross_slot_mapping"] = (
            attn_metadata.decode_metadata.cross_slot_mapping)
        input_buffers["cross_block_tables"] = (
            attn_metadata.decode_metadata.cross_block_tables)

    def _prepare_input_buffers_for_enc_dec_model(self, attn_metadata,
                                                 input_buffers: Dict[str,
                                                                     Any]):
        """
        Populates input buffers with data from the encoder-decoder model's
        attention metadata.

        This method fills the input buffers with encoder-decoder specific
        tensors. It copies data from the `attn_metadata` and keyword arguments
        (`kwargs`) into corresponding buffers in the `input_buffers` dictionary.
        The copied data includes attention-related metadata as well as input 
        IDs and positional information for the encoder.
        """
        input_buffers["encoder_seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.encoder_seq_lens_tensor,
            non_blocking=True)
        input_buffers["cross_slot_mapping"].copy_(
            attn_metadata.decode_metadata.cross_slot_mapping,
            non_blocking=True)
        input_buffers["cross_block_tables"].copy_(
            attn_metadata.decode_metadata.cross_block_tables,
            non_blocking=True)


def is_all_encoder_attn_metadata_set(attn_metadata):
    '''
    All attention metadata required for encoder attention is set.
    '''
    return ((attn_metadata.encoder_seq_lens is not None)
            and (attn_metadata.encoder_seq_lens_tensor is not None)
            and (attn_metadata.max_encoder_seq_len is not None))


def is_all_cross_attn_metadata_set(attn_metadata):
    '''
    All attention metadata required for enc/dec cross-attention is set.

    Superset of encoder attention required metadata.
    '''
    return (attn_metadata.is_all_encoder_attn_metadata_set
            and (attn_metadata.cross_slot_mapping is not None)
            and (attn_metadata.cross_block_tables is not None))


def get_seq_len_block_table_args(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    '''
    The particular choice of sequence-length- and block-table-related
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths & 
                                  cross-attn block-tables fields
    Encoder attn -> select encoder sequence lengths fields & no block tables
    
    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * is_prompt: True if prefill, False otherwise
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:

    * Appropriate sequence-lengths tensor
    * Appropriate max sequence-length scalar
    * Appropriate block tables (or None)
    '''

    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_lens_tensor, max_seq_len,
                attn_metadata.block_tables)
    elif attn_type == AttentionType.ENCODER_DECODER:
        # Enc/dec cross-attention KVs match encoder sequence length;
        # cross-attention utilizes special "cross" block tables
        return (attn_metadata.encoder_seq_lens_tensor,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.cross_block_tables)
    elif attn_type == AttentionType.ENCODER:
        # No block tables associated with encoder attention
        return (attn_metadata.encoder_seq_lens_tensor,
                attn_metadata.max_encoder_seq_len, None)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def get_num_prefill_decode_query_kv_tokens(
    attn_metadata,
    attn_type: str,
) -> Tuple[int, int, int]:
    """
    Calculate the number of prefill and decode tokens for query, key/value
    based on the attention metadata and the specified attention type.

    Args:
        attn_metadata (FlashAttentionMetadata): Attention Metadata object.
        attn_type (AttentionType): The type of attention being used.
    Returns:
        Tuple[int, int, int]: A tuple containing three integers:
            - The number of prefill query tokens.
            - The number of prefill key/value tokens.
            - The number of decode query tokens.

    Raises:
        AssertionError: If the number of encoder tokens in `attn_metadata` 
        is `None` when required for the calculations.
    """
    num_prefill_query_tokens = 0
    num_decode_query_tokens = 0
    num_prefill_kv_tokens = 0
    if attn_type == AttentionType.ENCODER:
        # Encoder attention is only invoked during prefill phase.
        # The same input servers a both query and key.
        assert attn_metadata.num_encoder_tokens is not None
        num_prefill_query_tokens = attn_metadata.num_encoder_tokens
        num_prefill_kv_tokens = attn_metadata.num_encoder_tokens
        num_decode_query_tokens = 0
    elif attn_type == AttentionType.ENCODER_DECODER:
        assert attn_metadata.num_encoder_tokens is not None
        num_prefill_query_tokens = attn_metadata.num_prefill_tokens
        # The key is the encoder/cross-attention.
        num_prefill_kv_tokens = attn_metadata.num_encoder_tokens
        num_decode_query_tokens = attn_metadata.num_decode_tokens
    else:  # attn_type == AttentionType.DECODER or
        # attn_type == AttentionType.ENCODER_ONLY
        num_prefill_query_tokens = attn_metadata.num_prefill_tokens
        num_prefill_kv_tokens = attn_metadata.num_prefill_tokens
        num_decode_query_tokens = attn_metadata.num_decode_tokens

    return (num_prefill_query_tokens, num_prefill_kv_tokens,
            num_decode_query_tokens)
