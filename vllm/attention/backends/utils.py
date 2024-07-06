"""Attention backend utils"""
from typing import TYPE_CHECKING, Dict, List, Union

import torch

from vllm.sequence import SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad

# Error string(s) for encoder/decoder
# unsupported attention scenarios
STR_NOT_IMPL_ENC_DEC_ROCM_HIP = ("ROCm/HIP is not currently supported "
                                 "with encoder/decoder models.")

PAD_SLOT_ID = -1

if TYPE_CHECKING:
    from vllm.worker.model_runner import GPUModelRunnerBase


def is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False


def compute_slot_mapping_start_idx(is_prompt, query_len, context_len,
                                   sliding_window, use_v2_block_manager):
    """TBA."""
    start_idx = 0
    if is_prompt and sliding_window is not None:
        assert use_v2_block_manager \
            or context_len == 0, (
            "Prefix caching is currently not supported with "
            "sliding window attention in V1 block manager")
        # When prefill, we use it to not write slots to kv cache
        # to save memory.
        start_idx = max(0, query_len - sliding_window)
    return start_idx


def compute_slot_mapping(is_profile_run, slot_mapping, seq_id, seq_len,
                         context_len, start_idx, block_size, block_tables):
    """TBA."""
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
    block_table = block_tables[seq_id]
    slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
    for i in range(max(start_idx, context_len), seq_len):
        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)


def metadata_builder_add_seq_group_common(
        attn_metadata_builder, seq_group_metadata: SequenceGroupMetadata,
        token_lens: List[int], seq_lens: List[int],
        sliding_seq_lens: List[int], query_lens: List[int],
        context_lens: List[int], curr_sliding_window_blocks: List[int],
        prefix_cache_hit, chunked_prefill_enabled):
    is_prompt = seq_group_metadata.is_prompt
    block_tables = seq_group_metadata.block_tables
    computed_block_nums = seq_group_metadata.computed_block_nums

    for (seq_id, token_len, seq_len, sliding_seq_len, query_len, context_len,
         curr_sliding_window_block) in zip(seq_group_metadata.seq_data.keys(),
                                           token_lens, seq_lens,
                                           sliding_seq_lens, query_lens,
                                           context_lens,
                                           curr_sliding_window_blocks):
        attn_metadata_builder.context_lens.append(context_len)
        if is_prompt:
            attn_metadata_builder.num_prefills += 1
            attn_metadata_builder.num_prefill_tokens += token_len
            attn_metadata_builder.prefill_seq_lens.append(seq_len)
        else:
            assert query_len == 1, (
                "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len))
            attn_metadata_builder.num_decode_tokens += query_len
            attn_metadata_builder.decode_seq_lens.append(sliding_seq_len)

        # Compute block table.
        # TODO(sang): Combine chunked prefill and prefix caching by
        # only allowing multiple of block_size chunk size.
        # NOTE: This only works for oooooooxxx style attention.
        block_table = []
        if prefix_cache_hit:
            block_table = computed_block_nums
        elif ((chunked_prefill_enabled or not is_prompt)
              and block_tables is not None):
            block_table = block_tables[seq_id][-curr_sliding_window_block:]
        attn_metadata_builder.block_tables.append(block_table)

        # Compute slot mapping.
        is_profile_run = is_block_tables_empty(block_tables)
        start_idx = compute_slot_mapping_start_idx(
            is_prompt, query_len, context_len,
            attn_metadata_builder.sliding_window,
            attn_metadata_builder.use_v2_block_manager)
        compute_slot_mapping(is_profile_run,
                             attn_metadata_builder.slot_mapping, seq_id,
                             seq_len, context_len, start_idx,
                             attn_metadata_builder.block_size,
                             seq_group_metadata.block_tables)


def metadata_builder_build_common(attn_metadata_builder, metadata_cls,
                                  runner: "GPUModelRunnerBase", seq_lens,
                                  query_lens, use_captured_graph: bool,
                                  cuda_graph_pad_size: int, batch_size: int):
    device = runner.device

    logits_soft_cap = getattr(runner.model_config.hf_config,
                              "attn_logit_softcapping", None)
    if logits_soft_cap is not None:
        raise ValueError(
            "Please use Flashinfer backend for models with logits_soft_cap"
            " (i.e., Gemma-2). Otherwise, the output might be wrong."
            " Set Flashinfer backend by "
            "export VLLM_ATTENTION_BACKEND=FLASHINFER.")

    max_query_len = max(query_lens)
    max_prefill_seq_len = max(attn_metadata_builder.prefill_seq_lens,
                              default=0)
    max_decode_seq_len = max(attn_metadata_builder.decode_seq_lens, default=0)
    num_decode_tokens = attn_metadata_builder.num_decode_tokens

    if use_captured_graph:
        attn_metadata_builder.slot_mapping.extend([PAD_SLOT_ID] *
                                                  cuda_graph_pad_size)
        attn_metadata_builder.block_tables.extend([] * cuda_graph_pad_size)
        num_decode_tokens = batch_size + cuda_graph_pad_size

        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        input_block_tables = runner.graph_block_tables[:batch_size]
        for i, block_table in enumerate(attn_metadata_builder.block_tables):
            if block_table:
                input_block_tables[i, :len(block_table)] = block_table
        block_tables = torch.tensor(input_block_tables, device=device)
    else:
        max_block_table_len = max(
            len(block_table)
            for block_table in attn_metadata_builder.block_tables)
        block_tables = make_tensor_with_pad(
            attn_metadata_builder.block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=device,
        )
    assert max_query_len > 0, ("query_lens: {}".format(query_lens))

    context_lens_tensor = torch.tensor(attn_metadata_builder.context_lens,
                                       dtype=torch.int,
                                       device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
    query_lens_tensor = torch.tensor(query_lens,
                                     dtype=torch.long,
                                     device=device)
    query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                  dtype=torch.int32,
                                  device=device)
    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=device)
    torch.cumsum(seq_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])
    torch.cumsum(query_lens_tensor,
                 dim=0,
                 dtype=query_start_loc.dtype,
                 out=query_start_loc[1:])

    slot_mapping_tensor = torch.tensor(attn_metadata_builder.slot_mapping,
                                       dtype=torch.long,
                                       device=device)

    return metadata_cls(
        num_prefills=attn_metadata_builder.num_prefills,
        slot_mapping=slot_mapping_tensor,
        num_prefill_tokens=attn_metadata_builder.num_prefill_tokens,
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
