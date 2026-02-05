# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from dataclasses import dataclass, field, fields, make_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    get_args,
)

import numpy as np
import torch
from typing_extensions import runtime_checkable

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)
KVCacheLayoutType = Literal["NHD", "HND"]
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

PAD_SLOT_ID = -1


def is_valid_kv_cache_layout(value: str) -> bool:
    return value in get_args(KVCacheLayoutType)


@functools.lru_cache
def get_kv_cache_layout():
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    cache_layout: Literal["NHD", "HND"] | None = None
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.info_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        assert is_valid_kv_cache_layout(cache_layout)
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


def set_kv_cache_layout(cache_layout: KVCacheLayoutType):
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout


@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters. Should not be used for
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = False
    # has same params for all layers
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)


def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)

        # Infer hyperparameters from the attention layer
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale
        has_sinks = getattr(impl, "sinks", None) is not None

        per_layer_params[key] = PerLayerParameters(
            window_left, logits_soft_cap, sm_scale, has_sinks
        )

    return per_layer_params


def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters:
    """
    Currently, FlashInfer backend other than trtllm-gen
    only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]

    global_params.has_same_window_lefts = all(
        params.window_left == global_params.window_left for params in param_sets
    )
    global_params.has_same_all_params = all(
        params == global_params for params in param_sets
    )

    return global_params


#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}"
    )
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)

    # Save as a lambda so we can return this for update_block_table
    make_block_table = lambda block_table: block_table[
        batch_indices_torch, block_indices_torch
    ].view(virtual_batches, -1)
    block_table_local = make_block_table(block_table)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
    ), make_block_table


def make_kv_sharing_fast_prefill_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    if common_attn_metadata.max_query_len == 1:
        # All requests are decode (assume 1 token for now)
        # Skip computing fast prefill path
        return common_attn_metadata

    assert common_attn_metadata.logits_indices_padded is not None
    assert common_attn_metadata.num_logits_indices is not None

    logits_indices_padded = common_attn_metadata.logits_indices_padded
    num_logits_indices = common_attn_metadata.num_logits_indices
    # Get rid of CUDAGraph padding, if any
    logits_indices = logits_indices_padded[:num_logits_indices]
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc = common_attn_metadata.query_start_loc
    # Example inputs
    # num_reqs: 3
    # generation_indices:  [14, 18, 19, 27]
    # query_start_loc: [0, 15, 20, 28]
    # seq_lens:        [41, 31, 40]

    # Find how many decode indices belong to each request
    # request_ids: [0, 1, 1, 2]
    request_ids = torch.bucketize(logits_indices, query_start_loc[1:], right=True)

    # Figure out how many tokens are in each request
    # num_decode_tokens: [1, 2, 1]
    num_decode_tokens = torch.bincount(request_ids, minlength=num_reqs)

    # Calculate new query_start_loc with tokens in generation_indices
    # decode_query_start_loc: [0, 1, 3, 4]
    decode_query_start_loc = torch.empty(
        num_reqs + 1, device=query_start_loc.device, dtype=query_start_loc.dtype
    )

    decode_query_start_loc[0] = 0
    decode_query_start_loc[1:] = torch.cumsum(num_decode_tokens, dim=0)
    decode_max_query_len = int(num_decode_tokens.max().item())
    total_num_decode_tokens = int(num_decode_tokens.sum().item())

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=decode_query_start_loc,
        query_start_loc_cpu=decode_query_start_loc.to("cpu", non_blocking=True),
        seq_lens=common_attn_metadata.seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_decode_tokens,
        max_query_len=decode_max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
    )
    return common_attn_metadata


def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_extends: The number of extend requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_extend_tokens: The number of tokens in the extend requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    seq_lens = common_attn_metadata.seq_lens_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    if not torch.any(is_prefill_or_extend):
        return (num_decodes, 0, 0, num_decode_tokens, 0, 0)

    num_prefills_or_extends = num_reqs - num_decodes
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    if not torch.any(is_prefill):
        return (
            num_decodes,
            num_prefills_or_extends,
            0,
            num_decode_tokens,
            num_prefill_or_extend_tokens,
            0,
        )

    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill

    num_prefill_tokens = num_tokens - query_start_loc[first_prefill]
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold and (
        not require_uniform or decode_threshold <= 1
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            assert num_reqs * query_lens[0] == num_tokens, "tokens not padded correctly"
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    """
    Split the prefill requests into chunks such that the total sequence length
    of each chunk is less than or equal to the workspace size.

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests on CPU.
        workspace_size: The maximum workspace size (in tokens) per chunk.
        request_offset: The offset to add to the request indices.
    Returns:
        A list of tuples of (reqs_start, reqs_end) representing chunk boundaries.
    """
    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert torch.all(seq_lens_cpu <= workspace_size).item()

    while i < n:
        start, chunk_total = i, 0
        while i < n and (chunk_total + (s := seq_lens_cpu[i].item())) <= workspace_size:
            chunk_total += s
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    Returns:
        True if the batch was modified, False otherwise.
    """
    # We now want to reorder the batch into decode → extend → prefill order
    # where:
    #   decode: request with num_scheduled_tokens <= decode_threshold
    #   extend: non-decode request with existing context
    #   prefill: non-decode request with no existing context
    # NOTE for now we loosely use "decode" to mean requests where attention is
    #  likely memory-bound and "prefill" to mean requests where attention is
    #  likely compute-bound,
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]

    is_prefill = num_computed_tokens_np == 0
    is_decode = (num_scheduled_tokens_np <= decode_threshold) & (~is_prefill)
    is_extend = (num_scheduled_tokens_np > decode_threshold) & (~is_prefill)

    # Desired order: decode → extend → prefill
    req_regions = np.zeros(is_decode.shape, dtype=np.int32)  # 0 = decode by default
    req_regions[is_extend] = 1
    req_regions[is_prefill] = 2

    num_decodes = int(is_decode.sum())
    num_extends = int(is_extend.sum())

    target_regions = np.zeros(num_reqs, dtype=np.int32)
    target_regions[num_decodes : num_decodes + num_extends] = 1
    target_regions[num_decodes + num_extends :] = 2

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True


def reshape_query_for_spec_decode(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reshapes the query tensor for the specified batch size, so that
    it has shape (batch_size, seq_len, num_heads, head_dim).
    """
    assert query.dim() == 3, f"query must be 3D, got {query.dim()}D"
    total_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    assert total_tokens % batch_size == 0, (
        f"{total_tokens=} is not divisible by {batch_size=}"
    )
    seq_len = total_tokens // batch_size
    return query.view(batch_size, seq_len, num_heads, head_dim)


def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the attention output tensor, so that
    the batch_size and seq_len dimensions are combined.
    """
    if attn_output.dim() == 3:
        # Already in the correct shape
        return attn_output
    assert attn_output.dim() == 4, f"attn_output must be 4D, got {attn_output.dim()}D"
    total_tokens = attn_output.shape[0] * attn_output.shape[1]
    return attn_output.view(total_tokens, attn_output.shape[2], attn_output.shape[3])


def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields
    """
    name: str = name_prefix + metadata_cls.__name__  # type: ignore
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls,))
    return Wrapped


@runtime_checkable
class KVSharingFastPrefillMetadata(Protocol):
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None


def create_fast_prefill_custom_backend(
    prefix: str,
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class FastPrefillAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = (
                make_kv_sharing_fast_prefill_common_attn_metadata(common_attn_metadata)
            )
            metadata = super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

            class KVSharingFastPrefillAttentionMetadata(
                metadata.__class__,  #  type: ignore
                KVSharingFastPrefillMetadata,
            ):
                def __init__(self, metadata, common_attn_metadata):
                    # Shallow copy all fields in metadata cls
                    for _field in fields(metadata.__class__):
                        setattr(self, _field.name, getattr(metadata, _field.name))

                    self.logits_indices_padded = (
                        common_attn_metadata.logits_indices_padded
                    )
                    self.num_logits_indices = common_attn_metadata.num_logits_indices

            return KVSharingFastPrefillAttentionMetadata(metadata, common_attn_metadata)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=FastPrefillAttentionBuilder,
    )

    return attn_backend


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor,
    *,
    device: torch.device,
):
    # Needed for causal_conv1d. Use the CPU query_start_loc to avoid DtoH sync.
    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict = {}  # type: ignore
    batch_ptr = None
    token_chunk_offset_ptr = None
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist)
        token_chunk_offset_ptr[  # type: ignore
            0:mlist_len
        ].copy_(offsetlist)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each dcp rank.
    Only consider dcp now, we can extend the case of cp based on this.
    """
    num_requests = seq_lens.size(0)
    if dcp_rank is None:
        rank_offsets = (
            torch.arange(dcp_size, dtype=torch.int32, device=seq_lens.device)
            .unsqueeze(0)
            .repeat(num_requests, 1)
        )
    else:
        rank_offsets = torch.tensor(
            [[dcp_rank]], dtype=torch.int32, device=seq_lens.device
        )
    seq_lens_tiled = (
        seq_lens.to(torch.int32).unsqueeze(-1).repeat(1, rank_offsets.shape[1])
    )
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    return dcp_local_seq_lens.squeeze(1)


def extend_all_queries_by_1(
    common_attn_metadata: CommonAttentionMetadata,
    arange: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by 1.
    Also all seq lens are increased by 1.
    This is useful e.g. in speculative decoding with draft models, where we
    extend each sequence by 1 token.
    The slot mapping is computed externally, as it requires more information.
    """
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +1, +2, ..., +batch_size]
    new_query_start_loc = cad.query_start_loc + arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + 1,
        # each request is extended by 1 token -> batch_size tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size(),
        # All query lens increase by 1, so max query len increases by 1
        max_query_len=cad.max_query_len + 1,
        max_seq_len=cad.max_seq_len + 1,
    )
    return new_cad


def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
    """
    Get the block table tensor for mamba kernels from the input
    common_attn_metadata.block_table_tensor given different mamba cache modes.

    - "all":   input  (#requests, cdiv(max_model_len, block_size));
               output (#requests, cdiv(max_model_len, block_size)).

    - "none":  input  (#requests, 1 + num_speculative_blocks);
               output (#requests, 1 + num_speculative_blocks).

    - "align": input  (#requests, cdiv(max_model_len, block_size));
               output (#requests, 1 + num_speculative_blocks), which are the last
               1 + num_speculative_blocks of each request.
    """
    if mamba_cache_mode in ("all", "none"):
        return block_table
    else:
        assert isinstance(kv_cache_spec, MambaSpec)
        # NOTE: For 0-length requests in CUDA graph, use a start_index of 0
        # to handle the invalid block table.
        start_indices = torch.clamp(
            (seq_lens - 1) // kv_cache_spec.block_size,
            min=0,
        )
        offsets = torch.arange(
            1 + kv_cache_spec.num_speculative_blocks, device=block_table.device
        )
        indices_to_gather = start_indices.unsqueeze(1) + offsets
        return torch.gather(block_table, 1, indices_to_gather)
