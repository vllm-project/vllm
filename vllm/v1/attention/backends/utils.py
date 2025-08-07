# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import abc
import enum
import functools
from abc import abstractmethod
from dataclasses import dataclass, make_dataclass
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Generic, Optional,
                    TypeVar)

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.utils import cdiv

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionImpl
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
_KV_CACHE_LAYOUT_OVERRIDE = None


@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    num_computed_tokens_cpu: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool = True


@dataclass
class UbatchSlice:
    request_slice: slice
    token_slice: slice


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in 
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return query_start_loc[request_slice.start: request_slice.stop + 1] -\
        query_start_loc[request_slice.start]


def _make_metadata_with_slice(
        ubatch_slice: UbatchSlice,
        attn_metadata: CommonAttentionMetadata) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to 
    the requests included in ubatch_slice
    """

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    query_start_loc = slice_query_start_locs(attn_metadata.query_start_loc,
                                             request_slice)
    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, "
        f"got {len(query_start_loc)}")
    query_start_loc_cpu = slice_query_start_locs(
        attn_metadata.query_start_loc_cpu, request_slice)

    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[
        request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] -
                            query_start_loc_cpu[:-1])).item())

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )


def split_attn_metadata(
    ubatch_slices: list[UbatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the 
    requests for each UbatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(
            _make_metadata_with_slice(ubatch_slice, common_attn_metadata))
    return results


M = TypeVar("M")


class AttentionCGSupport(enum.Enum):
    """ Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    NEVER = 0
    """NO cudagraph support"""
    PURE_DECODE_ONLY = 1
    """Cudagraph supported for pure decode, need to run without
    cudagraph for mixed prefill-decode batches"""
    ALWAYS = 2
    """Cudagraph always supported"""


class AttentionMetadataBuilder(abc.ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention.
    attn_cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[Optional[int]] = None

    @abstractmethod
    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.kv_cache_spec = kv_cache_spec

    @abstractmethod
    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.
        
        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        """
        Can this batch (with given metadata) use CUDA Graphs for attention.
        """
        return False

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(common_prefix_len=0,
                          common_attn_metadata=common_attn_metadata)

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.
        
        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(common_prefix_len=0,
                          common_attn_metadata=common_attn_metadata,
                          fast_build=True)

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
    ) -> bool:
        return False


@functools.lru_cache
def get_kv_cache_layout():
    global _KV_CACHE_LAYOUT_OVERRIDE
    # Override with format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    if cache_layout is None:
        if envs.VLLM_USE_TRTLLM_ATTENTION:
            cache_layout = "HND"
        else:
            cache_layout = get_kv_connector_cache_layout()
    else:
        logger.info_once("`VLLM_KV_CACHE_LAYOUT` environment variable " \
        "detected. Setting KV cache layout to %s.", cache_layout)
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
    return cache_layout


def set_kv_cache_layout(cache_layout: str):
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
    logits_soft_cap: Optional[float]
    sm_scale: float


def get_per_layer_parameters(
        vllm_config: VllmConfig, layer_names: list[str],
        cls_: type['AttentionImpl']) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)

        # Infer hyperparameters from the attention layer
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale

        per_layer_params[key] = PerLayerParameters(window_left,
                                                   logits_soft_cap, sm_scale)

    return per_layer_params


def infer_global_hyperparameters(
        per_layer_params: dict[str, PerLayerParameters]) -> PerLayerParameters:
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

    # trtllm attention doesn't need global hyper params so disable the check
    if not envs.VLLM_USE_TRTLLM_ATTENTION:
        for params in param_sets:
            if params.window_left != global_params.window_left:
                raise ValueError(
                    "Window left is not the same for all layers. " \
                    "One potential fix is to set disable_sliding_window=True")
            assert params == global_params, (
                "FlashInfer backend currently only supports models in which all"
                "layers share the same values "
                "for the following hyperparameters:"
                "`window_left`, `logits_soft_cap`, `sm_scale`.")

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
) -> CommonAttentionMetadata:
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
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size),
        q_seqlens).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block,
                            attn_chunk_size)

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
    seqlens_q_local = \
        np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1),
        attn_chunk_size)[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0))\
        .astype(np.int32)

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1],
                              attn_chunk_size,
                              dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - \
        (rarange * attn_chunk_size + \
            np.repeat(tokens_in_last_block, local_blocks))
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, \
        f"attn_chunk_size {attn_chunk_size} is not " \
        f"divisible by block_size {block_size}"
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
    block_indices= np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch)) \
            + np.expand_dims(block_starts, axis=1)
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(np.arange(actual_batch_size, dtype=np.int32),
                              local_blocks * pages_per_local_batch)
    block_table_local = block_table[batch_indices, block_indices]\
        .view(virtual_batches, -1)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)

    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device=device,
                                               non_blocking=True),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
    )


def subclass_attention_metadata_builder(
    name_prefix: str,
    builder_cls: type[AttentionMetadataBuilder[M]],
    build_preprocess_fn: Callable[[CommonAttentionMetadata],
                                  CommonAttentionMetadata],
) -> type[AttentionMetadataBuilder[M]]:
    """
    Return a new subclass of `builder_cls` whose .build(...) method
    first calls build_preprocess_fn(common_attn_metadata) on the metadata.
    """
    name: str = name_prefix + builder_cls.__name__  # type: ignore

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False):
        return builder_cls.build(self, common_prefix_len,
                                 build_preprocess_fn(common_attn_metadata),
                                 fast_build)

    Wrapped = type(
        name,
        (builder_cls, ),  # inherit from the original
        {
            "build": build,
        })
    return Wrapped  # type: ignore


def subclass_attention_backend(
        name_prefix: str, attention_backend_cls: type[AttentionBackend],
        builder_cls: type[AttentionMetadataBuilder[M]]
) -> type[AttentionBackend]:
    """
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(name, (attention_backend_cls, ),
                {"get_builder_cls": lambda: builder_cls})


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

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

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[first_prefill:] > decode_threshold)
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


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
    # We now want to reorder the batch so that the "decode" requests are at
    # the front and the "prefill" requests are at the back using the least
    # amount of swaps possible. (NOTE for now we loosely use "decode" to mean
    # requests where attention is likely memory-bound and "prefill" to mean
    # requests where attention is likely compute-bound, TODO(lucas): figure out
    # a better naming here)
    decodes = []
    prefills = []
    num_decode_tokens = 0
    num_prefill_tokens = 0

    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        # for now treat 1 scheduled token as "decode" even if its not,
        # we should update this to something like < 8 in the future but
        # currently the TritonMLA._forward_decode only supports
        # num_tokens = 1
        if num_tokens <= decode_threshold:
            decodes.append(i)
            num_decode_tokens += num_tokens
        else:
            prefills.append(i)
            num_prefill_tokens += num_tokens

    # We hope that this is fairly minimal since decodes
    # should be around for a number of iterations so hopefully they are
    # relatively stationary (and new request are generally appended to the
    # persistent batch so already should be at the back)
    # To achieve this we loop over the decodes in descending order and
    # the prefills in ascending order. We swap decodes from the  "back"
    # i.e. past where the last decode should be in the reodorered with
    # prefills from the front of the batch.
    # `decodes` and `prefills` are already in ascending order just based on
    # the above loop
    num_decodes = len(decodes)
    num_prefills = len(prefills)
    modified_batch = False

    for i in range(1, min(num_decodes, num_prefills) + 1):
        # If the decode is at the "back" of the batch, i, we can swap it
        # with the prefill closest to the front of the batch
        decode_idx = decodes[num_decodes - i]
        if decode_idx < num_decodes:
            break

        input_batch.swap_states(prefills[i - 1], decode_idx)
        modified_batch = True

    return modified_batch


KV_SHARING_FAST_PREFILL_METADATA_FIELDS = [
    ('logits_indices_padded', Optional[torch.Tensor], None),
    ('num_logits_indices', int, 0),
]


def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields
    """
    name: str = name_prefix + metadata_cls.__name__  # type: ignore
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls, ))
    return Wrapped


def make_kv_sharing_fast_prefill_attention_metadata(
    metadata_cls: Any, ) -> Any:
    """
    Return a new subclass of `metadata_cls` for fast prefill
    """
    return subclass_attention_metadata(
        name_prefix="KVSharingFastPrefill",
        metadata_cls=metadata_cls,
        fields=KV_SHARING_FAST_PREFILL_METADATA_FIELDS,
    )
