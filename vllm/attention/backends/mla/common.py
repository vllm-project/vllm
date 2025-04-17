# SPDX-License-Identifier: Apache-2.0
"""
This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach, we generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is "small", is near 1)
and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is "large").

NOTE what we deem small and large is currently determined by if its labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is example of both paths assuming batchsize = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "_forward_prefill"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      spda_o shape [Sq, N, V]
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
) 
return spda_o @ W_O

NOTE: in the actual code, 
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "_forward_decode"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      spda_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
spda_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", spda_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ self.num_heads @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are 
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to 
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`

To mitigate this, we chunk the computation of attention with respect to the 
current context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a 
fixed workspace size.

The chunked prefill approach is as follows:

MCC        Max chunk of context to process per iter, computed dynamically, 
           used to bound the memory usage

q_c        = h_t @ W_DQ
q_nope     = (q_c @ W_UQ).view(Sq, N, P)
q_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c   = h_t @ W_DKV
new_k_pe   = RoPE(h_t @ W_KR)
new_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)
new_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)

// MHA between queries and new KV
//     with QK headdim = P + R
//           V headdim = V
//    curr_o   shape [Sq, N, V]
//    curr_lse shape [N, Sq], this is just order FA returns
curr_o, curr_lse = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    new_v,
    casual=True,
    return_softmax_lse=True
) 

// Compute attention with the already existing context
for chunk_idx in range(cdiv(C, MCC)):
    chunk_start  = chunk_idx * MCC
    chunk_end    = min(chunk_start + MCC, C)
    Sc           = chunk_end - chunk_start
    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]
    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]
    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)
    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)

    chunk_o, chunk_lse = scaled_dot_product_attention(
        torch.cat([q_nope, q_pe], dim=-1),
        torch.cat([cache_k_nope_chunk,
                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],
                   dim=-1),
        cache_v_chunk,
        casual=False,
        return_softmax_lse=True
    )

    curr_o, curr_lse = merge_attn_states(
        suffix_output=curr_o,
        suffix_lse=curr_lse,
        prefix_output=chunk_o,
        prefix_lse=chunk_lse,
    )

return curr_o @ W_O
"""

import functools
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar)

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, MLAAttentionImpl)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils import async_tensor_h2d, cdiv, make_tensor_with_pad, round_down
from vllm.vllm_flash_attn.fa_utils import get_flash_attn_version

if HAS_TRITON:
    from vllm.attention.ops.triton_flash_attention import triton_attention
else:
    triton_attention = None

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    is_vllm_fa = True
except ImportError:
    is_vllm_fa = False
    try:
        # For rocm use upstream flash attention
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        flash_attn_varlen_func = None

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

is_hip = current_platform.is_rocm()


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


T = TypeVar("T", bound="MLACommonMetadata")


class MLACommonState(AttentionState, Generic[T]):

    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False

        scheduler_config = runner.scheduler_config
        self.model_config = runner.model_config
        cache_config = runner.cache_config

        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.enable_prefix_caching = cache_config.enable_prefix_caching

        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            self.context_chunk_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(
                    8 * self.model_config.max_model_len, 4 *
                    scheduler_config.max_num_seqs * cache_config.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.context_chunk_workspace_size >= \
                scheduler_config.max_num_seqs * cache_config.block_size

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
            self,
            batch_size: int,
            is_encoder_decoder_model: bool = False) -> T:
        assert self._is_graph_capturing

        attn_metadata = self.runner.attn_backend.make_metadata(
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            use_cuda_graph=True,
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
        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            if not hasattr(self, "context_chunk_workspace"):
                # not self.runner.device does not return the correct device
                # for this process, (init_device sets the correct device but
                # only on the Worker). The only way Ive figured out to get the
                # correct device is to allocate the workspace on the first call
                # to begin_forward and use the device of the input tokens
                assert model_input.input_tokens is not None
                self.context_chunk_workspace = torch.empty(
                    (self.context_chunk_workspace_size,
                     self.model_config.get_head_size()),
                    dtype=self.model_config.dtype,
                    device=model_input.input_tokens.device,
                )

            model_input.attn_metadata.context_chunk_workspace = \
                self.context_chunk_workspace


@dataclass
class MLACommonMetadata(AttentionMetadata):
    """Metadata for MLACommon. 
    
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # New for MLA (compared to FlashAttention)
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

    _cached_prefill_metadata: Optional[Any] = None
    _cached_decode_metadata: Optional[Any] = None

    num_prefill_tokens: int

    # The dimension of the attention heads
    head_dim: Optional[int] = None

    # Used when chunked prefill is enabled to simulate worst case workspace
    # allocations, hopefully to avoid going OOM
    is_profile_run: bool = False

    # New for MLA (compared to FlashAttention)
    # For chunked prefill
    context_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    context_chunk_starts: Optional[torch.Tensor] = None
    context_chunk_seq_tot: Optional[List[int]] = None
    context_chunk_max_seq_lens: Optional[List[int]] = None
    # Set by MLAAttentionState in `begin_forward` so it doesn't get broadcasted
    context_chunk_workspace: Optional[torch.Tensor] = None

    def __post_init__(self):
        supported_head_sizes = MLACommonBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f" received {self.head_dim}.")

    @property
    def prefill_metadata(self):
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

        self._cached_prefill_metadata = self.__class__(
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
            is_profile_run=self.is_profile_run,
            # MLACommonMetadata Chunk prefill specific
            context_chunk_cu_seq_lens=self.context_chunk_cu_seq_lens,
            context_chunk_starts=self.context_chunk_starts,
            context_chunk_seq_tot=self.context_chunk_seq_tot,
            context_chunk_max_seq_lens=self.context_chunk_max_seq_lens,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self):
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

        self._cached_decode_metadata = self.__class__(
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
            head_dim=self.head_dim,
            is_profile_run=self.is_profile_run)
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
            # When Multi-Step is enabled with Chunked-Prefill, prefills and
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


class MLACommonMetadataBuilder(AttentionMetadataBuilder[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class
    """

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.chunked_prefill_enabled = \
            self.runner.scheduler_config.chunked_prefill_enabled
        self.enable_prefix_caching = \
            self.runner.cache_config.enable_prefix_caching

        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            attn_state = self.input_builder.runner.attn_state
            self.context_chunk_workspace_size = \
                attn_state.context_chunk_workspace_size
            self.page_size = self.runner.block_size

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

        context_chunk_cu_seq_lens = None
        context_chunk_starts = None
        context_chunk_seq_tot = None
        context_chunk_max_seq_lens = None

        if (self.chunked_prefill_enabled or self.enable_prefix_caching) \
            and self.num_prefills > 0 \
            and context_lens_tensor is not None \
            and context_lens_tensor[:self.num_prefills].max() > 0:

            # NOTE: it is recommend you read the `Chunked Prefill` section in
            # the comment at the top of the file before trying to understand
            # the following code

            num_prefills_with_context = \
                (context_lens_tensor[:self.num_prefills] > 0).sum().item()

            # currently we allocate an equal amount of workspace for each
            # prefill in the batch, we could probably use a more advanced
            # algorithm here and allocate more workspace to prefills with
            # longer context lengths
            max_context_chunk = \
                self.context_chunk_workspace_size // num_prefills_with_context

            # align max_context_chunk to page_size by rounding down,
            # currently the `gather_cache` kernel cannot handle
            # `context_chunk_starts` that are not aligned to page_size
            max_context_chunk = round_down(max_context_chunk, self.page_size)
            assert max_context_chunk > 0
            num_chunks = cdiv(context_lens_tensor.max(), max_context_chunk)

            # if `max_context_chunk = 256`, `num_chunks = 3`, and
            #   `num_prefills_with_context = 4`, create a tensor that looks like
            #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
            context_chunk_starts = \
                torch.arange(num_chunks, device=device, dtype=torch.int32)\
                .unsqueeze(1).expand(-1, self.num_prefills)\
                * max_context_chunk
            chunk_ends = torch.min(context_lens_tensor[:self.num_prefills]\
                .unsqueeze(0), context_chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - context_chunk_starts).clamp(min=0)
            _context_chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(
                torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device)\
                .unsqueeze(-1)
            context_chunk_cu_seq_lens = \
                torch.cat([zero, _context_chunk_cu_seq_lens], dim=1)
            context_chunk_max_seq_lens = \
                chunk_seq_lens.max(dim=1).values.tolist()
            context_chunk_seq_tot = chunk_seq_lens.sum(dim=1).tolist()
            assert max(context_chunk_seq_tot) <= \
                self.context_chunk_workspace_size

        return self.runner.attn_backend.make_metadata(
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
            is_profile_run=self.runner.in_profile_run,
            # MLACommonMetadata Chunk prefill specific
            context_chunk_cu_seq_lens=context_chunk_cu_seq_lens,
            context_chunk_starts=context_chunk_starts,
            context_chunk_seq_tot=context_chunk_seq_tot,
            context_chunk_max_seq_lens=context_chunk_max_seq_lens,
        )


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class
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

        self.triton_fa_func = triton_attention
        # Handle the differences between the flash_attn_varlen from flash_attn
        # and the one from vllm_flash_attn. The former is used on RoCM and the
        # latter has an additional parameter to control FA2 vs FA3
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version()
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = \
                functools.partial(flash_attn_varlen_func,
                                  fa_version=self.vllm_flash_attn_version)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims
        # We don't need to pad V if we are on a hopper system with FA3
        self._pad_v = self.vllm_flash_attn_version is None or not (
            self.vllm_flash_attn_version == 3
            and current_platform.get_device_capability()[0] == 9)

    def _flash_attn_varlen_diff_headdims(self, q, k, v, softmax_scale,
                                         return_softmax_lse, **kwargs):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0)

        if is_hip and envs.VLLM_USE_TRITON_FLASH_ATTN \
            and not return_softmax_lse:
            attn_out = self.triton_fa_func(
                q,
                k,
                maybe_padded_v,
                **kwargs,
            )
        if is_vllm_fa:
            attn_out = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=maybe_padded_v,
                return_softmax_lse=return_softmax_lse,
                softmax_scale=softmax_scale,
                **kwargs,
            )
        else:
            # Use return_attn_probs instead of return_softmax_lse for RoCM
            attn_out = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=maybe_padded_v,
                return_attn_probs=return_softmax_lse,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        # Unpack the output if there is multiple results,
        # triton always returns (output, softmax_lse),
        # vllm_flash_attn returns (output, softmax_lse) when
        #  `return_softmax_lse = True`
        # flash_attn (RoCM) returns (output, softmax_lse, ...) when
        #  `return_attn_probs = True`
        rest = None
        if isinstance(attn_out, tuple):
            attn_out, *rest = attn_out

        # unpad if necessary
        if self._pad_v:
            attn_out = attn_out[..., :v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            assert rest is not None
            return attn_out, rest[0]
        return attn_out

    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(x)[0]

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = self.q_proj(x)[0]\
            .view(-1, self.num_heads, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
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

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None
        assert prefill_metadata.context_chunk_seq_tot is not None
        assert prefill_metadata.context_chunk_cu_seq_lens is not None
        assert prefill_metadata.context_chunk_starts is not None
        assert prefill_metadata.context_chunk_max_seq_lens is not None
        assert prefill_metadata.context_lens_tensor is not None

        output = None
        iters = len(prefill_metadata.context_chunk_seq_tot)

        # Fetch from attn_metadata directly, since it late bound by
        # MLAAttentionState, grabbing it directly `attn_metadata` can avoid
        # any weirdness around prefill_metadata caching
        assert attn_metadata.context_chunk_workspace is not None
        workspace = attn_metadata.context_chunk_workspace

        for i in range(iters):
            toks = prefill_metadata.context_chunk_seq_tot[i]

            ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_tables,
                cu_seq_lens=prefill_metadata.context_chunk_cu_seq_lens[i],
                batch_size=prefill_metadata.num_prefills,
                seq_starts=prefill_metadata.context_chunk_starts[i],
            )

            kv_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank]
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

            attn_output, attn_softmax_lse = \
                self._flash_attn_varlen_diff_headdims(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i],
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.context_chunk_max_seq_lens[i],
                softmax_scale=self.scale,
                causal=False,  # Context is unmasked
                return_softmax_lse=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
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

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        has_context = prefill_metadata.context_lens_tensor is not None \
            and prefill_metadata.context_lens_tensor.max() > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill_metadata.query_start_loc,
            cu_seqlens_k=prefill_metadata.query_start_loc,
            max_seqlen_q=prefill_metadata.max_prefill_seq_len,
            max_seqlen_k=prefill_metadata.max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=has_context,
        )

        if has_context:
            # ROCm flash_attn_varlen_func will return 3 objects instead of 2
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

        return self.o_proj(output.flatten(start_dim=-2))[0]

    @abstractmethod
    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
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

        if attn_metadata.is_profile_run and \
            attn_metadata.context_chunk_workspace is not None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (attn_metadata.context_chunk_workspace.shape[0],
                 self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

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
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_hs_or_q_c)
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
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata)

        return output
