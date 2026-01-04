# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from dataclasses import replace

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    subclass_attention_backend,
)
from vllm.v1.attention.backends.utils import (
    make_local_attention_virtual_batches,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    KVCacheSpec,
)

"""
Chunked Local Attention Implementation

Implements chunked local attention by splitting sequences into "virtual batches"
- one per attention chunk. Each virtual batch attends only to tokens within its
chunk, enabling chunked local attention without modifying the underlying backend.

For example, if performing a chunked prefill on a batch of 3 sequences:
  q_seqlens  = [4, 10, 5]
  kv_seqlens = [6, 17, 9]

For regular attention, batch idx 0 (q_seqlens=4, kv_seqlens=6) would use mask:
       k_toks >   0 1 2 3 4 5
       q_toks v  _____________
              0 | 1 1 1
              1 | 1 1 1 1
              2 | 1 1 1 1 1
              3 | 1 1 1 1 1 1

For chunked local attention (attn_chunk_size=4), the mask becomes:
       k_toks >   0 1 2 3 4 5
       q_toks v  _____________
              0 | 1 1 1
              1 | 1 1 1 1
              2 |         1
              3 |         1 1

We simulate this by breaking sequences into virtual batches, each covering one
attention chunk. Batch idx 0 becomes:

  virtual batch 0 (q_seqlens=2, kv_seqlens=4):
       k_toks >   0 1 2 3
       q_toks v  _____________
              0 | 1 1 1
              1 | 1 1 1 1

  virtual batch 1 (q_seqlens=2, kv_seqlens=2):
       k_toks >   4 5
       q_toks v  _____________
              2 | 1
              3 | 1 1
"""


@torch.compile(dynamic=True)
def _compute_cu_num_vb(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Compute cumulative virtual batches per request (fused via torch.compile)."""
    q_seqlens = query_start_loc[1:] - query_start_loc[:-1]
    context_lens = seq_lens - q_seqlens
    space_in_first_chunk = chunk - context_lens % chunk
    q_in_first_chunk = torch.minimum(space_in_first_chunk, q_seqlens)
    q_in_remaining_chunks = q_seqlens - q_in_first_chunk
    num_vb_per_req = 1 + (q_in_remaining_chunks + chunk - 1) // chunk
    # Prepend 0 and compute cumsum for output offsets
    cu_num_vb = torch.zeros(
        seq_lens.shape[0] + 1, dtype=torch.int32, device=seq_lens.device
    )
    cu_num_vb[1:] = torch.cumsum(num_vb_per_req, dim=0)
    return cu_num_vb


@triton.jit
def _compute_virtual_batches_attn_metadata_kernel(
    # Inputs
    query_start_loc_ptr,  # [batch_size + 1]
    seq_lens_ptr,  # [batch_size]
    cu_num_vb_ptr,  # [batch_size + 1] - cumsum of virtual batches per request
    block_table_ptr,  # [batch_size, max_blocks_per_seq]
    # Outputs
    seqlens_k_ptr,  # [num_vb_ub] - virtual batch kv seqlens
    cu_seqlens_q_ptr,  # [num_vb_ub + 1] - cumulative query seqlens
    virtual_batches_block_table_ptr,  # [num_vb_ub * pages_per_virtual_batch]
    batch_mapping_ptr,  # [num_vb_ub] - maps vb -> original batch
    block_indices_ptr,  # [num_vb_ub * pages_per_virtual_batch] - block indices
    # Sizes
    batch_size,
    max_blocks_per_seq,
    # Constants
    ATTN_CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGES_PER_VIRTUAL_BATCH: tl.constexpr,
    MAX_VIRTUAL_BATCHES: tl.constexpr,  # Max virtual batches per request
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Load batch boundaries and sequence data
    output_start = tl.load(cu_num_vb_ptr + batch_idx)
    output_end = tl.load(cu_num_vb_ptr + batch_idx + 1)
    num_vb = output_end - output_start

    q_start = tl.load(query_start_loc_ptr + batch_idx)
    q_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    q_seqlen = q_end - q_start
    kv_seqlen = tl.load(seq_lens_ptr + batch_idx)

    # Compute q_tokens_in_first_block
    context_len = kv_seqlen - q_seqlen
    remainder = context_len % ATTN_CHUNK_SIZE
    space_in_first = tl.where(
        remainder == 0, ATTN_CHUNK_SIZE, ATTN_CHUNK_SIZE - remainder
    )
    q_first = tl.minimum(space_in_first, q_seqlen)

    # Compute tokens_in_last_block
    last_remainder = kv_seqlen % ATTN_CHUNK_SIZE
    tokens_last = tl.where(last_remainder == 0, ATTN_CHUNK_SIZE, last_remainder)

    # Running sum for cu_seqlens_q (base offset is q_start from query_start_loc)
    cu_q_running = q_start

    # Loop over virtual batches for this request (use mask instead of break)
    for vb_local_idx in range(MAX_VIRTUAL_BATCHES):
        valid = vb_local_idx < num_vb
        vb_idx = output_start + vb_local_idx

        # Compute seqlen_q
        is_first = vb_local_idx == 0
        consumed = tl.where(vb_local_idx > 0, (vb_local_idx - 1) * ATTN_CHUNK_SIZE, 0)
        remaining = q_seqlen - q_first - consumed
        seqlen_q = tl.where(
            is_first, q_first, tl.minimum(tl.maximum(remaining, 0), ATTN_CHUNK_SIZE)
        )

        # Compute seqlen_k (0 for padding entries where kv_seqlen=0)
        is_last = vb_local_idx == num_vb - 1
        seqlen_k = tl.where(
            kv_seqlen > 0, tl.where(is_last, tokens_last, ATTN_CHUNK_SIZE), 0
        )

        # Compute block_start_idx for block table
        rarange = num_vb - vb_local_idx - 1
        k_seqstart = kv_seqlen - (rarange * ATTN_CHUNK_SIZE + tokens_last)
        k_seqstart = tl.maximum(k_seqstart, 0)
        block_start_idx = k_seqstart // BLOCK_SIZE

        # Store outputs (masked)
        tl.store(seqlens_k_ptr + vb_idx, seqlen_k, mask=valid)
        tl.store(batch_mapping_ptr + vb_idx, batch_idx, mask=valid)

        # Update and store cu_seqlens_q
        cu_q_running = tl.where(valid, cu_q_running + seqlen_q, cu_q_running)
        tl.store(cu_seqlens_q_ptr + vb_idx + 1, cu_q_running, mask=valid)

        # Store block table entries and indices (masked)
        for page_idx in range(PAGES_PER_VIRTUAL_BATCH):
            flat_idx = vb_idx * PAGES_PER_VIRTUAL_BATCH + page_idx
            block_idx = tl.minimum(block_start_idx + page_idx, max_blocks_per_seq - 1)
            src_idx = batch_idx * max_blocks_per_seq + block_idx
            block_val = tl.load(block_table_ptr + src_idx, mask=valid, other=0)
            tl.store(virtual_batches_block_table_ptr + flat_idx, block_val, mask=valid)
            tl.store(block_indices_ptr + flat_idx, block_idx, mask=valid)


@functools.lru_cache
def create_chunked_local_attention_backend(
    underlying_attn_backend: AttentionBackend,
    attention_chunk_size: int,
    block_size: int,
) -> type[AttentionBackend]:
    prefix = f"ChunkedLocalAttention_{attention_chunk_size}_{block_size}_"

    underlying_builder = underlying_attn_backend.get_builder_cls()
    assert issubclass(underlying_builder, AttentionMetadataBuilder)

    class ChunkedLocalAttentionBuilder(underlying_builder):  # type: ignore
        supports_update_block_table: bool = True

        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            # Compute loose, upper bound on number of virtual batches
            # for persistent buffer allocation
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            sched = vllm_config.scheduler_config
            max_num_batched_tokens = sched.max_num_batched_tokens
            max_vb_per_req = (
                1
                + (max_num_batched_tokens + attention_chunk_size - 2)
                // attention_chunk_size
            )
            num_vb_ub = max_num_seqs * max_vb_per_req
            pages_per_virtual_batch = attention_chunk_size // block_size

            # Create modified config with num_vb_ub as max_num_seqs so the
            # underlying builder allocates buffers large enough for virtual
            # batches (required for CUDA graph support so the underlying builder
            # can allocate persistent buffers large enough for virtual batches)
            # Also bump max_num_batched_tokens if needed to satisfy the
            # max_num_batched_tokens >= max_num_seqs validation
            modified_max_batched = max(max_num_batched_tokens, num_vb_ub)
            modified_scheduler_config = replace(
                vllm_config.scheduler_config,
                max_model_len=vllm_config.model_config.max_model_len,
                is_encoder_decoder=vllm_config.model_config.is_encoder_decoder,
                max_num_seqs=num_vb_ub,
                max_num_batched_tokens=modified_max_batched,
            )
            modified_vllm_config = replace(
                vllm_config, scheduler_config=modified_scheduler_config
            )

            # Call parent __init__ with modified config
            super().__init__(kv_cache_spec, layer_names, modified_vllm_config, device)

            # Store for use in build()
            self._pages_per_virtual_batch = pages_per_virtual_batch

            # Pre-allocate persistent buffers for virtual batch metadata
            self._virtual_seqlens = torch.zeros(
                num_vb_ub, dtype=torch.int32, device=device
            )
            self._cu_virtual_seqlens_q = torch.zeros(
                num_vb_ub + 1, dtype=torch.int32, device=device
            )
            self._virtual_batch_to_batch_mapping = torch.zeros(
                num_vb_ub, dtype=torch.int32, device=device
            )
            self._virtual_batches_block_table = torch.zeros(
                (num_vb_ub, pages_per_virtual_batch),
                dtype=torch.int32,
                device=device,
            )
            self._virtual_batch_block_indices = torch.zeros(
                (num_vb_ub, pages_per_virtual_batch),
                dtype=torch.int32,
                device=device,
            )

            # Pinned memory buffer for async GPU->CPU copy of cu_virtual_seqlens_q
            self._cu_virtual_seqlens_q_cpu = torch.zeros(
                num_vb_ub + 1, dtype=torch.int32, pin_memory=True
            )

        @classmethod
        def get_cudagraph_support(
            cls: type["AttentionMetadataBuilder"],
            vllm_config: VllmConfig,
            kv_cache_spec: AttentionSpec,
        ) -> AttentionCGSupport:
            # Support UNIFORM_BATCH for FULL CGs (decode with uniform q_len=1)
            # Each request produces exactly 1 virtual batch, so num_vb = batch_size
            return AttentionCGSupport.UNIFORM_BATCH

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ):
            batch_size = common_attn_metadata.num_reqs
            block_table = common_attn_metadata.block_table_tensor
            query_start_loc = common_attn_metadata.query_start_loc
            seq_lens = common_attn_metadata.seq_lens
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            chunk = attention_chunk_size

            # Compute num_vb_ub from CPU data (no GPU sync)
            # N query tokens can span at most: 1 + (N + chunk - 2) // chunk
            q_seqlens_cpu = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).numpy()
            num_vb_per_req_ub = 1 + (q_seqlens_cpu + chunk - 2) // chunk
            num_vb_ub = int(num_vb_per_req_ub.sum())
            max_vb_per_req = max(1, int(num_vb_per_req_ub.max()))

            # Compute cumulative virtual batches per request on GPU
            cu_num_vb = _compute_cu_num_vb(query_start_loc, seq_lens, chunk)

            # Get max_blocks_per_seq from actual block_table shape
            max_blocks_per_seq = block_table.shape[1]
            pages_per_vb = self._pages_per_virtual_batch

            # Zero buffers before kernel to clear stale data from previous
            # calls (kernel uses masked writes, stale data may remain)
            self._virtual_batch_to_batch_mapping[:num_vb_ub].zero_()
            self._virtual_batch_block_indices[:num_vb_ub, :pages_per_vb].zero_()

            _compute_virtual_batches_attn_metadata_kernel[(batch_size,)](
                query_start_loc,
                seq_lens,
                cu_num_vb,
                block_table,
                self._virtual_seqlens,
                self._cu_virtual_seqlens_q,
                self._virtual_batches_block_table,
                self._virtual_batch_to_batch_mapping,
                self._virtual_batch_block_indices,
                batch_size,
                max_blocks_per_seq,
                ATTN_CHUNK_SIZE=chunk,
                BLOCK_SIZE=block_size,
                PAGES_PER_VIRTUAL_BATCH=pages_per_vb,
                MAX_VIRTUAL_BATCHES=max_vb_per_req,
            )

            # Pad cu_virtual_seqlens_q for FULL CG (must be monotonic)
            total_tokens = int(query_start_loc_cpu[-1])
            self._cu_virtual_seqlens_q[num_vb_ub + 1 :].fill_(total_tokens)

            # Compute query_start_loc_cpu for virtual batches.
            # We handle two cases differently to avoid CPU<>GPU sync:
            #
            # 1. Uniform single token decode case (max_q_len == 1):
            #    Each request has exactly 1 query token, which means each request
            #    produces exactly 1 virtual batch. Therefore num_vb == batch_size
            #    and cu_virtual_seqlens = [0, 1, 2, ..., batch_size], which is
            #    identical to the input query_start_loc_cpu. We can reuse it.
            #
            # 2. Spec-decode / Prefill case (max_q_len > 1):
            #    Requests have varying query lengths and may span multiple chunks,
            #    so we don't know the virtual batch boundaries without running the
            #    Triton kernel. We use a non-blocking copy from GPU to pinned CPU
            #    memory so backends like FlashAttn (which don't use query_start_loc_cpu)
            #    can continue building metadata asynchronously.
            max_q_len = common_attn_metadata.max_query_len
            if max_q_len == 1:
                # Uniform single token decode: reuse input cu_seqlens directly
                cu_virtual_seqlens_q_cpu = query_start_loc_cpu
            else:
                # Spec-decode / Prefill: async copy to pinned memory
                self._cu_virtual_seqlens_q_cpu[: num_vb_ub + 1].copy_(
                    self._cu_virtual_seqlens_q[: num_vb_ub + 1], non_blocking=True
                )
                cu_virtual_seqlens_q_cpu = self._cu_virtual_seqlens_q_cpu[
                    : num_vb_ub + 1
                ]

            # Use dynamically sized tensors (sliced to actual virtual batch count)
            cm = CommonAttentionMetadata(
                query_start_loc=self._cu_virtual_seqlens_q[: num_vb_ub + 1],
                query_start_loc_cpu=cu_virtual_seqlens_q_cpu,
                seq_lens=self._virtual_seqlens[:num_vb_ub],
                num_reqs=num_vb_ub,
                num_actual_tokens=common_attn_metadata.num_actual_tokens,
                max_query_len=chunk,
                max_seq_len=chunk,
                block_table_tensor=self._virtual_batches_block_table[:num_vb_ub],
                slot_mapping=common_attn_metadata.slot_mapping,
                causal=True,
            )

            metadata = super().build(common_prefix_len, cm, fast_build)

            # Clone indices onto metadata so they're stable for
            # update_block_table (different layers have different builders)
            metadata._virtual_batch_to_batch_mapping = (
                self._virtual_batch_to_batch_mapping[:num_vb_ub].clone()
            )
            # Only keep the columns we actually use (clamped to max_blocks_per_seq)
            metadata._virtual_batch_block_indices = self._virtual_batch_block_indices[
                :num_vb_ub, :pages_per_vb
            ].clone()
            return metadata

        def update_block_table(
            self, metadata, blk_table: torch.Tensor, slot_mapping: torch.Tensor
        ):
            # Use cloned indices stored on metadata (stable across builders)
            new_block_table = blk_table[
                metadata._virtual_batch_to_batch_mapping.unsqueeze(1),
                metadata._virtual_batch_block_indices,
            ]
            return super().update_block_table(metadata, new_block_table, slot_mapping)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=ChunkedLocalAttentionBuilder,
    )

    return attn_backend


class ChunkedLocalAttention(Attention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        attention_chunk_size: int,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        kv_sharing_target_layer_name: str | None = None,
        prefix: str = "",
    ):
        self.attention_chunk_size = attention_chunk_size
        dtype = torch.get_default_dtype()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        underlying_attn_backend = get_attn_backend(
            head_size, dtype, kv_cache_dtype, block_size
        )
        attn_backend = create_chunked_local_attention_backend(
            underlying_attn_backend, attention_chunk_size, block_size
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        assert self.attention_chunk_size
        return ChunkedLocalAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            attention_chunk_size=self.attention_chunk_size,
        )
