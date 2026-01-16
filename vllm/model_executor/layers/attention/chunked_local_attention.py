# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from dataclasses import replace

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import subclass_attention_backend
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
def _compute_virtual_query_start_locs(
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
    # Padding requests (q_seqlen=0) produce 0 VBs; others get at least 1
    num_vb_per_req = (q_seqlens > 0) * (
        1 + (q_in_remaining_chunks + chunk - 1) // chunk
    )
    # Prepend 0 and compute cumsum for output offsets
    virtual_query_start_locs = torch.zeros(
        seq_lens.shape[0] + 1, dtype=torch.int32, device=seq_lens.device
    )
    virtual_query_start_locs[1:] = torch.cumsum(num_vb_per_req, dim=0)
    return virtual_query_start_locs


@triton.jit
def _compute_virtual_batches_attn_metadata_kernel(
    # Inputs
    query_start_loc_ptr,  # [bs + 1]
    seq_lens_ptr,  # [bs]
    virtual_query_start_locs_ptr,  # [bs + 1] - cumsum of vb per request
    block_table_ptr,  # [bs, max_blocks_per_seq]
    # Outputs
    seqlens_k_ptr,  # [num_vb_ub] - virtual batch kv seqlens
    virtual_query_start_locs_out_ptr,  # [num_vb_ub + 1] - cumulative query seqlens
    virtual_batches_block_table_ptr,  # [num_vb_ub * pages_per_virtual_batch]
    batch_mapping_ptr,  # [num_vb_ub] - maps vb -> original batch
    block_indices_ptr,  # [num_vb_ub * pages_per_virtual_batch] - block indices
    # Sizes
    bs,
    max_blocks_per_seq,
    # Constants
    ATTN_CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGES_PER_VIRTUAL_BATCH: tl.constexpr,
    MAX_VIRTUAL_BATCHES: tl.constexpr,  # Max virtual batches per request
):
    batch_idx = tl.program_id(0)
    if batch_idx >= bs:
        return

    # Load batch boundaries and sequence data
    output_start = tl.load(virtual_query_start_locs_ptr + batch_idx)
    output_end = tl.load(virtual_query_start_locs_ptr + batch_idx + 1)
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
    tokens_first = tl.minimum(space_in_first, q_seqlen)

    # Compute tokens_in_last_block
    last_remainder = kv_seqlen % ATTN_CHUNK_SIZE
    tokens_last = tl.where(last_remainder == 0, ATTN_CHUNK_SIZE, last_remainder)

    # Running sum for virtual_query_start_locs
    cu_q_running = q_start

    # Loop over virtual batches for this request (use mask instead of break)
    for vb_local_idx in range(MAX_VIRTUAL_BATCHES):
        valid = vb_local_idx < num_vb
        vb_idx = output_start + vb_local_idx

        # Compute seqlen_q
        is_first = vb_local_idx == 0
        consumed = tl.where(vb_local_idx > 0, (vb_local_idx - 1) * ATTN_CHUNK_SIZE, 0)
        remaining = q_seqlen - tokens_first - consumed
        seqlen_q = tl.where(
            is_first,
            tokens_first,
            tl.minimum(tl.maximum(remaining, 0), ATTN_CHUNK_SIZE),
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

        # Update and store virtual_query_start_locs
        cu_q_running = tl.where(valid, cu_q_running + seqlen_q, cu_q_running)
        tl.store(
            virtual_query_start_locs_out_ptr + vb_idx + 1, cu_q_running, mask=valid
        )

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
            self._virtual_query_start_loc = torch.zeros(
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
            self._virtual_query_start_loc_cpu = torch.zeros(
                num_vb_ub + 1, dtype=torch.int32, pin_memory=True
            )

        @classmethod
        def get_cudagraph_support(
            cls: type["AttentionMetadataBuilder"],
            vllm_config: VllmConfig,
            kv_cache_spec: AttentionSpec,
        ) -> AttentionCGSupport:
            # Support UNIFORM_BATCH for FULL CGs (decode with uniform q_len=1)
            # Each request produces exactly 1 virtual batch, so num_vb = bs
            return AttentionCGSupport.UNIFORM_BATCH

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ):
            block_table = common_attn_metadata.block_table_tensor
            query_start_loc = common_attn_metadata.query_start_loc
            seq_lens = common_attn_metadata.seq_lens
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            chunk = attention_chunk_size

            bs = common_attn_metadata.num_reqs

            # Compute num_vb from CPU data (no GPU sync needed)
            # N query tokens span at most: 1 + (N + chunk - 2) // chunk
            # Padding requests (q_seqlen=0) produce 0 VBs
            query_start_loc_np = query_start_loc_cpu.numpy()
            q_seqlens_np = query_start_loc_np[1 : bs + 1] - query_start_loc_np[:bs]
            # _ub stands for upper bound
            num_vb_per_req_ub = (q_seqlens_np > 0) * (
                1 + (q_seqlens_np + chunk - 2) // chunk
            )
            num_vb_ub = int(num_vb_per_req_ub.sum())
            max_vb_per_req = max(1, int(num_vb_per_req_ub.max()))

            # Compute cumulative virtual batches per request on GPU
            virtual_query_start_locs = _compute_virtual_query_start_locs(
                query_start_loc, seq_lens, chunk
            )

            # Get max_blocks_per_seq from actual block_table shape
            max_blocks_per_seq = block_table.shape[1]
            pages_per_vb = self._pages_per_virtual_batch

            # Zero buffers before kernel to clear stale data from previous
            # calls (kernel uses masked writes, stale data may remain)
            self._virtual_batch_to_batch_mapping[:num_vb_ub].zero_()
            self._virtual_batch_block_indices[:num_vb_ub, :pages_per_vb].zero_()
            # Pre-fill query_start_loc with total_tokens for monotonicity;
            # kernel will overwrite positions 1..actual_num_vb
            total_tokens = int(query_start_loc_cpu[-1])
            self._virtual_query_start_loc[: num_vb_ub + 1].fill_(total_tokens)
            self._virtual_query_start_loc[0] = 0

            _compute_virtual_batches_attn_metadata_kernel[(bs,)](
                query_start_loc,
                seq_lens,
                virtual_query_start_locs,
                block_table,
                self._virtual_seqlens,
                self._virtual_query_start_loc,
                self._virtual_batches_block_table,
                self._virtual_batch_to_batch_mapping,
                self._virtual_batch_block_indices,
                bs,
                max_blocks_per_seq,
                ATTN_CHUNK_SIZE=chunk,
                BLOCK_SIZE=block_size,
                PAGES_PER_VIRTUAL_BATCH=pages_per_vb,
                MAX_VIRTUAL_BATCHES=max_vb_per_req,
            )

            # Pad remaining positions for FULL CG (must be monotonic)
            self._virtual_query_start_loc[num_vb_ub + 1 :].fill_(total_tokens)

            # Record event after kernel + padding for lazy D2H sync
            kernel_done_event = current_stream().record_event()

            # Compute query_start_loc_cpu for virtual batches.
            # Uniform decode (max_q_len == 1): reuse input directly.
            # Prefill/spec-decode: async D2H copy with lazy sync on access.
            max_q_len = common_attn_metadata.max_query_len
            if max_q_len == 1:
                virtual_query_start_loc_cpu = query_start_loc_cpu
            else:
                # this buffer will get lazily populated below
                virtual_query_start_loc_cpu = self._virtual_query_start_loc_cpu[
                    : num_vb_ub + 1
                ]

            # Build metadata with virtual batch tensors
            cm = CommonAttentionMetadata(
                query_start_loc=self._virtual_query_start_loc[: num_vb_ub + 1],
                query_start_loc_cpu=virtual_query_start_loc_cpu,
                seq_lens=self._virtual_seqlens[:num_vb_ub],
                num_reqs=num_vb_ub,
                num_actual_tokens=common_attn_metadata.num_actual_tokens,
                max_query_len=chunk,
                max_seq_len=chunk,
                block_table_tensor=self._virtual_batches_block_table[:num_vb_ub],
                slot_mapping=common_attn_metadata.slot_mapping,
                causal=True,
            )

            # lazily populate the cpu buffer if needed
            if max_q_len > 1:
                # Add a hook to copy on the first access to query_start_loc_cpu so
                # for backends that don't use it, never sync.
                cm._virtual_query_start_loc_cpu_copied = False

                def _get_cpu(self):
                    if not cm._virtual_query_start_loc_cpu_copied:
                        # Wait for kernel to complete before D2H copy
                        kernel_done_event.synchronize()
                        virtual_query_start_loc_cpu.copy_(
                            self.query_start_loc[: num_vb_ub + 1]
                        )
                        cm._virtual_query_start_loc_cpu_copied = True
                    return virtual_query_start_loc_cpu

                cm.__class__ = type(
                    "CM_LazySync",
                    (cm.__class__,),
                    {"query_start_loc_cpu": property(_get_cpu)},
                )

            metadata = super().build(common_prefix_len, cm, fast_build)

            batch_mapping = self._virtual_batch_to_batch_mapping[:num_vb_ub]
            block_indices = self._virtual_batch_block_indices[:num_vb_ub, :pages_per_vb]

            def make_virtual_batches_block_table(blk_table: torch.Tensor):
                return blk_table[batch_mapping.unsqueeze(1), block_indices]

            metadata.make_virtual_batches_block_table = make_virtual_batches_block_table
            return metadata

        def update_block_table(
            self, metadata, blk_table: torch.Tensor, slot_mapping: torch.Tensor
        ):
            new_block_table = metadata.make_virtual_batches_block_table(blk_table)
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
