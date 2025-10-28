# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlexAttention."""

import math
from dataclasses import dataclass

import torch
import torch._dynamo.decorators
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
)

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

create_block_mask_compiled = torch.compile(
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _offsets_to_doc_ids_tensor(offsets: torch.Tensor) -> torch.Tensor:
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int):
    difference = (multiple - (x.shape[dim] % multiple)) % multiple
    if difference == 0:
        return x

    dim = dim if dim >= 0 else x.ndim + dim
    pad_list = []

    for i in range(x.ndim - 1, dim - 1, -1):
        if i == dim:
            pad_list.extend([0, difference])
        else:
            pad_list.extend([0, 0])

    return F.pad(x, pad_list, mode="constant", value=0)


class FlexAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        return  # FlexAttention supports any head size

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlexAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


# @torch.compile(fullgraph=True, mode="reduce-overhead")
def physical_to_logical_mapping(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> torch.Tensor:
    """
    Creates an inverse mapping from physical block locations to logical indices.

    The original block_table maps from logical blocks to physical locations:

    Logical to Physical (Original block_table):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Logical Blocks:  0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Physical Blocks: 3  5  1  7  4  2  0  6   │
    └───────────────────────────────────────────┘

    This function creates the inverse mapping:

    Physical to Logical (Inverse mapping):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Physical Blocks: 0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Logical Blocks:  6  2  5  0  4  1  7  3   │
    └───────────────────────────────────────────┘

    If multiple logical blocks map to the same physical block,
    this function returns the first (minimum) logical block index.

    If a physical block is not mapped to by any logical block,
    its value in the result will be -1.

    IMPORTANT: Garbage Value Protection
    ────────────────────────────────────
    The block_table tensor may contain garbage values in unused positions
    (beyond the actual sequence length). For example, if a sequence only
    needs 3 blocks but the table has space for 8:

        block_table[0] = [10, 25, 7, 999, 1234, 888, ...]
                                    ^^^^^^^^^^^^^^^^^^^^
                                    garbage values

    These garbage values can cause issues because:
    1. They may map to valid physical blocks by coincidence
    2. The scatter_ operation will assign them logical indices
    3. Later attention computations may incorrectly access these blocks

    To prevent this, we use seq_lens and block_size to mask out unused
    entries, ensuring only valid block references are processed.

    Args:
        block_table: Tensor of shape [max_reqs, max_num_blocks]
            mapping logical blocks to physical locations. May contain
            garbage values in unused positions.
        seq_lens: Tensor of sequence lengths for each request. Used to
            determine how many blocks are actually needed per sequence.
        block_size: Size of each block in tokens. Used with seq_lens to
            compute the number of valid blocks per sequence.
        total_blocks: Total number of physical blocks available

    Returns:
        A tensor of shape [max_reqs, total_blocks] where each entry
        physical_to_logical[req_id, physical_block] contains the logical
        block index for that physical block, or -1 if unused.
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    physical_to_logical = torch.full(
        (max_reqs, total_blocks), -1, dtype=torch.long, device=device
    )

    # Only process valid blocks to avoid garbage values
    num_blocks_per_seq = cdiv(seq_lens, block_size)
    mask = (
        torch.arange(max_num_blocks, device=device)[None, :]
        < num_blocks_per_seq[:, None]
    )

    valid_block_table = torch.where(mask, block_table, 0)
    valid_logical_indices = torch.where(
        mask, torch.arange(max_num_blocks, device=device)[None, :], 0
    )

    physical_to_logical.scatter_(
        -1, valid_block_table.to(torch.int64), valid_logical_indices
    )
    # NB - Seems like block 0 is always empty so we reset it manually
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def unique_static_unsorted(
    x: torch.Tensor,
    *,
    M: int,  # maximum positive value (0 is “skip me”)
    dim: int = -1,  # axis along which to deduplicate
    ignored_val: int = 0,  # value to ignore
    pad_val: int = -1,  # sentinel for unused slots
) -> torch.Tensor:
    """
    - Keeps the first occurrence of each non-zero value while preserving order,
      then left-packs those uniques and fills the rest with `pad_val`.
    - Returns (packed, keep_mask) with the *same shape* as `x`.
    - Requires that all values be in the range [0, M]
    - Skips ignored_val

    Works on CPU or GPU, no Python loops, O(B·N) time / O(B·M) memory.

    Example:
    x =[3, 1, 0, 1, 2], M=3, ignored_val=0 => [3, 1, 2, -1, -1]
    """
    if not (-1 <= pad_val <= M):
        raise ValueError("`pad_val` must lie in [-1, M]")

    # ── move `dim` to the end so we can treat tensor as [B, N] ──────────
    dim = dim % x.ndim
    x_perm = x.movedim(dim, -1)  # shape [..., N]
    B, N = x_perm.numel() // x_perm.shape[-1], x_perm.shape[-1]
    x_flat = x_perm.reshape(B, N)  # [B, N]

    device = x.device
    idx = torch.arange(N, device=device).expand(B, N)  # per-row indices

    # ── build first-occurrence table for every v ∈ [0, M] ───────────────
    first_idx = torch.full((B, M + 1), N, device=device)  # “∞”
    # scatter_reduce_: first_idx[b, v] = min(first_idx[b, v], i) for each i
    first_idx.scatter_reduce_(1, x_flat, idx, reduce="amin")

    # ── keep mask: first occurrence *and* value ≠ 0 ─────────────────────
    keep = (x_flat != ignored_val) & (idx == first_idx.gather(1, x_flat))  # [B, N]

    # ── left-pack uniques into a fresh tensor ───────────────────────────
    dest_pos = torch.cumsum(keep.to(torch.long), dim=1) - 1  # where to go
    packed_flat = torch.full_like(x_flat, pad_val)

    rows, src_cols = torch.nonzero(keep, as_tuple=True)
    packed_flat[rows, dest_pos[rows, src_cols]] = x_flat[rows, src_cols]

    # ── restore original layout ─────────────────────────────────────────
    packed = packed_flat.reshape(x_perm.shape).movedim(-1, dim)
    return packed


def causal_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    return q_idx >= kv_idx


@dataclass
class FlexAttentionMetadata:
    causal: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Block info
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor
    num_blocks_per_seq: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # Flex Metadata
    num_blocks = 0
    block_mask: BlockMask | None = None
    score_mod: _score_mod_signature | None = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod
    doc_ids: torch.Tensor | None = None
    direct_build: bool = True
    q_block_size: int = 16
    kv_block_size: int = 16
    transformed_score_mod: _score_mod_signature | None = None
    sliding_window: int | None = None

    def _convert_physical_to_logical(
        self,
        request_lookup: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert physical indices to logical indices for both query and kv.

        NB is_within_lower_bound: do sequences start on block_boundaries?

        Returns:
            tuple of (is_valid, logical_q_idx, logical_kv_idx)
        """
        # Map query indices to corresponding request indices
        q_req = request_lookup[q_idx]

        # Convert physical KV indices to logical indices
        physical_kv_block = physical_kv_idx // self.block_size
        physical_kv_offset = physical_kv_idx % self.block_size
        logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
        logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

        # Determine valid kv indices
        live_block = logical_block_idx >= 0
        within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
        within_lower_bound = logical_kv_idx >= 0
        is_valid = live_block & within_upper_bound & within_lower_bound

        # Convert physical query indices to logical indices
        local_q_idx = q_idx - self.query_start_loc[q_req]
        logical_q_idx = local_q_idx + self.decode_offset[q_req]

        return is_valid, logical_q_idx, logical_kv_idx

    def get_causal_mask_mod(self) -> _mask_mod_signature:
        """Creates the mask_mod function for FlexAttention.

        This function creates the combined mask mod function that handles:
            1. The paged attention block mapping
            2. The mapping from packed query sequences to logical query entries

        It also by defaults adds the decoding offset to the query indices.
        With this info we create the "logical" indices that are passed to
        mask_mod functions. This allows mask mod functions to be agnostic to
        layout of the query and key/value tensors.
        """
        assert self.doc_ids is not None

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            # Apply mask modification only for valid indices
            return torch.where(
                is_valid,
                self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def get_bidirectional_mask_mod(self) -> _mask_mod_signature:
        """Creates the encoder mask_mod function for FlexAttention.

        Since the encoder bidirectional attention doesn't run with
        KV cache, this function creates a mask based on the
        packed query sequences.
        """
        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return request_lookup[q_idx] == request_lookup[kv_idx]

        return final_mask_mod

    def get_sliding_window_mask_mod(self) -> _mask_mod_signature:
        """Creates the sliding window mask_mod function for FlexAttention.

        Note that the sliding window mask here is bidirectional, we need
        to mask it with the bidirectional/causal mask for encoder/decoder.
        """

        if self.sliding_window is None:
            raise ValueError("sliding_window must be set for sliding window attention")

        def sliding_window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return torch.abs(q_idx - kv_idx) < self.sliding_window

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return torch.where(
                is_valid,
                sliding_window_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod if self.causal else sliding_window_mask_mod

    def get_mask_mod(self):
        # Stage-1: initialize the base mask_mod
        # (causal mask for decoder or bidirectional mask for encoder)
        if self.causal:
            mask_mod = self.get_causal_mask_mod()
        else:
            mask_mod = self.get_bidirectional_mask_mod()
        # stage-2: add external mask_mod for special attention during
        # forwarding runtime to create the combined mask_mod.
        if self.sliding_window is not None:
            # Add sliding window mask for sliding window attention
            sliding_window_mask_mod = self.get_sliding_window_mask_mod()
            mask_mod = and_masks(mask_mod, sliding_window_mask_mod)
        return mask_mod

    def get_transformed_score_mod(self) -> _score_mod_signature | None:
        """Creates the transformed score_mod function for FlexAttention.

        This function wraps the user's score_mod to handle physical-to-logical
        index conversion, similar to how get_mask_mod works for mask functions.
        """
        if self.score_mod is None:
            return None

        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)
        user_score_mod = self.score_mod

        def transformed_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(
                    request_lookup, q_idx, physical_kv_idx
                )
            )

            return torch.where(
                is_valid,
                user_score_mod(
                    score, b, h, logical_q_idx, logical_kv_idx, physical_q=q_idx
                ),
                -float("inf"),
            )

        return transformed_score_mod

    def _build_block_mask_direct(self) -> BlockMask:
        """Direct block mask construction for standard causal attention.

        This method constructs the block mask directly using
        BlockMask.from_kv_blocks which is much more efficient than the
        generic create_block_mask approach.

        The direct path works as follows:
        1. For each query token, fetch blocks from block_table using max_seq_len
           (this fetches more blocks than needed for shorter sequences)
        2. Group query tokens into chunks of q_block_size
        3. For each group, deduplicate the blocks using unique_static_unsorted
        4. Create BlockMask using the deduplicated block indices

        Over-estimation occurs when a group of q_block_size tokens contains
        multiple sequence IDs (doc_ids). In this case, we fetch ALL blocks for
        each sequence represented in the group, even though individual query
        tokens may only need a subset of those blocks based on causal masking
        and their position.

        """
        page_to_block_ratio = self.kv_block_size // self.block_size
        if page_to_block_ratio != 1:
            raise ValueError(
                f"FlexAttention currently requires the cache block size "
                f"({self.block_size}) to be equal to the kv_block_size "
                f"({self.kv_block_size}). Please check your model's "
                f"configuration."
            )

        used_pages = self.block_table[
            self.doc_ids, : cdiv(self.max_seq_len, self.block_size)
        ]
        used_pages_padded = pad_to_multiple(
            used_pages, multiple=self.q_block_size, dim=0
        )
        used_pages_padded = used_pages_padded.reshape(
            used_pages_padded.shape[0] // self.q_block_size, -1
        )
        used_pages_padded = used_pages_padded // page_to_block_ratio
        kv_indices = unique_static_unsorted(
            (used_pages_padded.long()), M=self.num_blocks
        ).to(torch.int32)

        kv_num_blocks = (kv_indices >= 0).sum(dim=-1).to(torch.int32)
        block_mask_kwargs = {
            "seq_lengths": (self.num_actual_tokens, self.total_cache_tokens),
            "kv_num_blocks": kv_num_blocks[None, None],
            "kv_indices": kv_indices[None, None],
            "full_kv_num_blocks": None,
            "full_kv_indices": None,
            "BLOCK_SIZE": (self.q_block_size, self.kv_block_size),
            "mask_mod": self.mask_mod,
        }

        # compute_q_blocks parameter is available in PyTorch 2.9+
        if is_torch_equal_or_newer("2.9.0.dev0"):
            block_mask_kwargs["compute_q_blocks"] = False
        return BlockMask.from_kv_blocks(**block_mask_kwargs)

    def build_block_mask(self) -> BlockMask:
        mask_mod = self.get_mask_mod()
        kv_len = self.total_cache_tokens if self.causal else self.num_actual_tokens
        return create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )

    def __post_init__(self):
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        # Create a lookup mapping from query indices -> request number
        self.doc_ids = _offsets_to_doc_ids_tensor(self.query_start_loc)
        self.num_blocks = self.total_cache_tokens // self.block_size

        self.mask_mod = self.get_mask_mod()
        self.transformed_score_mod = self.get_transformed_score_mod()

        if self.direct_build and self.causal:
            self.block_mask = self._build_block_mask_direct()
        else:
            self.block_mask = self.build_block_mask()


class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        supports_small_blocks = is_torch_equal_or_newer("2.9.0.dev0")
        self.direct_build: bool = supports_small_blocks
        self.q_block_size: int = 16 if supports_small_blocks else 128
        self.kv_block_size: int = self.block_size if supports_small_blocks else 128

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks_per_seq = cdiv(seq_lens, self.block_size)

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        block_size = self.kv_cache_spec.block_size
        max_possible_seq_len = self.model_config.max_model_len
        num_gpu_blocks = self.cache_config.num_gpu_blocks

        assert num_gpu_blocks is not None, (
            "FlexAttention requires num_gpu_blocks to be set"
        )
        total_cache_tokens = num_gpu_blocks * block_size

        inverse_block_table = physical_to_logical_mapping(
            block_table_tensor, seq_lens, block_size, num_gpu_blocks
        )

        offset_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            self.device, non_blocking=True
        )

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            num_blocks_per_seq=num_blocks_per_seq,
            # FIXME(Isotr0py): direct build has issue to build bidirectional
            # attention block mask for encoder-only models, disable it temporarily.
            # see: https://github.com/vllm-project/vllm/pull/27329#issuecomment-3431484053
            direct_build=(self.direct_build and common_attn_metadata.causal),
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
        )
        return out

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class FlexAttentionImpl(AttentionImpl):
    sliding_window: int | None
    alibi_slopes: torch.Tensor | None
    logits_soft_cap: float | None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
            raise NotImplementedError(
                f"FlexAttention does not support {attn_type} attention"
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlexAttention does not support alibi slopes yet."
            )
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window

        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "FlexAttention does not support logits soft cap yet."
            )

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FlexAttention does not support kv sharing yet.")

        FlexAttentionBackend.validate_head_size(head_size)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet"
            )

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """View a 3d tensor as 4D."""
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FLexAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlexAttentionImpl"
            )

        enable_gqa = self.num_kv_heads != self.num_heads

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
            # query = self.view_as_4d(query).permute(0, 2, 1, 3)
            # return torch.empty_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if attn_metadata.sliding_window != self.sliding_window:
            attn_metadata.sliding_window = self.sliding_window
            if attn_metadata.direct_build:
                # TODO: Support skipping the computation of sliding window
                # in direct block mask building code path.
                logger.warning_once(
                    "Using direct block mask building with sliding window, "
                    "which is suboptimal now. Performance may be degraded."
                )
                # update mask mod in attention metadata
                attn_metadata.mask_mod = attn_metadata.get_mask_mod()
                attn_metadata.block_mask = attn_metadata._build_block_mask_direct()
            else:
                attn_metadata.block_mask = attn_metadata.build_block_mask()

        if not attn_metadata.causal:
            assert self.attn_type == AttentionType.ENCODER_ONLY

            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key, value),
            )

            query = query[:, :, :num_actual_tokens, :]
            if (key_tensor.size(-2) > num_actual_tokens) or (
                value_tensor.size(-2) > num_actual_tokens
            ):
                # In the encoder-only model with torch.compile,
                # qkv might be padded, which might cause exception.
                # see: https://github.com/vllm-project/vllm/pull/24872#discussion_r2353252290
                key_tensor = key_tensor[:, :, :num_actual_tokens, :]
                value_tensor = value_tensor[:, :, :num_actual_tokens, :]

        else:
            assert self.attn_type == AttentionType.DECODER
            key_cache, value_cache = kv_cache.unbind(0)

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

            # View out the block_size dim
            key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key_cache, value_cache),
            )

            query = query[:, :, :num_actual_tokens, :]

        # Doesn't work for now -> constraint violation
        # torch._dynamo.try_mark_dynamic(query, 2)

        assert attn_metadata.block_mask is not None
        block_m, block_n = attn_metadata.block_mask.BLOCK_SIZE

        kernel_options = get_kernel_options(
            query, block_m, block_n, attn_metadata.direct_build
        )
        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

        # Flex doesn't have an out variant today, rely on epilogue fusion
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output[:num_actual_tokens, :, :].copy_(out)
        return output


def get_kernel_options(
    query, block_m, block_n, use_direct_build: bool
) -> dict[str, int | bool]:
    kernel_options: dict[str, int | bool] = {
        "FORCE_USE_FLEX_ATTENTION": True,
    }

    def ensure_divisible(candidate: int, block_size: int) -> int:
        """Pick a kernel block size that divides the logical block."""
        if block_size <= 0:
            return candidate
        candidate = min(candidate, block_size)
        if candidate <= 0:
            return block_size
        if block_size % candidate == 0:
            return candidate

        candidate = math.gcd(candidate, block_size)
        if candidate <= 1:
            return block_size
        return candidate

    if vllm_is_batch_invariant():
        kernel_options["BLOCK_M"] = 16
        kernel_options["BLOCK_N"] = 16
        kernel_options["IS_DIVISIBLE"] = False
        return kernel_options
    if use_direct_build:
        kernel_options["BLOCK_M"] = block_m
        kernel_options["BLOCK_N"] = block_n
        return kernel_options
    else:
        preferred_block = 32 if query.dtype == torch.float32 else 64
        block_m_candidate = ensure_divisible(preferred_block, block_m)
        block_n_candidate = ensure_divisible(preferred_block, block_n)

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties()
            max_shared_memory = device_props.shared_memory_per_block_optin
            if max_shared_memory < 144 * 1024:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        kernel_options["BLOCK_M"] = block_m_candidate
        kernel_options["BLOCK_N"] = block_n_candidate

    return kernel_options
