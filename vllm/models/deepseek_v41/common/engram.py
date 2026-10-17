# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash lookups gated into the hyper-connection stream.

Port of the reference ``inference/engram.py`` + ``Engram`` /
``ParallelEngramEmbedding`` from ``inference/model.py`` (DeepSeek V4.1
checkpoint layout). Engram modules live on the backbone layers listed in
``engram_layer_ids`` only.

Two pieces of cross-forward state are needed because vLLM streams tokens
chunk-by-chunk while an n-gram at position ``p`` needs the token ids at
``p-1..p-3``:

- ``token_map``: token id -> compressed vocab id, built once from the
  model's tokenizer at init (deterministic; asserted against
  ``engram_compressed_vocab_size``).
- ``hash_cache``: one int32 slot per KV slot of the first local layer's
  sliding-window cache, holding the compressed id (or DEAD) of the token
  last written to that slot. Slots are stable per (request, position) —
  the block table pins a position to a physical slot, prefix-cache hits
  reuse both the physical blocks and the identical token ids, and
  spec-decode rollbacks rewrite the same slots — so lookbacks read back
  exactly what the owning request wrote. Lookback depth (3) is far inside
  the sliding window (128), so window eviction never frees a block a
  live lookback still needs.

  Slots are not part of the KV cache, so KV loaded from another instance
  (P/D, offload connectors) leaves them unwritten. The runner therefore
  passes ``lookback_token_ids``, the ids just before each request's chunk
  start, which take precedence over the slots. The V2 runner reads them
  from its device-resident token history and needs no slot cache; the V1
  runner's CPU token table holds placeholders for generated tokens under
  async scheduling, so it passes prompt positions only and keeps the slot
  cache for the rest.
"""

import ctypes
import mmap
import os
import tempfile
import weakref
from contextlib import ExitStack

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.ops.query_quant import (
    can_fuse_query_quant,
    mxfp8_scale_bytes,
    mxfp8_scale_offset,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# Per-rank token slot from which the Engram DP exchange switches from peer
# reads to all-to-all.
ENGRAM_A2A_MIN_SLOT = 128

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1


def _engram_lookup_thresholds(device: torch.device) -> tuple[int | None, int | None]:
    """Host lookup thresholds on `device`: the rows from which to sort, and the
    tokens (max over DP ranks) from which to run inline. None disables that path."""
    if current_platform.is_cuda() and current_platform.is_device_capability_family(
        100, device.index or 0
    ):
        # Measured on GB200.
        return 36864, 4096
    # TODO: Add hand-tuned thresholds for other GPU families.
    return None, None


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers
    hash differently. Kept odd and bounded so `token_id * multiplier` cannot
    overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
    split over `n_heads` heads. Every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, which keeps the ranges disjoint.
    """

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
        self.compressed_vocab_size: int = config.engram_compressed_vocab_size
        self.pad_token_id: int = config.engram_pad_token_id
        assert len(self.layer_ids) == len(self.num_embeddings)

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, n_hash_cols]

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


@triton.jit(do_not_specialize=["num_tokens"])
def _write_hash_cache_kernel(
    input_ids,
    token_map,
    dead_mask,
    slot_mapping,
    cache,
    num_tokens,
    input_stride,
    mask_stride,
    slot_stride,
    BLOCK_SIZE: tl.constexpr,
    dead_id,
):
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(
        slot_mapping + token_idx * slot_stride, token_idx < num_tokens, other=-1
    ).to(tl.int64)
    valid = (token_idx < num_tokens) & (slot >= 0)
    token = tl.load(input_ids + token_idx * input_stride, valid, other=0)
    value = tl.load(token_map + token, valid, other=0)
    dead = tl.load(dead_mask + token_idx * mask_stride, valid, other=False)
    value = tl.where(dead, dead_id, value)
    tl.store(cache + slot, value, valid)


# Keep request shapes out of the cache key so one warmed variant covers runtime batches.
@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_query_rows",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_query_rows,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # Upper bound in query_start_loc[1:], including repeated padding boundaries.
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(
            query_start_loc + (mid + 1) * query_stride,
            lo < hi,
            other=0,
        )
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(
            input_ids + batch_idx * input_stride, valid & in_batch, other=0
        )
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(
            dead_mask + batch_idx * mask_stride, valid & in_batch, other=False
        )
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask
            + req * lookback_mask_row_stride
            + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        if cache is not None:
            clamped = tl.minimum(
                tl.maximum(lookback, 0), max_blocks * cache_block_size - 1
            )
            block_row = tl.minimum(req, num_table_rows - 1)
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table
                + block_row * table_stride
                + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        source = tl.where(
            in_batch, batch_source, tl.where(known, window_source, fallback)
        ).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * (
                (MAX_NGRAM - 1) * num_heads
            ) + col[None, :]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless on the V2 runner, which supplies every lookback token id. On
    the V1 runner it also keeps `hash_cache`, the slot-keyed rolling store
    of compressed ids (see module docstring), for generated tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.block_size: int = swa_cache_module.block_size
        self.lookback_depth: int = layout.max_ngram_size - 1
        self.use_slot_cache: bool = not vllm_config.use_v2_model_runner
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; "
                "every hash multiplier derives from it, so the engram tables "
                "would be silently rehashed."
            )
        self.pad_id = token_map[layout.pad_token_id]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        logger.info(
            "Built engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )

    def ensure_cache(self) -> bool:
        """Lazily size the slot-keyed cache from the bound SWA KV cache.

        Returns False while the KV cache is unbound (profile run); the caller
        skips engram hashing then. Without the slot cache only that check
        remains.
        """
        kv_cache = self.swa_cache_module.kv_cache
        if kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        # Graph memory profiling binds a temporary, smaller KV cache first.
        # Rebinding must discard its hash history without retaining KV storage.
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def dummy_hashes(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Participate in DP lookups without valid rows or hash-cache updates."""
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        hashes = input_ids.new_full(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads),
            DEAD_ID,
            dtype=torch.int32,
        )
        keep = torch.zeros(num_tokens, dtype=torch.bool, device=input_ids.device)
        return hashes, keep

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache. V2 needs only one launch.
        """
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32
        )
        if num_tokens == 0:
            return output
        if self.use_slot_cache:
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
            # Finish writes before other thread blocks read fallback history.
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                self.token_map,
                dead_mask,
                slot_mapping,
                cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                256,
                DEAD_ID,
            )
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_query_rows=query_start_loc.numel() - 1,
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output


def _engram_head_shard_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Load this rank's complete head buckets. ue8m0 scales arrive as
    float8_e8m0fnu; keep the raw bytes (the param stores uint8)."""
    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    shard = loaded_weight.narrow(0, param.engram_vocab_start, part_rows)
    assert shard.shape == param.shape, (
        f"engram shard {tuple(shard.shape)} does not fit param {tuple(param.shape)}"
    )
    param.data.copy_(shard)


# Branching on SORTED at runtime intermittently crashes Triton 3.8's
# RemoveLayoutConversions.
# TODO: Remove the 3.8 check once Triton ships
# https://github.com/triton-lang/triton/pull/10706.
_SORTED_IS_CONSTEXPR = tl.constexpr(
    triton.__version__.startswith("3.8") and current_platform.is_rocm()
)


@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
        *(() if _SORTED_IS_CONSTEXPR else ("SORTED",)),
    ]
)
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    sorted_dst,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID,
    SORTED: tl.constexpr if _SORTED_IS_CONSTEXPR else None,  # type: ignore[valid-type]
    PACKED: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
):
    """Gather FP8 rows as packed values/scales or dequantized BF16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA. If SORTED,
    `ids` holds rows relative to vocab_start in table order and `sorted_dst`
    their output rows.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        if SORTED:
            local = tl.load(ids + rows, mask=valid, other=-1).to(tl.int64)
            dst = tl.load(sorted_dst + rows, mask=valid, other=0).to(tl.int32)
            owned = (local >= 0) & (local < vocab_end - vocab_start)
        else:
            head = HEAD_START + rows % LOCAL_HEADS
            token = (rows // LOCAL_HEADS).to(tl.int64)
            index = tl.load(
                ids + token * ids_stride_t + head * ids_stride_h,
                mask=valid & (head < TOTAL_HEADS),
                other=-1,
            ).to(tl.int64)
            dst = rows
            owned = valid & (head < TOTAL_HEADS)
            owned &= (index >= vocab_start) & (index < vocab_end)
            local = index - vocab_start
        local = tl.where(owned, local, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        if PACKED:
            width: tl.constexpr = DIM + DIM // QUANT_BLOCK
            dst = dst.to(tl.int64)
            dst = dst // LOCAL_HEADS * OUT_STRIDE + dst % LOCAL_HEADS * width
            tl.store(
                out + dst[:, None] + cols[None, :],
                values.to(tl.uint8, bitcast=True),
                mask=valid[:, None],
            )
            tl.store(
                out + dst[:, None] + DIM + scale_cols[None, :],
                scale,
                mask=valid[:, None] & (cols[None, :] % QUANT_BLOCK == 0),
            )
        else:
            # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
            scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (values.to(tl.float32) * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(
        dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max()
    )


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID
        )
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


class DPSharedEngramStorage:
    """Registered host weights shared by a node-local DP group with one writer."""

    def __init__(
        self, num_rows: int, dim: int, block_size: int, group: GroupCoordinator
    ) -> None:
        self.group = group
        weight_bytes = num_rows * dim
        storage = self._allocate(weight_bytes + weight_bytes // block_size)
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(-1, dim)
        self.weight_scale_inv = storage[weight_bytes:].view(-1, dim // block_size)
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map and register one physical allocation across a node-local DP group."""
        group = self.group
        with ExitStack() as stack:
            path = error = None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_engram_", dir=SHM_PATH)
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            mapping = owner = tensor = None
            try:
                with open(path, "r+b") as file:
                    mapping = mmap.mmap(file.fileno(), num_bytes, flags=mmap.MAP_SHARED)
                owner = np.frombuffer(mapping, dtype=np.uint8)
                pointer = owner.ctypes.data
                tensor = torch.from_numpy(owner)
                result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                if result.value != 0:
                    raise RuntimeError(f"cudaHostRegister failed: {result}")
                finalizer = weakref.finalize(owner, self._unregister, mapping, pointer)
                finalizer.atexit = False  # type: ignore[misc]
                # The UVA helper otherwise allocates a private pinned copy.
                if not tensor.is_pinned():
                    raise RuntimeError(
                        "cudaHostRegister did not pin the shared Engram mapping"
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            # Also fences peer mappings before the leader unlinks the file.
            errors: list[str | None] = [None] * group.world_size
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            if failures:
                # Dropping the owner runs the finalizer; the mapping then unmaps.
                del tensor, owner, mapping
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            assert tensor is not None
            return tensor

    @staticmethod
    def _unregister(mapping: mmap.mmap, pointer: int) -> None:
        # Torch storage retains the numpy owner, including through cached UVA views.
        # Keep its mmap alive until CUDA has released the registration.
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        if result.value != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if self.group.rank_in_group == 0:
            _engram_head_shard_weight_loader(param, loaded_weight)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self.group.cpu_group)

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        if self._views is None:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv),
            )
        return self._views


def engram_table_bytes(layout: EngramLayout, block_size: int = 32) -> int:
    """Host bytes of one full set of Engram tables: fp8 rows plus ue8m0 scales."""
    return sum(layout.num_embeddings) * (
        layout.head_dim + layout.head_dim // block_size
    )


def can_share_engram_tables(layout: EngramLayout, block_size: int = 32) -> bool:
    """Whether co-located DP replicas exist and /dev/shm can hold the full tables."""
    if get_engram_dp_size() == 1:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded tables per rank instead of sharing them."
        )
        return False
    num_bytes = engram_table_bytes(layout, block_size)
    group = get_engram_dp_group()
    assert group is not None
    # Only the leader allocates, so every replica follows its check.
    error = None
    if group.rank_in_group == 0:
        if not os.path.isdir(SHM_PATH):
            error = f"{SHM_PATH} is not mounted"
        else:
            try:
                check_shm_free_space(num_bytes, allocation_name="Engram tables")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    error = group.broadcast_object(error)
    if error is not None:
        logger.warning_once(
            "Sharding the offloaded Engram tables across DP replicas: %s", error
        )
    return error is None


def _allocate_huge_page_storage(num_bytes: int) -> torch.Tensor | None:
    """Register prefaulted huge pages, or return None for pinned-memory fallback."""
    try:
        mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mapping.madvise(mmap.MADV_HUGEPAGE)
    except OSError as exc:
        logger.warning(
            "Engram huge-page allocation failed (%s); using pinned memory.", exc
        )
        return None

    owner = np.frombuffer(mapping, dtype=np.uint8)
    # Fault the pages in before CUDA pins them, so they can be huge.
    owner[:: mmap.PAGESIZE] = 0
    tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    if result.value != 0:
        logger.warning(
            "Engram cudaHostRegister failed (%s); using pinned memory.", result
        )
        return None
    # Tensor views retain owner; its finalizer retains mapping until unregister.
    finalizer = weakref.finalize(
        owner, DPSharedEngramStorage._unregister, mapping, pointer
    )
    finalizer.atexit = False  # type: ignore[misc]
    if not tensor.is_pinned():
        logger.warning(
            "CUDA did not recognize Engram registration; using pinned memory."
        )
        return None
    return tensor


class ParallelEngramEmbedding(nn.Module):
    """Hash heads with FP8 rows and per-block E8M0 scales.

    Heads are TP-sharded, and additionally DP-sharded across a node-local
    group unless the table lives in (DP-shared) pinned host memory.
    """

    _weight_loader = staticmethod(_engram_head_shard_weight_loader)
    _shared_memory: DPSharedEngramStorage | None = None
    _packed: torch.Tensor | None = None

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        use_thp: bool = False,
    ) -> None:
        super().__init__()
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.use_thp = use_thp
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        num_shards, head_rank = self._get_shard_info()
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, num_shards)
        # TODO: Support row-wise sharding when there are too few hash heads.
        assert (num_shards - 1) * self.part_n_hash_cols < self.n_hash_cols, (
            f"Engram sharding leaves ranks without hash heads: "
            f"{self.n_hash_cols} heads over {num_shards} shards"
        )
        self.head_start = head_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self._num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count
        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": self._weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )
        if cpu_offload:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            # The ue8m0 encoding of scale 1.0 is exponent byte 127.
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB %s",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
                "shared across DP replicas" if dp_shared_memory else "per rank",
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return self.tp_size, get_tensor_model_parallel_rank()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = DPSharedEngramStorage(
                self.part_num_embeddings, self.dim, self.block_size, group
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        scale_dim = self.dim // self.block_size
        if not self.cpu_offload:
            return (
                torch.empty(
                    self.part_num_embeddings, self.dim, dtype=torch.float8_e4m3fn
                ),
                torch.empty(self.part_num_embeddings, scale_dim, dtype=torch.uint8),
            )
        if self.use_thp:
            weight_bytes = self.part_num_embeddings * self.dim
            packed = _allocate_huge_page_storage(
                weight_bytes + weight_bytes // self.block_size
            )
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(torch.float8_e4m3fn).view(-1, self.dim),
                    packed[weight_bytes:].view(-1, scale_dim),
                )
        # Model initialization may be inside a CUDA device context.
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.part_num_embeddings,
                scale_dim,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            ),
        )

    def collapse_huge_pages(self) -> None:
        """Best-effort MADV_COLLAPSE (Linux >= 6.1) of pages that faulted small."""
        if self._packed is None:
            return
        addr, num_bytes = self._packed.data_ptr(), self._packed.nbytes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        if libc.madvise(addr, num_bytes, 25) != 0:
            logger.warning(
                "Engram MADV_COLLAPSE failed; keeping existing pages: %s",
                os.strerror(ctypes.get_errno()),
            )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self.weight_scale_inv)
        if not self.cpu_offload:
            return self.weight.data, self.weight_scale_inv.data
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight.data),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads into BF16, or packed FP8 bytes followed by scales.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()
        ids_stride_t, ids_stride_h = indices.stride()
        sort_min_rows, _ = _engram_lookup_thresholds(indices.device)
        sort = (
            self.cpu_offload
            and self._packed is None
            and sort_min_rows is not None
            and rows >= sort_min_rows
        )
        dst = indices
        if sort:
            # Small-page host tables miss the TLB per row; in table order, rows on
            # one page share it.
            head_end = self.head_start + self.part_n_hash_cols
            local = indices[:, self.head_start : head_end] - self.vocab_start_idx
            if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
                local = nn.functional.pad(local, (0, pad), value=-1)
            indices, dst = local.flatten().sort()
            dst = dst.int()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            weight,
            scales,
            indices,
            dst,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            ids_stride_t,
            ids_stride_h,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
            SORTED=sort,
            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all shards for this replica's tokens."""
        num_tokens = indices.shape[0]
        if self.dp_size > 1:
            indices = gather_engram_hashes(indices)
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        if self.dp_size > 1:
            out = _gather_engram_rows(out, num_tokens)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _fused_engram_post_wkv_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = program_idx // HC_MULT
    hc_idx = program_idx % HC_MULT
    token_idx = token_idx.to(tl.int64)
    source_idx = token_idx
    source_valid = source_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv + source_idx * kv_stride_t + (hc_idx * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + dim_offsets * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + dim_offsets * k_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + source_idx * mask_stride,
            mask=source_valid,
            other=0,
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        kv + source_idx * kv_stride_t + (HC_MULT * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_select_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(
        gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0
    )
    tl.store(output + offsets, values, offsets < num_elements)


def _engram_select_rows(
    gathered: torch.Tensor,
    output: torch.Tensor,
    source_tokens: int,
    token_start: int,
    local_width: int,
) -> None:
    """Copy one token window out of a rank-major gathered buffer.

    Both gathers land rank-major (`[rank][token][local width]`); this walks the
    window the rank keeps and lays its ranks out side by side as width.
    """
    if output.numel() == 0:
        return
    _engram_select_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        gathered,
        output,
        source_tokens,
        token_start,
        output.numel(),
        local_width,
        output.shape[1] * output.shape[2],
        BLOCK_SIZE=1024,
    )


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows


@triton.jit(do_not_specialize=["num_tokens"])
def _engram_unpack_fp8_kernel(
    packed,
    values,
    scales,
    num_tokens,
    values_stride,
    scales_stride,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    LAYERS: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    PEER: tl.constexpr,
):
    """Split `[rank][token][layer][local head]` packed rows into MXFP8 values and
    F8_128x4 scales, one layer per grid column; `packed` holds per-rank
    pointers if PEER. Scales past `num_tokens` zero-pad the 128-row tile.
    """
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    layer = tl.program_id(1).to(tl.int64)
    token, head = row // HEADS, row % HEADS
    if PEER:
        src = tl.load(packed + head // LOCAL_HEADS)
        src = tl.multiple_of(src, DIM // 32).to(tl.pointer_type(tl.uint8))
        source = RANK * num_tokens + token
    else:
        src = packed
        source = head // LOCAL_HEADS * num_tokens + token
    source = (source * LAYERS + layer) * LOCAL_HEADS + head % LOCAL_HEADS
    # Packed rows are only scale-width aligned.
    src += tl.multiple_of(source * (DIM + DIM // 32), DIM // 32)
    valid = token < num_tokens
    cols = tl.arange(0, DIM)
    data = tl.load(src[:, None] + cols[None, :], mask=valid[:, None], other=0)
    values += layer * values_stride + row * DIM
    tl.store(values[:, None] + cols[None, :], data, mask=valid[:, None])
    scale_cols = tl.arange(0, DIM // 32)
    scale = tl.load(
        src[:, None] + DIM + scale_cols[None, :], mask=valid[:, None], other=0
    )
    group = head[:, None] * (DIM // 32) + scale_cols[None, :]
    offset = mxfp8_scale_offset(token[:, None], group, HEADS * DIM // 32)
    tl.store(scales + layer * scales_stride + offset, scale)


class EngramBatch(nn.Module):
    """Stage every Engram layer's WKV input as MXFP8 with one exchange.

    A module so the model owns the peer exchange's pointer buffers.
    """

    def __init__(self, layers: list["Engram"], max_tokens: int) -> None:
        super().__init__()
        first = layers[0].layer_hash_index
        hash_indices = [layer.layer_hash_index for layer in layers]
        assert hash_indices == list(range(first, first + len(layers)))
        self.layers = layers
        table = layers[0].embed_tokens
        device = layers[0].staged_rows.device
        self.packed_rows = torch.empty(
            max_tokens * table.dp_size,
            len(layers),
            table.part_n_hash_cols,
            table.dim + table.dim // 32,
            dtype=torch.uint8,
            device=device,
        )
        width = table.n_hash_cols * table.dim
        self.fp8_values = torch.empty(
            len(layers), max_tokens, width, dtype=torch.uint8, device=device
        )
        self.fp8_scales = torch.empty(
            len(layers),
            mxfp8_scale_bytes(max_tokens, width),
            dtype=torch.uint8,
            device=device,
        )
        max_peer_slot = min(max_tokens, ENGRAM_A2A_MIN_SLOT - 1)
        self.peer_exchange = None
        if table.cpu_offload:
            from vllm.models.deepseek_v41.nvidia.ops.engram_peer import (
                EngramPeerExchange,
            )

            self.peer_exchange = EngramPeerExchange.create(layers, max_peer_slot)
        for layer in layers:
            # Owned by the model; keep it out of each layer's submodules.
            object.__setattr__(layer, "_batch", self)
            # The layers read this batch's staging instead of their BF16 rows.
            layer.staged_rows = layer.staged_rows.new_empty(
                (0, *layer.staged_rows.shape[1:])
            )

    @classmethod
    def create(cls, layers: list["Engram"], max_tokens: int) -> "EngramBatch | None":
        """Batch DP-sharded tables whose WKV takes FlashInfer MXFP8 input."""
        if not layers:
            return None
        table = layers[0].embed_tokens
        if (
            table.dp_size == 1
            or table.tp_size > 1
            or table.dim % 128
            or get_current_vllm_config().lora_config is not None
            or not can_fuse_query_quant([layer.wkv for layer in layers])
        ):
            return None
        return cls(layers, max_tokens)

    def prepare_embeddings(self, hashes: torch.Tensor) -> None:
        """Stage every layer's WKV input before the decoder layers.

        DP-sharded tables decide on the group's slot, so every rank agrees.
        """
        slot = engram_gathered_num_tokens()
        peer = self.peer_exchange is not None and slot < ENGRAM_A2A_MIN_SLOT
        self._stage(hashes if peer else gather_engram_hashes(hashes), slot, peer)

    @eager_break_during_capture
    def _stage(self, hashes: torch.Tensor, slot: int, peer: bool) -> None:
        if slot == 0:
            return
        if peer:
            assert self.peer_exchange is not None
            rows = self.peer_exchange.exchange(hashes, slot)
            rank = self.peer_exchange.dp_rank
        else:
            rows, rank = self.packed_rows[: hashes.shape[0]], 0
            for index, layer in enumerate(self.layers):
                layer.embed_tokens.lookup(
                    hashes[:, layer.layer_hash_index], rows[:, index]
                )
            group = get_engram_dp_group()
            assert group is not None
            sent, rows = rows, torch.empty_like(rows)
            torch.distributed.all_to_all_single(rows, sent, group=group.device_group)
        table = self.layers[0].embed_tokens
        heads = table.n_hash_cols
        grid = (triton.cdiv(slot, 128) * 128 * heads // 16, len(self.layers))
        _engram_unpack_fp8_kernel[grid](
            rows,
            self.fp8_values,
            self.fp8_scales,
            slot,
            self.fp8_values.stride(0),
            self.fp8_scales.stride(0),
            HEADS=heads,
            LOCAL_HEADS=table.part_n_hash_cols,
            LAYERS=len(self.layers),
            RANK=rank,
            DIM=table.dim,
            BLOCK=16,
            PEER=peer,
        )

    def wkv_input(self, layer: "Engram", num_tokens: int) -> QuantizedActivation:
        index = layer.layer_hash_index - self.layers[0].layer_hash_index
        values = self.fp8_values[index, :num_tokens].view(torch.float8_e4m3fn)
        return QuantizedActivation(
            values,
            self.fp8_scales[index, : mxfp8_scale_bytes(num_tokens, values.shape[1])],
            torch.bfloat16,
            values.shape,
            kMxfp8Dynamic,
        )


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it
    matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per
    hc copy plus a shared value. The gate is a normalized dot product of the
    stream against the key, signed-sqrt'ed before the sigmoid (matching the
    training kernel).
    """

    _prefetch_stream: torch.cuda.Stream | None = None
    _prefetch_done: torch.cuda.Event | None = None
    _batch: EngramBatch | None = None

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        # Layers sharing one stream serialize their offloaded lookups, so they
        # take turns instead of jointly starving decoder compute of SMs.
        self._prefetch_stream = prefetch_stream
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.use_sequence_parallel = use_sequence_parallel

        # Named ``embed_tokens`` so the checkpoint's ``engram.embed.weight``
        # survives the mapper's ``embed.weight`` -> ``embed_tokens.weight``
        # suffix rule.
        self.embed_tokens = self._create_embedding(layout, layer_hash_index)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        # Without sequence parallelism every TP rank holds every token, so
        # shard the output columns instead of replicating the projection.
        self.wkv = ColumnParallelLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
            disable_tp=use_sequence_parallel,
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        self._init_staging(max_tokens, layout.head_dim)

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        return ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=bool(engram_config.dp_shared_memory),
            use_thp=engram_config.use_thp,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        # Persistent storage keeps lookup addresses stable across graph replays.
        self.staged_rows = torch.empty(
            max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols,
            head_dim,
            dtype=torch.bfloat16,
        )
        if not self.embed_tokens.cpu_offload:
            self._prefetch_stream = None
            return
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.staged_rows.device)
        self._prefetch_done = torch.cuda.Event()

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Look up, or prefetch from host memory, rows before the decoder layers
        consume them; DP-sharded tables take the group's gathered hash IDs."""
        rows = self.staged_rows[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        if self._prefetch_stream is None:
            self.embed_tokens.lookup(hash_ids, rows)
        else:
            self._start_prefetch(hash_ids, rows)

    @eager_break_during_capture
    def _start_prefetch(self, hash_ids: torch.Tensor, rows: torch.Tensor) -> None:
        # Eager boundaries let the lookup span piecewise graph segments, and
        # decide per replay (on the replay stream) whether it runs inline.
        num_tokens = hash_ids.shape[0]
        # All DP ranks must agree, or EP collectives wait on the slowest one.
        if is_forward_context_available() and (dp := get_forward_context().dp_metadata):
            num_tokens = int(dp.num_tokens_across_dp_cpu.max())
        current = stream = torch.cuda.current_stream()
        # Big lookups stall persistent main-stream kernels anyway: run them inline.
        _, inline_min_tokens = _engram_lookup_thresholds(hash_ids.device)
        if inline_min_tokens is not None and num_tokens >= inline_min_tokens:
            self.embed_tokens.lookup(hash_ids, rows)
        else:
            stream = self._prefetch_stream
            assert stream is not None
            stream.wait_stream(current)
            # Keep temporary hash storage alive until lookup finishes reading it.
            hash_ids.record_stream(stream)
            with torch.cuda.stream(stream):
                self.embed_tokens.lookup(hash_ids, rows, background=True)
        assert self._prefetch_done is not None
        self._prefetch_done.record(stream)

    @eager_break_during_capture
    def _finish_prefetch(self, event: torch.cuda.Event) -> None:
        # Wait for this layer's lookup only, not for later layers on the stream.
        torch.cuda.current_stream().wait_event(event)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_done is not None:
            self._finish_prefetch(self._prefetch_done)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return self.staged_rows[:num_tokens]

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        rows = self._ready_rows(hash_ids.shape[0])
        if self.embed_tokens.tp_size == 1:
            return rows[:, : self.embed_tokens.n_hash_cols]
        if self.use_sequence_parallel:
            tp_size = self.embed_tokens.tp_size
            num_tokens, local_heads, dim = rows.shape
            gathered = tensor_model_parallel_all_gather(rows, dim=0)
            chunk = (num_tokens + tp_size - 1) // tp_size
            out = rows.new_empty((chunk, self.embed_tokens.n_hash_cols, dim))
            _engram_select_rows(
                gathered,
                out,
                num_tokens,
                get_tensor_model_parallel_rank() * chunk,
                local_heads * dim,
            )
            return out
        rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, dim]; hash_ids: [T, n_hash_cols] (all
        tokens, pre sequence-parallel shard); token_mask: [T], False shuts
        the gate so those positions pass through untouched."""
        if self._batch is not None:
            kv = self.wkv(self._batch.wkv_input(self, hash_ids.shape[0]))
        else:
            kv = self.wkv(self.embed(hash_ids).flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
        assert token_mask is None or token_mask.shape == (num_kv_tokens,)
        if self.use_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = (num_kv_tokens + tp_size - 1) // tp_size
            assert hidden_states.shape[0] == shard_size
            start = min(tp_rank * shard_size, num_kv_tokens)
            num_kv_tokens = min(shard_size, num_kv_tokens - start)
            if token_mask is not None:
                token_mask = token_mask[start : start + num_kv_tokens]

        num_tokens, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = token_mask if token_mask is not None else hidden_states
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output
