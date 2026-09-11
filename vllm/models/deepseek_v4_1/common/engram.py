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

import weakref

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1


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


@triton.jit
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
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
    GRID: tl.constexpr,
):
    """Gather fp8 rows, apply their ue8m0 block scales, write bf16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
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
        # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded by complete hash heads over TP ranks.
    Rows stay fp8 and are dequantized with ue8m0 per-32 scales on lookup.

    With `cpu_offload` the shard lives in pinned host memory and is read over
    UVA instead of HBM; the TP sharding is unchanged either way.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size = tp_size
        self.cpu_offload = cpu_offload
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        self._num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, which would otherwise put the shard in HBM.
        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.weight = nn.Parameter(
            torch.empty(
                self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                dim // block_size,
                dtype=torch.uint8,
                **kwargs,
            ),
            requires_grad=False,
        )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )
        if cpu_offload:
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB per rank",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
            )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters when resident, else cached UVA views of the pinned shard.

        Rebuilt if anything swaps `.data`, so a stale device pointer cannot
        survive silently.
        """
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
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all TP shards."""
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
            out = out[:, : self.n_hash_cols]
        return out


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
def _engram_sp_rows_kernel(
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


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it
    matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per
    hc copy plus a shared value. The gate is a normalized dot product of the
    stream against the key, signed-sqrt'ed before the sigmoid (matching the
    training kernel).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.use_sequence_parallel = use_sequence_parallel

        # Named ``embed_tokens`` so the checkpoint's ``engram.embed.weight``
        # survives the mapper's ``embed.weight`` -> ``embed_tokens.weight``
        # suffix rule.
        engram_config = get_current_vllm_config().engram_config
        self.embed_tokens = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload if engram_config else True,
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
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
        # Keep lookup results alive across breakable graph segments.
        self.staged_rows = torch.empty(
            max_tokens,
            self.embed_tokens.part_n_hash_cols,
            layout.head_dim,
            dtype=torch.bfloat16,
        )
        parallel_config = get_current_vllm_config().parallel_config
        self._init_lookup_staging(
            parallel_config.num_ubatches if parallel_config.use_ubatching else 1,
        )

    def _init_lookup_staging(self, num_slots: int) -> None:
        self._lookup_rows = [self.staged_rows] + [
            torch.empty_like(self.staged_rows) for _ in range(num_slots - 1)
        ]

    def _lookup_slot(self) -> int:
        rows = getattr(self, "_lookup_rows", None)
        return dbo_current_ubatch_id() if rows is not None and len(rows) > 1 else 0

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Stage lookup rows in the current microbatch's own buffer."""
        buffers = getattr(self, "_lookup_rows", None)
        rows = (
            buffers[self._lookup_slot()] if buffers is not None else self.staged_rows
        )[: hash_ids.shape[0]]
        self.embed_tokens.lookup(hash_ids, rows)
        return rows

    def embed(
        self, hash_ids: torch.Tensor, prepared_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        if prepared_rows is None:
            buffers = getattr(self, "_lookup_rows", None)
            prepared_rows = (
                buffers[self._lookup_slot()]
                if buffers is not None
                else self.staged_rows
            )
        rows = prepared_rows[: hash_ids.shape[0]]
        if self.embed_tokens.tp_size == 1:
            return rows
        if self.use_sequence_parallel:
            tp_size = self.embed_tokens.tp_size
            num_tokens, local_heads, dim = rows.shape
            gathered = tensor_model_parallel_all_gather(rows, dim=0)
            chunk = (num_tokens + tp_size - 1) // tp_size
            rows = rows.new_empty((chunk, self.embed_tokens.n_hash_cols, dim))
            _engram_sp_rows_kernel[(triton.cdiv(rows.numel(), 1024),)](
                gathered,
                rows,
                num_tokens,
                get_tensor_model_parallel_rank() * chunk,
                rows.numel(),
                local_heads * dim,
                self.embed_tokens.n_hash_cols * dim,
                BLOCK_SIZE=1024,
            )
            return rows
        rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        prepared_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, dim]; hash_ids: [T, n_hash_cols] (all
        tokens, pre sequence-parallel shard); token_mask: [T], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids, prepared_rows).flatten(-2))
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
