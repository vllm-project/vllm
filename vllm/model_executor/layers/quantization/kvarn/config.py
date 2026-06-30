# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN configuration."""

import math
import os
from dataclasses import dataclass

# Named KVarN presets: each maps to a frozen set of config parameters.
# The trailing g<N> encodes the variance-normalization tile size, which must
# equal the vLLM block size. g128 is the current design point; g64 trades a
# little compression (more per-tile scale overhead per token) for finer
# quantization granularity (each tile's scales adapt to fewer tokens).
#
# Bit-width is fully parameterized in the quantizer and kernels (key_bits /
# value_bits), and the tile size flows through cfg.group everywhere (storage
# layout, Triton GROUP constexpr, flush / slot math), so additional presets are
# a one-line addition here. Keys carry more quantization sensitivity than values
# (key error propagates through the softmax exponentials, value error is averaged
# out by the softmax weights), so the shipped preset spends more bits on keys.
KVARN_PRESETS: dict[str, dict] = {
    "kvarn_k4v2_g128": {"key_bits": 4, "value_bits": 2, "group": 128},
    "kvarn_k4v4_g128": {"key_bits": 4, "value_bits": 4, "group": 128},
    "kvarn_k4v2_g64": {"key_bits": 4, "value_bits": 2, "group": 64},
    "kvarn_k4v4_g64": {"key_bits": 4, "value_bits": 4, "group": 64},
}


@dataclass
class KVarNConfig:
    """Configuration for KVarN KV-cache quantization.

    Pipeline per (block, head):
      1. Hadamard rotation along head_dim (orthonormal, applied via external GEMM).
      2. Iterative log-domain variance-normalization (Sinkhorn-like) over the
         [D, group] tile for K (per-channel orientation) and [group, D] tile for
         V (per-token orientation).
      3. Asymmetric per-row RTN at `key_bits` / `value_bits`.
      4. Absorb the per-row RTN scale and zero-point into the matching
         sinkhorn scale axis (K: into per-channel; V: into per-token-in-tile).
         Reconstruction: ``x = (q * absorbed_scale + absorbed_zp) * other_scale``.

    Cache layout (per (block, head)) is a single packed record — see the
    backend's `get_kv_cache_shape` override. There is no per-token slot
    because the scales are tile-shared; the block boundary IS the tile.

    Args:
        head_dim: Attention head dimension (power of 2; tested at 128).
        key_bits: Bits per key element (default 4).
        value_bits: Bits per value element (default 4).
        group: KVarN tile size in tokens. Must equal vLLM block_size so that
            one vLLM block = one KVarN tile per head.
        sinkhorn_iters: Iterations of the alternating column/row std-norm in
            the variance-normalization loop (default 8; lossless vs 16).
        boundary_skip_layers: Number of leading / trailing transformer layers
            to keep in fp16 (KVarN's sink/residual analogue). Default 2 mirrors
            TurboQuant's default.
    """

    head_dim: int = 128
    key_bits: int = 4
    value_bits: int = 4
    group: int = 128
    # converges by ~4 iters; 8 lossless vs 16 (validated Qwen3-4B + Qwen3.6-27B AIME)
    sinkhorn_iters: int = 8
    sink_tokens: int = 128  # first N tokens per request stay fp16 (NEVER quantised)
    boundary_skip_layers: int = (
        0  # layer-level skipping off by default; sink_tokens replaces it
    )

    # ── derived: storage layout ──────────────────────────────────────────────
    @property
    def k_packed_bytes(self) -> int:
        """Packed bytes for one K tile per head: D * group * key_bits / 8."""
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        """Packed bytes for one V tile per head: group * D * value_bits / 8."""
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_scale_bytes(self) -> int:
        """fp16 bytes for K scales: s_col_K' [D] + zp_K' [D] + s_row_K [group].

        s_col_K' = rtn_scale ⊙ s_chan_sinkhorn  (per-channel absorbed scale)
        zp_K'    = rtn_zp    ⊙ s_chan_sinkhorn  (per-channel absorbed zero)
        s_row_K  = s_tok_sinkhorn               (per-token-in-tile)
        """
        return (2 * self.head_dim + self.group) * 2

    @property
    def v_scale_bytes(self) -> int:
        """fp16 bytes for V scales: s_col_V [D] + s_row_V' [group] + zp_V' [group].

        s_col_V  = s_chan_sinkhorn              (per-channel, untouched)
        s_row_V' = rtn_scale ⊙ s_tok_sinkhorn   (per-token-in-tile absorbed scale)
        zp_V'    = rtn_zp    ⊙ s_tok_sinkhorn   (per-token-in-tile absorbed zero)
        """
        return (self.head_dim + 2 * self.group) * 2

    @property
    def tile_bytes(self) -> int:
        """Total packed bytes per (block, head): K + V combined."""
        return (
            self.k_packed_bytes
            + self.k_scale_bytes
            + self.v_packed_bytes
            + self.v_scale_bytes
        )

    @property
    def tile_bytes_aligned(self) -> int:
        """tile_bytes rounded up for nicer Triton loads.

        For head_dim >= 256 we round the PER-TOKEN slot (tile_bytes / group) up to
        a power of 2. This is required for models with heterogeneous head_dim
        (e.g. Gemma-4: 256 sliding-window layers + 512 global layers): the raw
        slot has a fixed per-token-group scale term that doesn't scale with D, so
        slot(512)/slot(256) is not an integer and vLLM's KV-cache page-size
        unification (which scales block_size by that ratio) fails. Power-of-2 slots
        make the ratio an exact power of 2. head_dim<=128 keeps the tight 8-byte
        alignment (the common case; no padding). Trailing pad only — offsets are
        unchanged, so the layout/kernels are byte-compatible."""
        if self.head_dim >= 256:
            slot = math.ceil(self.tile_bytes / self.group)
            slot_pow2 = 1 << (slot - 1).bit_length()
            return slot_pow2 * self.group
        return ((self.tile_bytes + 7) // 8) * 8

    # ── slot byte offsets within one tile (used by the kernels) ──────────────
    @property
    def k_packed_offset(self) -> int:
        return 0

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_offset + self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    # ── fp16 tail-pool sizing ────────────────────────────────────────────────
    # KVarN keeps a fixed-size fp16 side buffer ("tail pool") because a tile
    # cannot be quantized until its `group` tokens all exist. Per active request,
    # per layer, it holds two fp16 blocks: the permanent attention-sink block and
    # the in-progress tail. The pool must be pre-allocated at a fixed size (CUDA
    # graphs), so its size bounds how many requests can run concurrently. Rather
    # than size the pool to an arbitrary `max_num_seqs` (which can OOM at large
    # values or exhaust if under-sized), we pick a memory budget and cap the
    # scheduler's concurrency to what that budget supports — see
    # `max_supported_seqs` and the platform's check_and_update_config.
    # The pool and the paged KV cache draw from the SAME pot: the memory left
    # after model weights (i.e. `gpu_memory_utilization · total − weights`).
    # Sizing the pool as a fixed fraction of *total* GPU memory was the bug
    # behind the concurrency cap: on a 4B/24GB card the pool got 0.08·24≈1.9 GB and
    # concurrency capped to ~30 while the KV cache sat at ~3% utilization —
    # ~10 GB of usable memory wasted. We instead give the pool a share of the
    # post-weight usable envelope (POOL_USABLE_SHARE), which auto-scales: a small
    # model on a big card gets a large pool (high concurrency), a model that
    # nearly fills the card gets a small one (degrades to cap≈1, never OOMs).
    # The legacy fraction-of-total path is kept as a fallback for when the weight
    # size can't be read. Both are tunable via KVARN_POOL_MEM_FRAC (interpreted
    # as share-of-usable when weights are known, else fraction-of-total).
    POOL_MEM_FRAC_DEFAULT = 0.08  # legacy: fraction of TOTAL (fallback)
    POOL_USABLE_SHARE_DEFAULT = 0.5  # share of (util·total − weights)

    def _slot_bytes_per_layer(self, num_kv_heads: int) -> int:
        """Bytes for one pool slot in one layer: group·heads·head_dim fp16,
        for K and V combined (2 bytes/elem × 2 tensors = 4)."""
        return self.group * num_kv_heads * self.head_dim * 4

    def pool_slots(self, max_num_seqs: int, max_num_batched_tokens: int) -> int:
        """Structural peak of fp16 pool slots needed in a single step:
        sink + in-progress tail per active request (2·max_num_seqs), plus the
        full blocks a chunked prefill can touch before flushing, plus headroom.
        With concurrency capped (see max_supported_seqs) this fits the budget."""
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        # Floor/headroom kept small: at large head_dim·heads·layers (e.g. Gemma-4
        # 512·16·60 => ~251 MB/slot/layer) a big floor like 64 reserves tens of GB
        # and leaves no room for the KV cache. The real peak is sink+tail per seq
        # (2·S) plus the blocks an in-flight prefill touches before it flushes.
        return max(2 * max_num_seqs + prefill_blocks + 8, 8)

    def pool_budget_bytes(
        self,
        total_gpu_bytes: int,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        """GPU bytes the fp16 tail pool is allowed to occupy.

        Preferred (weight-aware): a share of the post-weight usable envelope,
        ``share · (gpu_memory_utilization · total − weight_bytes)``. This is the
        memory the pool and the paged KV cache actually compete for, so the
        budget tracks real headroom instead of an arbitrary slice of the whole
        card. ``share`` comes from KVARN_POOL_MEM_FRAC or
        POOL_USABLE_SHARE_DEFAULT.

        Fallback (weights unknown): the legacy ``frac · total`` with
        POOL_MEM_FRAC_DEFAULT, so behaviour is unchanged when we cannot read the
        weight size."""
        env = os.environ.get("KVARN_POOL_MEM_FRAC")
        if weight_bytes is not None and gpu_memory_utilization is not None:
            share = float(env) if env is not None else self.POOL_USABLE_SHARE_DEFAULT
            usable = gpu_memory_utilization * total_gpu_bytes - weight_bytes
            return max(0, int(share * usable))
        frac = float(env) if env is not None else self.POOL_MEM_FRAC_DEFAULT
        return int(total_gpu_bytes * frac)

    def max_supported_seqs(
        self,
        total_gpu_bytes: int,
        num_kv_heads: int,
        num_layers: int,
        max_num_batched_tokens: int,
        frac: float | None = None,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        """Largest max_num_seqs whose pool fits the pool budget.

        Inverts `pool_slots`: max_slots = budget / (slot_bytes · layers), then
        solve 2·S + prefill + 8 ≤ max_slots for S. Always ≥ 1. The budget is
        weight-aware when `weight_bytes`/`gpu_memory_utilization` are supplied
        (see `pool_budget_bytes`); `frac`, if given, forces the legacy
        fraction-of-total path."""
        if frac is not None:
            budget = int(total_gpu_bytes * frac)
        else:
            budget = self.pool_budget_bytes(
                total_gpu_bytes, gpu_memory_utilization, weight_bytes
            )
        slot_bytes = self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)
        max_slots = int(budget / slot_bytes)
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(1, (max_slots - prefill_blocks - 8) // 2)

    def pool_bytes(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_kv_heads: int,
        num_layers: int,
    ) -> int:
        """Total GPU bytes the fp16 tail pool occupies (this rank): pool_slots
        summed over every layer. Reserved up front in the worker so the lazy
        pool allocation never pushes past the KV-memory limit."""
        slots = self.pool_slots(max_num_seqs, max_num_batched_tokens)
        return slots * self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)

    @staticmethod
    def num_kvarn_layers(model_config, parallel_config) -> int:
        """Number of layers the KVarN fp16 tail pool actually spans = the
        full-attention layers. On a hybrid model (Qwen3.5/3.6, Jamba, ...) the
        Mamba/linear-attention layers have no KVarN pool, so sizing the pool by
        ALL layers over-reserves it ~Nx and starves the Mamba/KV caches (OOM or
        cap collapse). For a dense transformer this equals total layers, so the
        dense path is unchanged. Falls back to total layers if the per-type
        count is unavailable."""
        try:
            n = model_config.get_num_layers_by_block_type(parallel_config, "attention")
            if n and n > 0:
                return n
        except Exception:
            pass
        return model_config.get_num_layers(parallel_config)

    @staticmethod
    def estimate_weight_bytes(model: str, tensor_parallel_size: int = 1) -> int | None:
        """Best-effort per-rank model weight size in bytes, read from the
        checkpoint files on disk (exact, and cheap, with no CUDA context, which
        the early `check_and_update_config` hook must avoid). Returns None if the
        files can't be located, so the caller falls back to the legacy budget.

        Resolves a local directory directly, or the local HF cache snapshot for
        a repo id (never downloads). Prefers the shards named in a
        `*.safetensors.index.json` (or `*.bin.index.json`) manifest, which is
        exactly the set the loader reads. This avoids double-counting a repo that
        ships both a single consolidated checkpoint and the sharded HF set (e.g.
        Mistral-7B-Instruct-v0.3 carries `consolidated.safetensors` alongside
        `model-0000n-of-0000m.safetensors`, which a plain glob sums to ~2x the
        real weight size). Divides by the tensor-parallel degree (weights shard
        ~evenly across ranks)."""
        import glob as _glob
        import json as _json

        try:
            d = model
            if not os.path.isdir(d):
                # Repo id: resolve the already-cached snapshot, if any.
                try:
                    from huggingface_hub import snapshot_download

                    d = snapshot_download(model, local_files_only=True)
                except Exception:
                    return None

            # 1) Prefer the loader's own manifest: sum only the shards it lists,
            #    so a stray consolidated/single-file copy is not double-counted.
            for ext in ("safetensors", "bin"):
                indexes = _glob.glob(
                    os.path.join(d, "**", f"*.{ext}.index.json"), recursive=True
                )
                if not indexes:
                    continue
                try:
                    with open(indexes[0]) as fh:
                        weight_map = _json.load(fh).get("weight_map", {})
                    base = os.path.dirname(indexes[0])
                    names = sorted(set(weight_map.values()))
                    shards = [os.path.join(base, s) for s in names]
                    # Trust the manifest only when every listed shard is on
                    # disk: a partial set would under-estimate the weights and
                    # over-grow the pool budget, so fall through to the
                    # conservative glob instead.
                    if names and all(os.path.exists(p) for p in shards):
                        total = sum(os.path.getsize(p) for p in shards)
                        if total > 0:
                            return total // max(tensor_parallel_size, 1)
                except Exception:
                    pass  # fall through to the single-file / glob paths

            # 2) No usable manifest: prefer a canonical single-file checkpoint.
            for single in ("model.safetensors", "consolidated.safetensors"):
                p = os.path.join(d, single)
                if os.path.exists(p):
                    total = os.path.getsize(p)
                    if total > 0:
                        return total // max(tensor_parallel_size, 1)

            # 3) Fallback: sum whatever weight shards are present.
            files = _glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)
            if not files:
                files = _glob.glob(os.path.join(d, "**", "*.bin"), recursive=True)
            if not files:
                return None
            total = sum(os.path.getsize(f) for f in files)
            if total <= 0:
                return None
            return total // max(tensor_parallel_size, 1)
        except Exception:
            return None

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:
        """First-N + last-N transformer layer indices as strings, suitable
        for vLLM's ``kv_cache_dtype_skip_layers``. Mirrors TurboQuant
        (`TurboQuantConfig.get_boundary_skip_layers`)."""
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        return [str(i) for i in sorted(set(first + last))]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "KVarNConfig":
        """Create a config from a preset string like ``"kvarn_k4v4"``."""
        if cache_dtype not in KVARN_PRESETS:
            valid = ", ".join(KVARN_PRESETS.keys())
            raise ValueError(
                f"Unknown KVarN cache dtype: {cache_dtype!r}. Valid: {valid}"
            )
        preset = KVARN_PRESETS[cache_dtype]
        # Optional env override for Sinkhorn iteration count (KVARN_SINKHORN_ITERS).
        # Default 16 mirrors the paper; useful for testing convergence at large
        # model scale (e.g. 48-layer 30B-A3B-Thinking-2507 may benefit from more).
        iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "8"))
        sink_tokens = int(os.environ.get("KVARN_SINK_TOKENS", "128"))
        return KVarNConfig(
            head_dim=head_dim,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            group=preset["group"],
            sinkhorn_iters=iters,
            sink_tokens=sink_tokens,
        )
