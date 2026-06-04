# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN configuration."""

import math
import os
from dataclasses import dataclass

# Named KVarN presets: each maps to a frozen set of config parameters.
# The trailing g<N> encodes the variance-normalization tile size, which must
# equal the vLLM block size. g128 is the current design point.
#
# Bit-width is fully parameterized in the quantizer and kernels (key_bits /
# value_bits), so additional presets are a one-line addition here. Keys carry
# more quantization sensitivity than values (key error propagates through the
# softmax exponentials, value error is averaged out by the softmax weights), so
# the shipped preset spends more bits on keys than values.
KVARN_PRESETS: dict[str, dict] = {
    "kvarn_k4v2_g128": {"key_bits": 4, "value_bits": 2, "group": 128},
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
            the variance-normalization loop (default 16).
        boundary_skip_layers: Number of leading / trailing transformer layers
            to keep in fp16 (KVarN's sink/residual analogue). Default 2 mirrors
            TurboQuant's default.
    """

    head_dim: int = 128
    key_bits: int = 4
    value_bits: int = 4
    group: int = 128
    sinkhorn_iters: int = 16        # default; converges in practice by ~4 iters
    sink_tokens: int = 128          # first N tokens per request stay fp16 (NEVER quantised)
    boundary_skip_layers: int = 0   # layer-level skipping off by default; sink_tokens replaces it

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
        """tile_bytes rounded up to multiple of 8 for nicer Triton loads."""
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
    POOL_MEM_FRAC_DEFAULT = 0.08

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
        return max(2 * max_num_seqs + prefill_blocks + 32, 64)

    def max_supported_seqs(
        self,
        total_gpu_bytes: int,
        num_kv_heads: int,
        num_layers: int,
        max_num_batched_tokens: int,
        frac: float | None = None,
    ) -> int:
        """Largest max_num_seqs whose pool fits in `frac` of GPU memory.

        Inverts `pool_slots`: max_slots = budget / (slot_bytes · layers), then
        solve 2·S + prefill + 32 ≤ max_slots for S. Always ≥ 1."""
        if frac is None:
            frac = float(os.environ.get(
                "KVARN_POOL_MEM_FRAC", str(self.POOL_MEM_FRAC_DEFAULT)))
        slot_bytes = self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)
        max_slots = int(total_gpu_bytes * frac / slot_bytes)
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(1, (max_slots - prefill_blocks - 32) // 2)

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
        iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "16"))
        sink_tokens = int(os.environ.get("KVARN_SINK_TOKENS", "128"))
        return KVarNConfig(
            head_dim=head_dim,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            group=preset["group"],
            sinkhorn_iters=iters,
            sink_tokens=sink_tokens,
        )
