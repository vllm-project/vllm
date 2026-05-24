# SPDX-License-Identifier: Apache-2.0
"""FFN intermediate scratch pool for SiluAndMul / MulAndSilu / fatrelu_and_mul.

Cliff 1 fix on TQ3 path (PN12).

Background
----------
`vllm/model_executor/layers/activation.py:146` SiluAndMul.forward_cuda
allocates a fresh `[M, intermediate_size]` BF16 tensor PER LAYER PER STEP:

    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    self.op(out, x)
    return out

For Lorbus Qwen3.6-27B-int4 (intermediate_size=17408, num_hidden_layers=64)
that's **73-285 MiB transient × 64 layers = 4.7-18 GiB allocator churn per
forward step**. The 138 MiB OOM noonghunna reproduced on 3090 + 192K + tool
call (Cliff 1) matches this size class exactly.

PN8 (MTP draft online-quant propagation) closes Cliff 1 on FP8 by freeing
~600 MiB persistent draft VRAM, giving the fragmented heap enough slack
for the 138 MiB transient to land. On TQ3 PN8 only frees ~230 MiB → not
enough slack → OOM still fires. **Different memory class** — persistent
footprint vs transient peak.

PN12 fix: pool the SiluAndMul output across layers.

Why this is safe
----------------
1. **Sequential layer execution.** vLLM's transformer forward calls layers
   in strict sequence. Layer N's `out` tensor is fully consumed (down_proj
   reads it) BEFORE layer N+1 calls `silu_and_mul.forward_cuda` again.
   So a single shared buffer per (intermediate_size, dtype, device) cannot
   be raced against itself.

2. **Pointer-stable.** Same `(intermediate_size, dtype, device, num_tokens)`
   key → returns IDENTICAL `data_ptr()` across calls. Compatible with
   cudagraph capture (graph stores a fixed pointer; replays into the same
   memory).

3. **Slice-on-acquire.** Buffer is allocated at MAX seen num_tokens. Smaller
   subsequent acquires return a `[:M]` view of the same backing tensor.
   Larger acquires grow the buffer ONCE (re-allocate to new max). After
   growth, all subsequent same-or-smaller acquires reuse the new buffer.

4. **Per-shape registry.** Different `(intermediate_size, dtype, device)`
   tuples get distinct buffers — no false sharing across MoE expert sizes
   or KV cache groups with different dtype.

How to use
----------
Operator opt-in via env:

    GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1

Genesis text-patch (PN12) rewrites `SiluAndMul.forward_cuda` to:

    if FFNIntermediateCache.should_apply():
        out = FFNIntermediateCache.acquire_silu_out(
            num_tokens=x.shape[0],
            intermediate_size=x.shape[-1] // 2,
            dtype=x.dtype, device=x.device,
        )
    else:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    self.op(out, x)
    return out

Tradeoffs
---------
- Loses peak-shrink benefit of fresh allocation: the buffer holds at MAX
  seen num_tokens forever. For workloads that bursty-spike to 8K tokens
  once, then run at 64 tokens, the pool stays at the 8K size.
- This is intentional — graph capture safety requires pointer stability.
- Mitigation: the buffer is one shared allocation, not 64 (the gain).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Backport reference: vLLM PR #34207 (silu_and_mul.out variant) — alternative
strategy if Inductor `lower_custom_ops_to_out_variant=True` lands upstream.
Cross-engine reference: SGLang #15927 (piecewise CUDA graph absorption) and
TensorRT-LLM live-range activation reuse pattern (gold standard).
"""
from __future__ import annotations

import logging
import os

import torch

from vllm._genesis.guards import is_nvidia_cuda

log = logging.getLogger("genesis.kernels.ffn_intermediate_cache")


_ENV_FLAG = "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL"


class FFNIntermediateCache:
    """Class-level registry of FFN intermediate output buffers.

    Per-process singleton. Each TP rank has its own registry — buffers are
    NOT shared across processes (no cross-rank IPC).

    Lifecycle
    ---------
    1. First call with a unique `(intermediate_size, dtype, device)` key →
       allocate `[max_M, intermediate_size]` tensor (max_M is the requested
       num_tokens at first call).
    2. Subsequent calls with same key:
       - If requested `num_tokens <= max_M`: return slice `[:num_tokens]`.
         Pointer-stable (same data_ptr across calls).
       - If `num_tokens > max_M`: re-allocate at the new larger max_M.
         data_ptr changes ONCE; subsequent same-large calls stable.
    3. Process shutdown → tensors freed via Python GC.

    Anti-patterns
    -------------
    - Do NOT call `acquire_silu_out` from a thread other than the main
      worker thread. The registry is plain dict, not thread-safe.
    - Do NOT mutate the returned tensor across layers without synchronizing
      reads. (vLLM doesn't do this — strict layer sequencing — so we're
      safe in practice.)
    - Do NOT rely on zero-initialization. Buffer is `torch.empty` style;
      caller must overwrite all elements (which `silu_and_mul` op does
      via in-place write).
    """

    # Maps (intermediate_size, dtype, device, dtype_byte_size) → tensor
    # Public for testing — tests reset between cases.
    _BUFFER_REGISTRY: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}

    # ──────────────────────────────────────────────────────────────────
    # Platform / env gate
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def should_apply() -> bool:
        """Env gate. Operator must explicitly opt in.

        Independent of platform check — caching CPU tensors is also valid
        (e.g., for unit tests). Production callers must additionally check
        `is_nvidia_cuda()` if they care about GPU placement.
        """
        return os.environ.get(_ENV_FLAG, "").strip() in ("1", "true", "True")

    @staticmethod
    def is_production_eligible() -> bool:
        """Stricter check for the actual text-patch apply: env + NVIDIA CUDA."""
        return FFNIntermediateCache.should_apply() and is_nvidia_cuda()

    # ──────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def acquire_silu_out(
        cls,
        num_tokens: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a `[num_tokens, intermediate_size]` view of a pooled buffer.

        Caller writes the silu_and_mul output IN PLACE into the returned
        slice. The slice is a view — no new allocation when the requested
        num_tokens fits in the cached buffer.

        Args:
            num_tokens: rows in the output (M).
            intermediate_size: cols (D, model dimension).
            dtype: torch dtype matching `x.dtype` of the activation input.
            device: CUDA or CPU device.

        Returns:
            `torch.Tensor` of shape `(num_tokens, intermediate_size)`.

        Raises:
            ValueError: if `num_tokens <= 0` or `intermediate_size <= 0`.
        """
        if num_tokens <= 0:
            raise ValueError(
                f"num_tokens must be > 0, got {num_tokens}"
            )
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be > 0, got {intermediate_size}"
            )

        key = (intermediate_size, dtype, device)
        cached = cls._BUFFER_REGISTRY.get(key)

        if cached is None:
            # First allocation for this key — size to requested num_tokens.
            buf = torch.empty(
                (num_tokens, intermediate_size),
                dtype=dtype, device=device,
            )
            cls._BUFFER_REGISTRY[key] = buf
            log.info(
                "[PN12] first acquire silu_out: alloc [%d, %d] %s on %s "
                "(%.1f MiB)",
                num_tokens, intermediate_size, dtype, device,
                num_tokens * intermediate_size *
                _dtype_byte_size(dtype) / 1024 / 1024,
            )
            return buf

        # Cache hit. If requested fits, return a slice.
        cached_max = cached.shape[0]
        if num_tokens <= cached_max:
            return cached[:num_tokens]

        # Requested larger than cached — grow once.
        new_buf = torch.empty(
            (num_tokens, intermediate_size),
            dtype=dtype, device=device,
        )
        cls._BUFFER_REGISTRY[key] = new_buf
        log.info(
            "[PN12] grew silu_out buffer: %d → %d rows for %s on %s",
            cached_max, num_tokens, dtype, device,
        )
        return new_buf

    @classmethod
    def total_pooled_bytes(cls) -> int:
        """Sum of bytes held in pool. For introspection / logging."""
        total = 0
        for buf in cls._BUFFER_REGISTRY.values():
            total += buf.numel() * _dtype_byte_size(buf.dtype)
        return total

    @classmethod
    def num_pools(cls) -> int:
        """How many distinct (size, dtype, device) keys are pooled."""
        return len(cls._BUFFER_REGISTRY)


def _dtype_byte_size(dtype: torch.dtype) -> int:
    """Bytes per element for any torch dtype. Works without importing scipy."""
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.float64, torch.int64):
        return 8
    if dtype in (torch.uint8, torch.int8, torch.bool):
        return 1
    if dtype == torch.float8_e4m3fn:
        return 1
    if dtype == torch.float8_e5m2:
        return 1
    # Fallback — torch's own size accessor when present.
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4  # conservative default
