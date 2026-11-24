# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashTopK behind `fused_mqa_topk_prefill` (`VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk`,
see `fused_mqa_topk.py`): exact, deterministic top-k of the FP8 MQA indexer's prefill
scores with adaptive key pruning.

This module is the vLLM adapter; the kernels live in `topk.py` (`Selector`),
`plan.py`, `recovery.py` and `kernels/` (CuTe DSL and Triton, compiled at runtime),
imported on first use (`import_flashtopk`). They derive from FlashTopK
(https://github.com/Tencent-Hunyuan/FlashTopK, Apache-2.0) at commit
`090902abad86e5bb3324519124a680122fff91a1`, keeping only what the indexer uses (FP8
operands, adaptive pruning, unordered indices, caller-owned scratch); files derived
from FlashInfer, TensorRT-LLM and LiteTopK state so in their headers.

A `Selector` per `(device, top-k)` holds only small per-row state. Each call allocates
its scratch from PyTorch's caching allocator and frees it on return, and
`reserve_flashtopk` touches the peak scratch during the profiling run so it overlaps
dense logits and model activations instead of pinning KV-cache memory in the workspace
manager. `FLASHTOPK_WARMUP` compiles and loads the kernels of reserved calls during
`kernel_warmup`.
"""

import functools
import math
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform

logger = init_logger(__name__)

# FlashTopK prunes keys adaptively from this many keys (plan.ADAPTIVE_MIN_KEYS).
FLASHTOPK_MIN_KEYS = 1 << 17
_MAX_SLICE_ROWS = 8192


@functools.cache
def import_flashtopk():
    """Import FlashTopK's `topk` module on first use; raises if runtime dependencies
    (`nvidia-cutlass-dsl`, `apache-tvm-ffi`) are missing."""
    try:
        from . import topk
    except ImportError as e:
        raise ImportError(
            "VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk needs nvidia-cutlass-dsl and "
            "apache-tvm-ffi (requirements/cuda.txt)."
        ) from e
    return topk


def flashtopk_supported() -> bool:
    """FlashTopK runs on SM90 and SM100-family GPUs."""
    return any(current_platform.is_device_capability_family(f) for f in (90, 100))


@functools.cache
def flashtopk_available() -> bool:
    """True when `VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk` and this GPU supports it;
    raises if FlashTopK's runtime dependencies are missing."""
    if envs.VLLM_FUSED_MQA_TOPK_BACKEND != "flashtopk" or not (
        current_platform.is_cuda() and flashtopk_supported()
    ):
        return False
    import_flashtopk()
    return True


def _device_index(device: torch.device) -> int:
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


@functools.cache
def _chunk_size(index: int) -> int:
    """Rows per slice on `cuda:index`: two full waves of 16-row pruning tiles (four
    waves of 8-row scan/refine tiles), keeping 100% wave efficiency while bounding peak
    per-call scratch below the dense prefill logits budget."""
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    return min(_MAX_SLICE_ROWS, max(1024, 32 * sms))


def flashtopk_prunes(index: int, rows: int, keys: int) -> bool:
    """Whether a call of `rows` rows and `keys` keys on `cuda:index` has enough key
    length and CTA occupancy for adaptive pruning to apply."""
    if keys < FLASHTOPK_MIN_KEYS or rows <= 0:
        return False
    topk_mod = import_flashtopk()
    slice_rows = min(rows, _chunk_size(index))
    return topk_mod.planner.adaptive_threshold(index, slice_rows, keys) > 0.0


def flashtopk_chunk(
    max_seqlen_k: int,
    keys: int,
    rows: int,
    device: torch.device | None = None,
) -> bool:
    """Whether a prefill chunk takes FlashTopK: its longest row sees at least
    `FLASHTOPK_MIN_KEYS` keys, its gathered keys fit the fused path, and its rows give
    key pruning enough CTAs to beat scanning and dense logits."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_MAX_KEYS,
    )

    if (
        not flashtopk_available()
        or max_seqlen_k < FLASHTOPK_MIN_KEYS
        or keys > FUSED_MQA_TOPK_MAX_KEYS
    ):
        return False
    index = (
        torch.accelerator.current_device_index()
        if device is None
        else _device_index(device)
    )
    return flashtopk_prunes(index, rows, keys)


def flashtopk_takes(
    device: torch.device,
    rows: int,
    keys: int,
    max_seqlen_k: int | None = None,
) -> bool:
    """Whether `fused_mqa_topk_prefill` dispatches to FlashTopK: when `flashtopk_chunk`
    accepts the call, or always when no native op is available (SM90)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        fused_mqa_topk_available,
    )

    if not flashtopk_available():
        return False
    if not fused_mqa_topk_available():
        return True
    longest = keys if max_seqlen_k is None else max_seqlen_k
    return flashtopk_chunk(longest, keys, rows, device)


@functools.cache
def _selector(index: int, topk: int):
    """The cached `Selector` for `(device, top-k)`, holding only per-row state."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_MAX_KEYS,
    )

    chunk_size = _chunk_size(index)
    logger.info(
        "Planning the FlashTopK fused MQA+top-k prefill backend "
        "(top-k %d, chunk_size %d) on cuda:%d.",
        topk,
        chunk_size,
        index,
    )
    selector = import_flashtopk().Selector(
        topk, torch.device("cuda", index), chunk_size
    )
    return selector.plan(chunk_size, FUSED_MQA_TOPK_MAX_KEYS)


@functools.lru_cache(maxsize=64)
def flashtopk_scratch_bytes(index: int, rows: int, keys: int, topk: int) -> int:
    """Bytes of per-call scratch for up to `rows` rows and `keys` keys."""
    return _selector(index, topk).scratch_bytes(rows, keys)


def flashtopk_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run `fused_mqa_topk_prefill` on FlashTopK with per-call scratch, converting
    indices in place to vLLM's `key - start` convention with `-1` padding."""
    if not output.is_contiguous():
        raise ValueError("fused_mqa_topk_prefill: output must be contiguous")
    index, topk = _device_index(q.device), output.shape[1]
    scratch = torch.empty(
        flashtopk_scratch_bytes(index, q.shape[0], k.shape[0], topk),
        dtype=torch.uint8,
        device=q.device,
    )
    _selector(index, topk).run(q, k, weights, starts, ends, scales, output, scratch)
    output.sub_(starts[:, None]).clamp_(min=-1)


_RESERVED: dict[tuple[int, int], tuple[int, int]] = {}


def reserve_flashtopk(device: torch.device, rows: int, keys: int, topk: int) -> None:
    """Profiling run: accounts for FlashTopK's peak per-call scratch (freed again so it
    overlaps dense logits and model activations) and records calls for warmup."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_HEAD_DIM,
        FUSED_MQA_TOPK_HEADS,
        FUSED_MQA_TOPK_MAX_KEYS,
        FUSED_MQA_TOPK_MAX_ROWS,
    )

    rows = min(rows, FUSED_MQA_TOPK_MAX_ROWS)
    keys = min(keys, FUSED_MQA_TOPK_MAX_KEYS)
    if not flashtopk_chunk(keys, keys, rows, device):
        return
    index = _device_index(device)
    nbytes = flashtopk_scratch_bytes(index, rows, keys, topk)
    if torch.cuda.get_device_capability(index)[0] == 9:
        slice_rows = math.ceil(min(rows, _chunk_size(index)) / 8) * 8
        nbytes += slice_rows * FUSED_MQA_TOPK_HEADS * FUSED_MQA_TOPK_HEAD_DIM
    _ = torch.empty(nbytes, dtype=torch.uint8, device=device)
    key = (index, topk)
    reserved_rows, reserved_keys = _RESERVED.get(key, (0, 0))
    _RESERVED[key] = (max(reserved_rows, rows), max(reserved_keys, keys))


_WARMUP_KEYS = (FLASHTOPK_MIN_KEYS, FLASHTOPK_MIN_KEYS + 1, FLASHTOPK_MIN_KEYS + 13)


class FlashTopkWarmup(VllmJitKernel["FlashTopkWarmup.CompileKey"]):
    """`kernel_warmup` for FlashTopK: compiles CuTe variants and every reachable
    integer specialization of the Triton pruning planner before serving."""

    @dataclass(frozen=True)
    class CompileKey:
        device_index: int
        topk: int

    def get_warmup_keys(self) -> list["FlashTopkWarmup.CompileKey"]:
        return [self.CompileKey(*r) for r in sorted(_RESERVED)]

    def compile(self, compile_key: "FlashTopkWarmup.CompileKey") -> None:
        if compile_key in self._compiled_cache:
            return
        from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
            FUSED_MQA_TOPK_HEAD_DIM as head_dim,
        )
        from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
            FUSED_MQA_TOPK_HEADS as heads,
        )

        index, topk = compile_key.device_index, compile_key.topk
        _selector(index, topk).compile()
        reserved_rows, reserved_keys = _RESERVED[(index, topk)]
        chunk_size = _chunk_size(index)
        warmup_rows = (chunk_size, chunk_size - 1, 1000, 1)
        rows = sorted({min(r, reserved_rows) for r in warmup_rows})
        keys = sorted({min(n, reserved_keys) for n in _WARMUP_KEYS})
        device = torch.device("cuda", index)
        gen = torch.Generator(device=device).manual_seed(0)
        fp8 = current_platform.fp8_dtype()
        with torch.inference_mode():
            q = torch.randn(rows[-1], heads, head_dim, device=device, generator=gen).to(
                fp8
            )
            k = torch.randn(keys[-1], head_dim, device=device, generator=gen).to(fp8)
            scales = torch.ones(keys[-1], device=device)
            weights = torch.randn(rows[-1], heads, device=device, generator=gen) / heads
            starts = torch.zeros(rows[-1], dtype=torch.int32, device=device)
            output = torch.empty(rows[-1], topk, dtype=torch.int32, device=device)
            for r in rows:
                for n in keys:
                    ends = torch.full((r,), n, dtype=torch.int32, device=device)
                    flashtopk_mqa_topk_prefill(
                        q[:r],
                        k[:n],
                        scales[:n],
                        weights[:r],
                        starts[:r],
                        ends,
                        output[:r],
                    )
            torch.accelerator.synchronize()
        logger.info(
            "Warmed up the FlashTopK fused MQA+top-k prefill backend (top-k %d) "
            "on cuda:%d: %d calls.",
            topk,
            index,
            len(rows) * len(keys),
        )
        self._compiled_cache[compile_key] = True


FLASHTOPK_WARMUP = FlashTopkWarmup()
