# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashTopK (vllm/third_party/flashtopk) behind fused_mqa_topk_prefill
(VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk, see fused_mqa_topk.py).

A Selector per (device, top-k) holds only per-row state. Each call's scratch
(candidates, pruning plan and mask, recovery) is the caller's: the indexers
take it from the workspace manager with their K-gather buffers, so the
profiling run accounts for it and no call allocates it. kernel_warmup compiles
and loads the kernels of the calls the profiling run reserved (FLASHTOPK_WARMUP,
registered by the indexers when kernel_config.enable_jit_warmup).
"""

import functools
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform

logger = init_logger(__name__)

# FlashTopK prunes keys adaptively from this many keys (its ADAPTIVE_MIN_KEYS).
# Below it, it scans every key and is no faster than the native op.
FLASHTOPK_MIN_KEYS = 1 << 17
# Rows per slice: bounds the scratch (~8 * 8192 * max(4 * topk, 16384) bytes).
_CHUNK_SIZE = 8192
_PRUNING = "adaptive"
# FlashTopK's operands: [rows, 32, 128] queries and [keys, 128] keys.
_HEADS, _HEAD_DIM = 32, 128


@functools.cache
def import_flashtopk():
    """The vendored FlashTopK; raises if its runtime dependencies (CuTe DSL,
    TVM FFI) are missing."""
    try:
        from vllm.third_party import flashtopk
    except ImportError as e:
        raise ImportError(
            "VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk needs nvidia-cutlass-dsl and "
            "apache-tvm-ffi (requirements/cuda.txt)."
        ) from e
    return flashtopk


def flashtopk_supported() -> bool:
    """FlashTopK runs on SM90 and SM100-family GPUs."""
    return any(current_platform.is_device_capability_family(f) for f in (90, 100))


@functools.cache
def _selector(index: int, topk: int):
    """The Selector of (device, top-k), planned for the largest fused chunk
    without its scratch. Kernels compile in FLASHTOPK_WARMUP (or at first use
    when JIT warmup is off)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_MAX_KEYS,
        FUSED_MQA_TOPK_MAX_ROWS,
    )

    logger.info(
        "Planning the FlashTopK fused MQA+top-k prefill backend "
        "(top-k %d, chunk_size %d, pruning %s) on cuda:%d.",
        topk,
        _CHUNK_SIZE,
        _PRUNING,
        index,
    )
    return (
        import_flashtopk()
        .Selector(
            topk=topk,
            dtype=current_platform.fp8_dtype(),
            device=torch.device("cuda", index),
            pruning=_PRUNING,
            chunk_size=_CHUNK_SIZE,
        )
        .plan(
            FUSED_MQA_TOPK_MAX_ROWS,
            FUSED_MQA_TOPK_MAX_KEYS,
            warmup=False,
            scratch=False,
        )
    )


@functools.lru_cache(maxsize=64)
def flashtopk_scratch_bytes(index: int, rows: int, keys: int, topk: int) -> int:
    """Bytes of scratch of calls of up to `rows` rows and `keys` keys."""
    return _selector(index, topk).scratch_bytes(rows, keys)


def flashtopk_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> None:
    """fused_mqa_topk_prefill on FlashTopK, its scratch in `workspace`
    (uint8, at least flashtopk_scratch_bytes of this call; allocated when
    None). FlashTopK returns absolute key indices sorted by score, -1 padded;
    vLLM's convention is `key - start`."""
    if not output.is_contiguous():
        raise ValueError("fused_mqa_topk_prefill: output must be contiguous")
    index = q.device.index
    selector = _selector(index, output.shape[1])
    if workspace is None:
        workspace = torch.empty(
            flashtopk_scratch_bytes(index, q.shape[0], k.shape[0], output.shape[1]),
            dtype=torch.uint8,
            device=q.device,
        )
    selector.run(q, k, weights, starts, ends, scales, out=output, scratch=workspace)
    output.sub_(starts[:, None]).clamp_(min=-1)


# (device index, top-k) of the calls the profiling run reserved, with their
# most rows and keys: what FLASHTOPK_WARMUP compiles.
_RESERVED: dict[tuple[int, int], tuple[int, int]] = {}


def reserve_flashtopk(index: int, rows: int, keys: int, topk: int) -> None:
    """Records calls of up to `rows` rows and `keys` keys for FLASHTOPK_WARMUP."""
    reserved_rows, reserved_keys = _RESERVED.get((index, topk), (0, 0))
    _RESERVED[(index, topk)] = (max(reserved_rows, rows), max(reserved_keys, keys))


# Selector.plan(warmup=True) compiles FlashTopK's CuTe kernels but not the
# Triton planner of adaptive pruning, which specializes on its integer
# arguments: a slice's rows and their padding to the row group (2 rows on
# SM100, 16 on SM90) by divisibility by 16 and by 1, the keys and their
# padding to 4 by divisibility by 16. These rows and keys reach every
# combination. The calls also load each kernel module, which a first launch
# otherwise does inside a prefill.
_WARMUP_ROWS = (_CHUNK_SIZE, _CHUNK_SIZE - 1, 1000, 1)
_WARMUP_KEYS = (FLASHTOPK_MIN_KEYS, FLASHTOPK_MIN_KEYS + 1, FLASHTOPK_MIN_KEYS + 13)


def _warmup_scratch(index: int, rows: int, keys: int, topk: int) -> torch.Tensor:
    """The warmup calls' scratch, from the workspace the profiling run sized."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    spec = ((flashtopk_scratch_bytes(index, rows, keys, topk),), torch.uint8)
    if is_workspace_manager_initialized():
        return current_workspace_manager().get_simultaneous(spec)[0]
    return torch.empty(spec[0], dtype=spec[1], device=torch.device("cuda", index))


class FlashTopkWarmup(VllmJitKernel["FlashTopkWarmup.CompileKey"]):
    """kernel_warmup for FlashTopK: compiles and loads the kernels of each
    (device, top-k) the profiling run reserved, before the first prefill. With
    no reservation (FlashTopK takes no calls) it has no keys."""

    @dataclass(frozen=True)
    class CompileKey:
        device_index: int
        topk: int

    def get_warmup_keys(self) -> list["FlashTopkWarmup.CompileKey"]:
        return [self.CompileKey(*r) for r in sorted(_RESERVED)]

    def compile(self, compile_key: "FlashTopkWarmup.CompileKey") -> None:
        if compile_key in self._compiled_cache:
            return
        index, topk = compile_key.device_index, compile_key.topk
        selector = _selector(index, topk)
        selector.plan(selector.max_rows, selector.max_kv, warmup=True, scratch=False)
        reserved_rows, reserved_keys = _RESERVED[(index, topk)]
        rows = sorted({min(r, reserved_rows) for r in _WARMUP_ROWS})
        keys = sorted({min(n, reserved_keys) for n in _WARMUP_KEYS})
        scratch = _warmup_scratch(index, rows[-1], keys[-1], topk)
        device = torch.device("cuda", index)
        gen = torch.Generator(device=device).manual_seed(0)
        with torch.inference_mode():
            q = torch.randn(
                rows[-1], _HEADS, _HEAD_DIM, device=device, generator=gen
            ).to(current_platform.fp8_dtype())
            k = torch.randn(keys[-1], _HEAD_DIM, device=device, generator=gen).to(
                current_platform.fp8_dtype()
            )
            scales = torch.ones(keys[-1], device=device)
            weights = (
                torch.randn(rows[-1], _HEADS, device=device, generator=gen) / _HEADS
            )
            starts = torch.zeros(rows[-1], dtype=torch.int32, device=device)
            output = torch.empty(rows[-1], topk, dtype=torch.int32, device=device)
            for r in rows:
                for n in keys:
                    # Every row sees all keys: adaptive pruning then plans and
                    # launches both its scan and its pruning kernels.
                    ends = torch.full((r,), n, dtype=torch.int32, device=device)
                    flashtopk_mqa_topk_prefill(
                        q[:r],
                        k[:n],
                        scales[:n],
                        weights[:r],
                        starts[:r],
                        ends,
                        output[:r],
                        scratch,
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
