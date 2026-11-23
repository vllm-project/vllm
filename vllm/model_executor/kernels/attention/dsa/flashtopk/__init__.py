# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashTopK behind fused_mqa_topk_prefill (VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk,
see fused_mqa_topk.py): exact, deterministic top-k of the FP8 MQA indexer's
prefill scores with adaptive key pruning.

This module is the vLLM side; the kernels are topk.py (Selector), plan.py,
recovery.py and kernels/ (CuTe DSL and Triton, compiled at runtime), imported
on first use (import_flashtopk). They derive from FlashTopK
(https://github.com/Tencent-Hunyuan/FlashTopK, Apache-2.0) at commit
090902abad86e5bb3324519124a680122fff91a1, keeping only what the indexer uses
(FP8 operands, adaptive pruning, caller-owned scratch); files derived from
FlashInfer, TensorRT-LLM and LiteTopK say so in their headers.

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

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform

logger = init_logger(__name__)

# FlashTopK prunes keys adaptively from this many keys (its ADAPTIVE_MIN_KEYS).
# Below it, it scans every key and is no faster than the native op.
FLASHTOPK_MIN_KEYS = 1 << 17
# Rows per slice: bounds the scratch (~8 * 8192 * max(4 * topk, 16384) bytes).
_CHUNK_SIZE = 8192


@functools.cache
def import_flashtopk():
    """FlashTopK's kernels (the topk module, imported on first use so this
    package loads without CuTe DSL); raises if their runtime dependencies (CuTe
    DSL, TVM FFI) are missing."""
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
    """FlashTopK is the selected fused MQA+top-k backend
    (VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk) and runs on this GPU; raises if its
    runtime dependencies are missing."""
    if envs.VLLM_FUSED_MQA_TOPK_BACKEND != "flashtopk" or not (
        current_platform.is_cuda() and flashtopk_supported()
    ):
        return False
    import_flashtopk()
    return True


def flashtopk_takes(keys: int) -> bool:
    """Whether a fused_mqa_topk_prefill call over `keys` keys runs on FlashTopK:
    from FLASHTOPK_MIN_KEYS keys, where key pruning makes it faster than the
    native op, or always without the native op (SM90)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        fused_mqa_topk_available,
    )

    return flashtopk_available() and (
        keys >= FLASHTOPK_MIN_KEYS or not fused_mqa_topk_available()
    )


def _device_index(device: torch.device) -> int:
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


@functools.cache
def _selector(index: int, topk: int):
    """The Selector of (device, top-k), its per-row state planned for the
    largest fused chunk (a slice of _CHUNK_SIZE rows). Kernels compile in
    FLASHTOPK_WARMUP (or at first use when JIT warmup is off)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_MAX_KEYS,
    )

    logger.info(
        "Planning the FlashTopK fused MQA+top-k prefill backend "
        "(top-k %d, chunk_size %d) on cuda:%d.",
        topk,
        _CHUNK_SIZE,
        index,
    )
    selector = import_flashtopk().Selector(
        topk, torch.device("cuda", index), _CHUNK_SIZE
    )
    return selector.plan(_CHUNK_SIZE, FUSED_MQA_TOPK_MAX_KEYS)


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
    None). FlashTopK writes absolute key indices sorted by score, -1 padded;
    vLLM's convention is `key - start`."""
    if not output.is_contiguous():
        raise ValueError("fused_mqa_topk_prefill: output must be contiguous")
    index, topk = q.device.index, output.shape[1]
    if workspace is None:
        workspace = torch.empty(
            flashtopk_scratch_bytes(index, q.shape[0], k.shape[0], topk),
            dtype=torch.uint8,
            device=q.device,
        )
    _selector(index, topk).run(q, k, weights, starts, ends, scales, output, workspace)
    output.sub_(starts[:, None]).clamp_(min=-1)


# (device index, top-k) of the calls the profiling run reserved, with their
# most rows and keys: what FLASHTOPK_WARMUP compiles.
_RESERVED: dict[tuple[int, int], tuple[int, int]] = {}


def _max_keys(keys: int) -> int:
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_MAX_KEYS,
    )

    return min(keys, FUSED_MQA_TOPK_MAX_KEYS)


def flashtopk_workspace_specs(
    device: torch.device, rows: int, keys: int, topk: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    """The workspace-manager scratch of fused_mqa_topk_prefill calls of up to
    `rows` rows and `keys` keys when FlashTopK takes them, else none (the
    native op allocates its own workspaces). Request it in the same
    `get_simultaneous` call as the buffers that hold the calls' operands (the
    gathered K), so they do not overlap."""
    keys = _max_keys(keys)
    if not flashtopk_takes(keys):
        return []
    nbytes = flashtopk_scratch_bytes(_device_index(device), rows, keys, topk)
    return [((nbytes,), torch.uint8)]


def reserve_flashtopk(device: torch.device, rows: int, keys: int, topk: int) -> None:
    """Profiling run: records the FlashTopK calls of up to `rows` rows and
    `keys` keys for FLASHTOPK_WARMUP (their scratch: flashtopk_workspace_specs)."""
    keys = _max_keys(keys)
    if not flashtopk_takes(keys):
        return
    key = (_device_index(device), topk)
    reserved_rows, reserved_keys = _RESERVED.get(key, (0, 0))
    _RESERVED[key] = (max(reserved_rows, rows), max(reserved_keys, keys))


# Selector.compile() compiles FlashTopK's CuTe kernels but not the
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
        from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
            FUSED_MQA_TOPK_HEAD_DIM as head_dim,
        )
        from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
            FUSED_MQA_TOPK_HEADS as heads,
        )

        index, topk = compile_key.device_index, compile_key.topk
        _selector(index, topk).compile()
        reserved_rows, reserved_keys = _RESERVED[(index, topk)]
        rows = sorted({min(r, reserved_rows) for r in _WARMUP_ROWS})
        keys = sorted({min(n, reserved_keys) for n in _WARMUP_KEYS})
        scratch = _warmup_scratch(index, rows[-1], keys[-1], topk)
        device = torch.device("cuda", index)
        gen = torch.Generator(device=device).manual_seed(0)
        fp8 = current_platform.fp8_dtype()
        with torch.inference_mode():
            q = torch.randn(rows[-1], heads, head_dim, device=device, generator=gen)
            q = q.to(fp8)
            k = torch.randn(keys[-1], head_dim, device=device, generator=gen).to(fp8)
            scales = torch.ones(keys[-1], device=device)
            weights = torch.randn(rows[-1], heads, device=device, generator=gen) / heads
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
