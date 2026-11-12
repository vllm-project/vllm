# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused FP8 MQA-logits + top-k prefill selection.

Backends (VLLM_FUSED_MQA_TOPK_BACKEND): "native", vLLM's fused_mqa_topk op
(SM100 family), or "flashtopk", FlashTopK's Selector (SM90 and SM100 family).
Both share the eligibility rule (use_fused_mqa_topk), the chunk planner and the
per-chunk gate (fused_mqa_topk_chunk).

Startup: the profiling run reserves each backend's workspace
(reserve_fused_mqa_topk_workspaces); kernel_warmup then compiles and loads
the kernels of every (backend, device, top-k) it reserved
(FUSED_MQA_TOPK_WARMUP, registered by the indexers when
kernel_config.enable_jit_warmup), so no prefill compiles or loads one.
"""

import functools
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Current restrictions of the fused_mqa_topk kernel:
FUSED_MQA_TOPK_HEADS = 32
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPKS = (512, 1024, 2048)
# Most gathered keys (all requests of a chunk) one call accepts: candidates
# store the key index in 20 bits (kMaxKeys in csrc/fused_mqa_topk/constants.h).
# Chunks with more keys take the dense path; the planner never packs past it.
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20
# Rows per fused_mqa_topk call. The kernel has no row limit, but prefill chunks above
# 16384 rows (--max-num-batched-tokens > 16384) are not validated end to end yet,
# so the chunk planner splits fused_mqa_topk chunks at this many rows.
FUSED_MQA_TOPK_MAX_ROWS = 16384


def fused_mqa_topk_min_keys(topk: int) -> int:
    """Longest-row key count from which fused_mqa_topk beats the dense path
    on SM100f (shorter chunks stay dense): 32K, or 64K for top-2048."""
    return 1 << 16 if topk > 1024 else 1 << 15


# Mirrors csrc/fused_mqa_topk/constants.h (the op checks the workspace shapes).
_CANDIDATES_PER_ROW = 16384  # kCandidatesPerRow
_RETAINED_PER_ROW = 4096  # kRetainedPerRow
_SLOTS_PER_SM = 2  # candidate slots per persistent CTA
_BLOCK_ROWS = 8  # kBlockRows: query rows per block, one block per CTA at a time


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def fused_mqa_topk_backend() -> str:
    """The fused_mqa_topk backend: "native" or "flashtopk"."""
    return envs.VLLM_FUSED_MQA_TOPK_BACKEND


@functools.cache
def _flashtopk():
    """The vendored FlashTopK (vllm/third_party/flashtopk); raises if its
    runtime dependencies (CuTe DSL, TVM FFI) are missing."""
    try:
        from vllm.third_party import flashtopk
    except ImportError as e:
        raise ImportError(
            "VLLM_FUSED_MQA_TOPK_BACKEND=flashtopk needs nvidia-cutlass-dsl and "
            "apache-tvm-ffi (requirements/cuda.txt)."
        ) from e
    return flashtopk


@functools.cache
def fused_mqa_topk_available() -> bool:
    """True when the selected backend runs on this GPU: the native op is
    compiled in on an SM100-family GPU, or FlashTopK on SM90 or SM100 family."""
    if not current_platform.is_cuda():
        return False
    if fused_mqa_topk_backend() == "flashtopk":
        if not (
            current_platform.is_device_capability_family(90)
            or current_platform.is_device_capability_family(100)
        ):
            return False
        _flashtopk()
        return True
    return current_platform.is_device_capability_family(100) and hasattr(
        torch.ops._C, "fused_mqa_topk_prefill"
    )


def use_fused_mqa_topk(
    *,
    query_shape: tuple[int, ...],
    topk: int,
    fp4_cache: bool,
    full_context: bool,
    candidate_selection: bool = False,
) -> bool:
    """Whether an indexer may use the fused_mqa_topk kernel: the single
    eligibility rule of the prefill chunk planner and the indexer layers (each
    prefill chunk then also needs fused_mqa_topk_chunk).

    query_shape: one token's indexer queries, (heads, head_dim) = (32, 128);
    topk: the per-row top-k (pools for kpool indexers) in {512, 1024, 2048};
    fp4_cache: the MXFP4 indexer cache (and its FP4 queries) stays dense;
    full_context: every rank ranks its rows over the whole context (no DCP, or
    PCP after the KV all-gather); pure DCP needs the per-shard top-k merge;
    candidate_selection: v4.1 two-level selection needs the logits.
    VLLM_BATCH_INVARIANT keeps the dense path: the fused gate depends on the
    chunk's total keys, and ties at the k-th score are taken in atomic order.
    """
    return (
        envs.VLLM_USE_FUSED_MQA_TOPK
        and not envs.VLLM_BATCH_INVARIANT
        and not fp4_cache
        and full_context
        and not candidate_selection
        and tuple(query_shape) == (FUSED_MQA_TOPK_HEADS, FUSED_MQA_TOPK_HEAD_DIM)
        and topk in FUSED_MQA_TOPK_TOPKS
        and fused_mqa_topk_available()
    )


def fused_mqa_topk_chunk(
    max_seqlen_k: int, keys: int, topk: int, rows: int, dense_rows: int
) -> bool:
    """Whether a prefill chunk of a fused_mqa_topk-eligible layer takes the
    fused_mqa_topk kernel: its longest row sees at least fused_mqa_topk_min_keys
    keys (the dense path is as fast below), its gathered keys fit the kernel,
    and its rows keep the SMs busy.

    Each SM scans every key of one 8-row block at a time, so the blocks run in
    waves of one per SM. A last wave that fills less than half the SMs costs a
    whole wave and makes the kernel slower than dense, unless dense needs more
    than one logits call (rows > dense_rows, its rows per call).
    """
    sms = _sm_count(torch.accelerator.current_device_index())
    last_wave = (-(-rows // _BLOCK_ROWS) - 1) % sms + 1  # blocks, 1..sms
    return (
        max_seqlen_k >= fused_mqa_topk_min_keys(topk)
        and keys <= FUSED_MQA_TOPK_MAX_KEYS
        and (rows > dense_rows or 2 * last_wave >= sms)
    )


def get_fused_mqa_topk_workspaces(
    device: torch.device, rows: int
) -> list[torch.Tensor]:
    """The workspaces of one fused_mqa_topk call over `rows` query rows, in the
    op's argument order (see fused_mqa_topk.cu), all fresh: the zeroed block
    counter, then the scratch."""
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    blocks = -(-rows // _BLOCK_ROWS)
    # One persistent CTA per SM, at most one per block; S = 2 slots per CTA.
    slot_rows = _BLOCK_ROWS * _SLOTS_PER_SM * min(blocks, _sm_count(index))
    empty = functools.partial(torch.empty, device=device)
    return [
        torch.zeros(2, dtype=torch.int32, device=device),  # block counter
        empty((slot_rows, _CANDIDATES_PER_ROW), dtype=torch.uint16),  # fp24 lows
        empty((slot_rows, _CANDIDATES_PER_ROW), dtype=torch.int32),  # indices
        empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.float32),  # retained
        empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.int32),  # their indices
        empty(blocks * _BLOCK_ROWS, dtype=torch.int32),  # per-row candidate counts
        empty(blocks, dtype=torch.int32),  # blocks queued for the exact rescan
    ]


def fused_mqa_topk_key_scales(scales: torch.Tensor) -> torch.Tensor:
    """Per-key fp32 scales from the gathered indexer K-cache scales.

    scales: [keys, 4] uint8. SM100 caches store each power-of-two scale as four
    equal UE8M0 bytes; the fp32 value 2**(e - 127) has bit pattern `e << 23`.
    Other caches (SM90) store the fp32 scale itself.
    """
    if not current_platform.is_device_capability_family(100):
        return scales.view(torch.float32).squeeze(-1)
    exponents = scales.view(torch.int32).squeeze(-1) & 0xFF
    return (exponents << 23).view(torch.float32)


def fused_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fused replacement of `fp8_fp4_mqa_logits` + `top_k_per_row_prefill`.

    q: [rows, 32, 128] fp8 e4m3; k: [keys, 128] fp8 e4m3 (keys <=
    FUSED_MQA_TOPK_MAX_KEYS); scales: [keys] fp32; weights: [rows, 32] fp32;
    starts/ends: [rows] int32, the visible key range `[start, end)` of each
    row; output: [rows, k] int32 with k (the top-k) 512, 1024 or 2048. All
    contiguous; q, k, scales, weights and output 16-byte aligned (the op checks
    the contract).

    Writes the same convention as `ops.top_k_per_row_prefill`: indices
    relative to the row's `start` (i.e. `key - start`), unordered, `-1`
    padded when fewer than k keys are visible.
    """
    if fused_mqa_topk_backend() == "flashtopk":
        logger.info_once(
            "Using the FlashTopK fused MQA+top-k prefill backend (top-k %d).",
            output.shape[1],
        )
        _flashtopk_run(q, k, scales, weights, starts, ends, output)
        return
    logger.info_once(
        "Using the single-launch fused MQA+top-k prefill kernel "
        "(FP8, 32 heads, top-k %d).",
        output.shape[1],
    )
    _native_run(q, k, scales, weights, starts, ends, output)


def _native_run(q, k, scales, weights, starts, ends, output) -> None:
    torch.ops._C.fused_mqa_topk_prefill(
        q,
        k,
        scales,
        weights,
        starts,
        ends,
        output,
        *get_fused_mqa_topk_workspaces(q.device, q.shape[0]),
    )


# (backend, device index, top-k) of every call this process's profiling run
# reserved: the shapes FUSED_MQA_TOPK_WARMUP compiles.
_RESERVED: set[tuple[str, int, int]] = set()


def reserve_fused_mqa_topk_workspaces(
    device: torch.device, rows: int, topk: int
) -> None:
    """Allocates the largest workspaces of the selected backend during the
    profiling run, so they are accounted against the KV cache, and records
    the (backend, device, top-k) for FUSED_MQA_TOPK_WARMUP.

    native: allocated and freed again (they overlap the dense logits).
    flashtopk: the Selector's workspace, held for the process (about 1 GiB per
    device at chunk_size 8192). Its kernels compile in kernel_warmup.
    """
    backend, index = fused_mqa_topk_backend(), _device_index(device)
    _RESERVED.add((backend, index, topk))
    if backend == "flashtopk":
        _flashtopk_selector(index, topk)
    else:
        get_fused_mqa_topk_workspaces(device, rows)


# FlashTopK: rows per slice (workspace ~8 * 8192 * max(4 * topk, 16384) bytes)
# and its adaptive key pruning, as benchmarked against the native kernel.
_FLASHTOPK_CHUNK_SIZE = 8192
_FLASHTOPK_PRUNING = "adaptive"


def _device_index(device: torch.device) -> int:
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


@functools.cache
def _flashtopk_selector(index: int, topk: int):
    """One FlashTopK Selector per (device, top-k), its workspace planned for
    the largest fused chunk. Kernels compile in FUSED_MQA_TOPK_WARMUP (or at
    first use when JIT warmup is off)."""
    flashtopk = _flashtopk()
    logger.info(
        "Planning the FlashTopK fused MQA+top-k prefill backend "
        "(top-k %d, chunk_size %d, pruning %s) on cuda:%d.",
        topk,
        _FLASHTOPK_CHUNK_SIZE,
        _FLASHTOPK_PRUNING,
        index,
    )
    return flashtopk.Selector(
        topk=topk,
        num_heads=FUSED_MQA_TOPK_HEADS,
        dtype=current_platform.fp8_dtype(),
        device=torch.device("cuda", index),
        pruning=_FLASHTOPK_PRUNING,
        chunk_size=_FLASHTOPK_CHUNK_SIZE,
    ).plan(FUSED_MQA_TOPK_MAX_ROWS, FUSED_MQA_TOPK_MAX_KEYS, warmup=False)


def _flashtopk_run(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """fused_mqa_topk_prefill on FlashTopK. FlashTopK returns absolute key
    indices sorted by score, -1 padded; vLLM's convention is `key - start`."""
    selector = _flashtopk_selector(_device_index(q.device), output.shape[1])
    out = output if output.is_contiguous() else torch.empty_like(output)
    selector.run(q, k, weights, starts, ends, scales, out=out)
    out.sub_(starts[:, None]).clamp_(min=-1)
    if out is not output:
        output.copy_(out)


# FlashTopK's warmup calls. Selector.plan(warmup=True) compiles its CuTe
# kernels but not the Triton planner of adaptive pruning (from 128K keys),
# which specializes on its integer arguments: a slice's rows and their padding
# to the row group (2 rows on SM100, 16 on SM90) by divisibility by 16 and by
# 1, the keys and their padding to 4 by divisibility by 16. These rows and
# keys reach every combination. The calls also load each kernel module, which
# a first launch otherwise does inside a prefill.
_FLASHTOPK_WARMUP_ROWS = (FUSED_MQA_TOPK_MAX_ROWS, 8191, 1000, 1)
_FLASHTOPK_WARMUP_KEYS = (1 << 17, (1 << 17) + 1, (1 << 17) + 13)
# The native op: one call loads its module (no JIT specialization).
_NATIVE_WARMUP_ROWS, _NATIVE_WARMUP_KEYS = 1024, 1 << 16


class FusedMqaTopkWarmup(VllmJitKernel["FusedMqaTopkWarmup.CompileKey"]):
    """kernel_warmup for fused_mqa_topk_prefill: compiles and loads the kernels
    of each (backend, device, top-k) the profiling run reserved, before the
    first prefill. Registered by the sparse indexers; with no reservation
    (fused path not eligible) it has no keys."""

    @dataclass(frozen=True)
    class CompileKey:
        backend: str
        device_index: int
        topk: int

    def get_warmup_keys(self) -> list["FusedMqaTopkWarmup.CompileKey"]:
        return [self.CompileKey(*r) for r in sorted(_RESERVED)]

    def compile(self, compile_key: "FusedMqaTopkWarmup.CompileKey") -> None:
        if compile_key in self._compiled_cache:
            return
        device = torch.device("cuda", compile_key.device_index)
        if compile_key.backend == "flashtopk":
            selector = _flashtopk_selector(compile_key.device_index, compile_key.topk)
            selector.plan(FUSED_MQA_TOPK_MAX_ROWS, FUSED_MQA_TOPK_MAX_KEYS, warmup=True)
            shapes = [
                (rows, keys)
                for rows in _FLASHTOPK_WARMUP_ROWS
                for keys in _FLASHTOPK_WARMUP_KEYS
            ]
            run = _flashtopk_run
        else:
            shapes = [(_NATIVE_WARMUP_ROWS, _NATIVE_WARMUP_KEYS)]
            run = _native_run
        max_rows = max(r for r, _ in shapes)
        max_keys = max(n for _, n in shapes)
        gen = torch.Generator(device=device).manual_seed(0)
        fp8 = current_platform.fp8_dtype()
        with torch.inference_mode():
            q = torch.randn(
                max_rows,
                FUSED_MQA_TOPK_HEADS,
                FUSED_MQA_TOPK_HEAD_DIM,
                device=device,
                generator=gen,
            ).to(fp8)
            k = torch.randn(
                max_keys, FUSED_MQA_TOPK_HEAD_DIM, device=device, generator=gen
            ).to(fp8)
            scales = torch.ones(max_keys, device=device)
            weights = (
                torch.randn(
                    max_rows, FUSED_MQA_TOPK_HEADS, device=device, generator=gen
                )
                / FUSED_MQA_TOPK_HEADS
            )
            starts = torch.zeros(max_rows, dtype=torch.int32, device=device)
            output = torch.empty(
                max_rows, compile_key.topk, dtype=torch.int32, device=device
            )
            for rows, keys in shapes:
                # Every row sees all keys: adaptive pruning then plans and
                # launches both its scan and its pruning kernels.
                ends = torch.full((rows,), keys, dtype=torch.int32, device=device)
                run(
                    q[:rows],
                    k[:keys],
                    scales[:keys],
                    weights[:rows],
                    starts[:rows],
                    ends,
                    output[:rows],
                )
            torch.accelerator.synchronize()
        logger.info(
            "Warmed up the %s fused MQA+top-k prefill backend (top-k %d) on "
            "cuda:%d: %d shapes.",
            compile_key.backend,
            compile_key.topk,
            compile_key.device_index,
            len(shapes),
        )
        self._compiled_cache[compile_key] = True


FUSED_MQA_TOPK_WARMUP = FusedMqaTopkWarmup()
