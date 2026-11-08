# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state, VMM memory ops, and LD_PRELOAD env setup for the
foundry vLLM integration.

Merged from former `state.py` + `lifecycle.py` + `memory.py` +
`distributed.py`. All process-local globals and the primitives that
touch foundry's C extension live here.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from foundry import ops as fops
from foundry.allocation_region import parse_size

import vllm.envs as envs
from vllm.compilation.foundry_adapter.config import (
    CUDAGraphExtensionMode,
    _compute_rank_from_parallel_config,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

log = init_logger("vllm.foundry.runtime")


# ---------------------------------------------------------------------------
# Worker-scope state (per-process, set by setup_graph_extension)
# ---------------------------------------------------------------------------


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    # LOAD: strong-reference table keyed by GraphIdentifier. The upstream
    # CUDAGraphWrapper stores `entry.output = weak_ref_tensors(output)`, so
    # without a strong ref here the VMM-backed output tensors returned by
    # `FoundryCUDAGraph.finish_one_graph_load` get GC'd between captures and
    # the next graph replay writes to freed memory → CUDA illegal memory
    # access. Do not remove.
    loaded_graphs: dict = field(default_factory=dict)


_state: CUDAGraphExtensionState | None = None
_capture_active = False


def set_capture_active(active: bool) -> None:
    global _capture_active
    _capture_active = active


def is_capture_active() -> bool:
    return _capture_active


def get_state() -> CUDAGraphExtensionState | None:
    return _state


# ---------------------------------------------------------------------------
# VMM region setup (called from patched init_worker_distributed_environment)
# ---------------------------------------------------------------------------


def prepare_weight_cache_daemon(config_path: str, local_rank: int) -> tuple[int, int]:
    """Allocate exportable VMM weights for later Foundry IPC engines.

    Give the daemon a region disjoint from the engine's so the stock IPC hook
    can map weights at their exporter addresses. Legacy CUDA IPC mappings have
    process-dependent addresses and cannot be embedded in a reusable archive.
    """
    from vllm.compilation.foundry_adapter.config import CUDAGraphExtensionConfig

    cfg = CUDAGraphExtensionConfig.from_toml(config_path)
    if cfg.mode != CUDAGraphExtensionMode.SAVE:
        raise ValueError("Graph preparation requires a Foundry SAVE configuration")
    if cfg.weight_cache_size is None:
        raise ValueError("Set Foundry weight_cache_size for the daemon's weight pool")
    torch.accelerator.set_device_index(local_rank)
    region_size = parse_size(cfg.region_size)
    daemon_base = cfg.base_addr - region_size
    if daemon_base <= 0:
        raise ValueError("Foundry base_addr must leave room for the daemon region")
    fops.set_allocation_region(daemon_base, region_size)
    pool_size = parse_size(cfg.weight_cache_size)
    if pool_size <= 0 or pool_size > region_size:
        raise ValueError("weight_cache_size must be positive and fit in region_size")
    # Small cuMemAddressReserve requests can ignore the requested address.
    # A single large exported chunk avoids one reservation per Torch storage.
    if not fops.preallocate_region(pool_size):
        raise RuntimeError("Failed to preallocate the daemon's Foundry weight pool")
    # The temporary engines own graph/module archives. The persistent daemon
    # only exports weights, and must not publish artifacts into their archive.
    fops.set_pack_fatbins_on_exit(False)
    log.info("[foundry] weight daemon initialized exportable VMM allocations")
    return daemon_base, daemon_base + pool_size


def check_ipc_weight_layout(model: torch.nn.Module) -> None:
    """Reject archived kernels whose external weight pointers have changed."""
    cfg = get_config()
    assert cfg is not None and cfg.workspace_dir is not None
    path = Path(cfg.workspace_root) / (
        f"{Path(cfg.workspace_dir).name}_ipc_weight_layout.json"
    )
    tensors = [*model.named_parameters(), *model.named_buffers()]
    layout = {
        name: {
            "address": tensor.data_ptr(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in tensors
        if tensor.is_cuda
    }
    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        path.write_text(json.dumps(layout, sort_keys=True), encoding="utf-8")
    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != layout:
            changed = sorted(
                name
                for name in saved.keys() | layout.keys()
                if saved.get(name) != layout.get(name)
            )
            raise RuntimeError(
                "Foundry IPC weight layout changed before graph restoration: "
                f"{changed[:3]}. Prepare the weight daemon with the Foundry "
                "SAVE configuration before preparing graphs."
            )
        log.info("[foundry] validated IPC weight layout (%d tensors)", len(layout))


def setup_graph_extension(parallel_config: ParallelConfig) -> None:
    """VMM region + (LOAD only) CUDA modules. Called BEFORE NCCL."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return

    t_total = time.perf_counter()
    rank = _compute_rank_from_parallel_config(parallel_config)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    log.info("[foundry] rank=%d workspace_dir=%s", rank, workspace_dir)

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        fops.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            # rank_0 fallback is only safe for single-rank setups.
            # Under TP/PP/DP > 1 each rank has its own device addresses,
            # so loading rank 0's graphs on rank N would produce wrong
            # pointers and likely crash on first replay.
            is_single_rank = (
                parallel_config.tensor_parallel_size == 1
                and parallel_config.pipeline_parallel_size == 1
                and parallel_config.data_parallel_size == 1
                and parallel_config.nnodes == 1
            )
            fallback = Path(cfg.workspace_root) / "rank_0"
            if is_single_rank and fallback.exists():
                log.warning(
                    "Workspace %s missing; falling back to %s (safe only for "
                    "single-rank setups)",
                    workspace_dir,
                    fallback,
                )
                cfg.workspace_dir = str(fallback)
            else:
                raise RuntimeError(
                    f"Workspace for rank {rank} ({workspace_dir}) does not "
                    f"exist. For multi-rank LOAD, re-run SAVE with the same "
                    f"parallelism topology."
                )
        log.info("[foundry] LOAD: loading CUDA modules from %s", cfg.workspace_dir)
        t = time.perf_counter()
        fops.load_cuda_modules_and_libraries(cfg.workspace_dir)
        log.info(
            "[foundry TIMING] load_cuda_modules_and_libraries: %.3f s",
            time.perf_counter() - t,
        )

    # set_allocation_region MUST come after load_cuda_modules (which does
    # cuCtxCreate).
    region_size = parse_size(cfg.region_size)
    log.info(
        "[foundry] Setting up VMM region at %s size %s",
        hex(cfg.base_addr),
        cfg.region_size,
    )
    fops.set_allocation_region(cfg.base_addr, region_size)
    # NOTE(liuxs): workspace init is in prepare_graph_capture
    # # Bring cuBLAS handle up so its tiny bring-up lands in scratch.
    # _ = torch._C._cuda_getCurrentBlasHandle()

    _state = CUDAGraphExtensionState()
    log.info(
        "[foundry TIMING] setup_graph_extension total: %.3f s",
        time.perf_counter() - t_total,
    )


def skip_to_scratch_boundary() -> None:
    """Advance allocator past scratch region. Called AFTER NCCL."""
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = fops.get_current_alloc_offset()
    if current > scratch:
        log.warning(
            "[foundry] Current offset %d > scratch_space_size %d. Increase "
            "scratch_space_size.",
            current,
            scratch,
        )
        return
    log.info(
        "[foundry] Skip to scratch boundary: %d bytes (current was %d)",
        scratch,
        current,
    )
    fops.set_current_alloc_offset(scratch)


def capture_final_alloc_offset() -> int:
    """Record per-rank watermark after all graph captures on SAVE."""
    cfg = get_config()
    offset = fops.get_current_alloc_offset()
    log.info(
        "[foundry] Captured final_alloc_offset: %d bytes (%.2f GB)",
        offset,
        offset / (1024**3),
    )
    if cfg is not None and cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        # Persist the live VMM ranges, not only the cursor watermark.  A
        # contiguous preallocation can end in the middle of a CUDA allocator
        # segment; LOAD would then try to map an overlapping tail and either
        # fall back to a different address or corrupt the restored graph.
        ranges = fops.get_live_region_ranges()
        with open(path, "w") as f:
            json.dump(
                {
                    "final_alloc_offset": offset,
                    "live_ranges": ranges,
                },
                f,
            )
        log.info("[foundry] Saved per-rank final_alloc_offset to %s", path)
    return offset


# ---------------------------------------------------------------------------
# VMM preallocation (LOAD only)
# ---------------------------------------------------------------------------


def preallocate_for_load_mode() -> None:
    """One cuMemCreate + cuMemMap covering the deterministic region."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    # Each rank records its own archived allocation span.
    final = 0
    live_ranges = None
    if cfg.workspace_dir is not None:
        p = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
                final = data.get("final_alloc_offset", 0)
                live_ranges = data.get("live_ranges")
    if final <= 0:
        return

    current = fops.get_current_alloc_offset()
    remaining = final - current
    if remaining <= 0:
        log.info("[foundry] LOAD: no preallocation needed")
        return

    log.info(
        "[foundry] LOAD: preallocating %d bytes (%.2f GB)",
        remaining,
        remaining / (1024**3),
    )
    t = time.perf_counter()
    # Graph kernel parameters can retain pointers to temporary workspaces that
    # were freed after SAVE capture.  ``live_ranges`` intentionally excludes
    # those allocations, so mapping only the live set leaves their addresses
    # backed by unmapped or unrelated memory on LOAD.  This call runs before
    # weight restoration, while the deterministic region is still empty after
    # the scratch boundary, so the complete span can be mapped safely and all
    # archived graph pointers remain valid.
    ok = fops.preallocate_region(remaining)
    if not ok and live_ranges is not None:
        ranges = [
            (int(offset), int(size))
            for offset, size in live_ranges
            if int(offset) + int(size) > current and int(offset) < final
        ]
        log.warning(
            "[foundry] LOAD: full-span preallocation failed; falling back to "
            "live ranges (archived temporary pointers may be unavailable)"
        )
        ok = fops.preallocate_ranges(ranges, final)
    if ok:
        log.info("[foundry TIMING] preallocate_region: %.3f s", time.perf_counter() - t)
    else:
        log.warning("[foundry] preallocate_region failed for %d bytes", remaining)


# ---------------------------------------------------------------------------
# Graph-capture workspace pinning (SAVE + LOAD)
# ---------------------------------------------------------------------------


def _preallocate_attention_workspaces(metadata_builders: list | None) -> None:
    try:
        from vllm.v1.attention.backends.flashinfer import (
            FlashInferMetadataBuilder,
            _get_trtllm_workspace_buffer,
        )
    except ImportError:
        return
    _get_trtllm_workspace_buffer()
    if metadata_builders:
        for b in metadata_builders:
            if isinstance(b, FlashInferMetadataBuilder):
                b._get_workspace_buffer()


def _preallocate_flashinfer_allreduce_workspaces(vllm_config) -> None:
    """Create communication workspaces before restoring captured graphs.

    LOAD skips the compilation warmup that normally creates these lazily.  A
    restored graph can still reference the workspace, so it must be created at
    the same point in the deterministic allocation sequence as SAVE.
    """
    if not envs.VLLM_ALLREDUCE_USE_FLASHINFER:
        return
    try:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            get_fi_ar_quant_workspace,
            get_fi_ar_workspace,
        )
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        max_size = vllm_config.compilation_config.pass_config.flashinfer_max_size(
            tp_size
        )
        if max_size is None or tp_size <= 1:
            return
        hidden_dim = vllm_config.model_config.get_hidden_size()
        dtype = vllm_config.model_config.dtype
        element_size = torch.empty((), dtype=dtype, device="cpu").element_size()
        max_token_num = max_size // (hidden_dim * element_size)
        kwargs = dict(
            world_size=tp_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            group=get_tp_group().cpu_group,
        )
        get_fi_ar_workspace(**kwargs)
        get_fi_ar_quant_workspace(**kwargs)
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        log.debug(
            "[foundry] FlashInfer all-reduce workspace preallocation skipped: %s",
            e,
        )


def _eager_inductor_lazy_init(device: torch.device) -> None:
    """Pre-fire `torch._inductor.fx_passes.joint_graph.lazy_init` outside
    any CUDA-graph capture window.

    The first in-capture compile inside `capture_model` triggers
    `joint_graph.lazy_init` which calls `_sfdp_init` — that builds SDPA
    pattern templates by copying CPU constants onto the device, which is
    illegal inside a capture window:

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        graph capture unless the CPU tensor is pinned.

    Pre-warming retires the lazy init before capture starts.

    Why only `joint_graph` and not the other `lazy_init` functions in
    `torch._inductor.fx_passes.*`: `post_grad.lazy_init` and friends
    are also `@init_once_fakemode` (`@functools.cache`-keyed on
    `input_device`), but inductor's own `post_grad_passes` calls them
    with **no argument** (`input_device=None`). If we pre-warm with the
    indexed `cuda:N` device the cache keys don't match, both calls run,
    and they re-register the same patterns:

        torch._inductor.exc.InductorError: Duplicate pattern: amax_default

    Only `joint_graph.lazy_init` is dangerous to leave for capture-time
    (its body actually allocates on the device); the others just register
    patterns and are safe to leave to inductor.
    """
    try:
        from torch._inductor.fx_passes.joint_graph import lazy_init
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init not importable: %s", e)
        return
    try:
        lazy_init(device)
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init(%s) skipped: %s", device, e)


def prepare_graph_capture(
    metadata_builders: list | None = None,
    device: torch.device | None = None,
    vllm_config=None,
) -> None:
    """Force cuBLAS + attention workspaces to land at deterministic offsets,
    and pre-warm torch._inductor's pattern-matcher lazy init so the first
    in-capture compile doesn't try to copy CPU tensors onto the device.

    `device` is the model_runner's device; passing it lets the inductor
    lazy-init cache key match the in-capture compile path.
    """
    import foundry as foundry_pkg

    if device is not None:
        _eager_inductor_lazy_init(device)
    fops.preallocate_cublas_workspaces()
    if vllm_config is not None:
        _preallocate_flashinfer_allreduce_workspaces(vllm_config)
    _preallocate_attention_workspaces(metadata_builders)
    if foundry_pkg.graph.default_capture_stream is None:
        foundry_pkg.graph.default_capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(foundry_pkg.graph.default_capture_stream):
        fops.preallocate_cublas_workspaces()
        _preallocate_attention_workspaces(metadata_builders)


# ---------------------------------------------------------------------------
# Subprocess env setup (LD_PRELOAD + FOUNDRY_MODE consumed by foundry/csrc/hook.cpp)
# ---------------------------------------------------------------------------


def setup_ld_preload_env() -> None:
    """Mutate os.environ so child processes inherit the foundry hook.

    Idempotent, non-restoring — keeps the mutation for the parent's
    lifetime. Called from patched subprocess-spawn sites.

    Also stamps `FOUNDRY_SPAWN_T0_NS` right before the child fork so
    the child can log `spawn → install_hooks` wall time (dominated by
    Python + torch + vllm cold import with LD_PRELOAD'd hook).
    """
    current = os.environ.get("LD_PRELOAD", "")
    hook = get_hook_library_path()
    if hook and hook not in current:
        current = f"{hook}:{current}" if current else hook
    nvs = get_nvshmem_host_path()
    if nvs and nvs not in current:
        current = f"{nvs}:{current}" if current else nvs
    if current:
        os.environ["LD_PRELOAD"] = current

    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value

    # Early region reserve: on LOAD the hook claims [base, base+region_size)
    # right after context creation, before module loading can trigger the
    # CUDA driver's lazily-placed VA arena. Size must equal region_size
    # exactly so setup_graph_extension's set_allocation_region call is an
    # idempotent no-op against the early reservation.
    cfg = get_config()
    if cfg is not None and mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_EARLY_RESERVE_BASE"] = hex(cfg.base_addr)
        os.environ["FOUNDRY_EARLY_RESERVE_SIZE"] = str(parse_size(cfg.region_size))

    # Overwrite every spawn site — child reads it on entry.
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())

    log.info("[foundry] LD_PRELOAD set (mode=%s)", mode.value)
