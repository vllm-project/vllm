# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import atexit
import os
import random
import threading
from functools import cache
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config.compilation import PassConfig
from vllm.distributed.parallel_state import get_node_count
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The empirical value for small batch
PDL_ADVANCE_LAUNCH_TOKENS = 16

fi_ar_available = False
try:
    import flashinfer.comm as flashinfer_comm  # type: ignore[no-redef]
    from flashinfer.comm.mnnvl import (
        TorchDistBackend,  # type: ignore[import-not-found, no-redef]
    )

    fi_ar_available = hasattr(flashinfer_comm, "allreduce_fusion")
except ImportError:
    pass

# Workspace for standalone allreduce and non-quant ar+rms fusion
_fi_ar_workspace = None
# Extra workspace for quant fusion patterns. This may use either the primary
# allreduce backend or a fallback backend when the primary workspace is not
# available on the current topology.
_fi_ar_quant_workspace = None
# Extra workspace for the MoE finalize fusion, which is trtllm-only and so
# cannot ride on an mnnvl workspace.
_fi_ar_moe_finalize_workspace = None
_fi_ar_workspace_groups: dict[int, ProcessGroup] = {}


def _create_workspace(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """Create a flashinfer allreduce workspace, returning None on failure."""
    comm_backend = TorchDistBackend(group=group)
    rng_state = random.getstate()
    try:
        random.seed(int.from_bytes(os.urandom(16), byteorder="big"))
        workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=comm_backend,
            group=group,
        )
        if backend == "mnnvl" and not getattr(workspace, "mc_ptr", 0):
            workspace.destroy()
            logger.warning_once(
                "FlashInfer MNNVL multicast is unavailable on the current topology."
            )
            return None
    except Exception as e:
        if "multicast" in str(e).lower():
            logger.warning_once(
                "Failed to initialize FlashInfer All Reduce workspace: %s. "
                "This is expected on GPUs without NVSwitch (e.g., NVLink "
                "bridge-only or PCIe topologies).",
                e,
            )
        else:
            logger.warning_once(
                "Failed to initialize FlashInfer All Reduce workspace: %s.",
                e,
            )
        return None
    finally:
        random.setstate(rng_state)
    workspace_id = id(workspace)
    workspace_group = _fi_ar_workspace_groups.get(workspace_id)
    if workspace_group is not None and workspace_group is not group:
        raise RuntimeError(
            "FlashInfer returned an all-reduce workspace already associated "
            "with a different process group"
        )
    _fi_ar_workspace_groups[workspace_id] = group
    logger.debug(
        "Initialized FlashInfer All Reduce workspace: backend=%s, "
        "world_size=%d, rank=%d, max_token_num=%d, hidden_dim=%d, dtype=%s",
        backend,
        world_size,
        rank,
        max_token_num,
        hidden_dim,
        dtype,
    )
    return workspace


def _resolve_fi_ar_backend() -> tuple[str, bool]:
    """Resolve the flashinfer allreduce backend for the current setup.

    Returns:
        A ``(backend, allow_trtllm_fallback)`` tuple. ``allow_trtllm_fallback``
        is True only when ``auto`` selects mnnvl for a single node, so that
        workspace creation can fall back to trtllm on single-node topologies
        without NVSwitch multicast support (where mnnvl is unavailable).
    """
    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    if backend != "auto":
        logger.debug_once("Using flashinfer allreduce backend: %s", backend)
        return backend, False

    # Default to mnnvl for both single- and multi-node setups. The mnnvl
    # cudagraph hang that previously forced single-node to trtllm
    # (https://github.com/vllm-project/vllm/issues/35772) was fixed upstream in
    # FlashInfer (>= 0.6.12, vLLM pins 0.6.15), so mnnvl is safe here. trtllm
    # does not support multi-node allreduce, so mnnvl is required there anyway.
    # mnnvl needs NVSwitch multicast; on single-node topologies without it,
    # fall back to trtllm so fused allreduce stays enabled.
    backend = "mnnvl"
    allow_trtllm_fallback = get_node_count() == 1

    logger.debug_once("Auto-selected flashinfer allreduce backend: %s", backend)
    return backend, allow_trtllm_fallback


def get_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """
    Return the allreduce workspace for non-quant patterns, initializing if needed.

    Used by AllReduceFusionPass (non-quant patterns) and FlashInferAllReduce
    for standalone allreduce. Backend is controlled by
    VLLM_FLASHINFER_ALLREDUCE_BACKEND env var.
    """
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        return _fi_ar_workspace

    backend, allow_trtllm_fallback = _resolve_fi_ar_backend()

    if get_node_count() > 1 and backend == "trtllm":
        raise ValueError(
            "Flashinfer allreduce is not supported for multi-node allreduce with "
            "'trtllm' backend. Please use 'mnnvl' backend instead."
        )

    def _get_or_create(be: str):
        # Reuse the quant workspace if it was already created with the same backend
        if _fi_ar_quant_workspace is not None and _fi_ar_quant_workspace.backend == be:
            return _fi_ar_quant_workspace
        return _create_workspace(
            be, world_size, rank, max_token_num, hidden_dim, dtype, group
        )

    _fi_ar_workspace = _get_or_create(backend)
    if _fi_ar_workspace is None and allow_trtllm_fallback and backend != "trtllm":
        logger.warning_once(
            "FlashInfer mnnvl allreduce workspace unavailable (likely no NVSwitch "
            "multicast support); falling back to trtllm backend for single node."
        )
        backend = "trtllm"
        _fi_ar_workspace = _get_or_create(backend)

    if _fi_ar_workspace is not None:
        logger.info_once(
            "Initialized FlashInfer Allreduce norm fusion workspace "
            f"with backend={backend}"
        )
    else:
        logger.warning_once(
            "Failed to initialize FlashInfer Allreduce norm fusion workspace "
            f"with backend={backend}"
        )

    return _fi_ar_workspace


def get_fi_ar_quant_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """
    Return the allreduce workspace for quant patterns, initializing if needed.

    Backend is controlled by VLLM_FLASHINFER_ALLREDUCE_BACKEND env var, matching
    non-quant fusion. With ``auto`` this prefers mnnvl and falls back to trtllm
    only on single-node topologies where mnnvl multicast is unavailable.
    """
    global _fi_ar_quant_workspace
    if _fi_ar_quant_workspace is not None:
        return _fi_ar_quant_workspace

    backend, allow_trtllm_fallback = _resolve_fi_ar_backend()

    if get_node_count() > 1 and backend == "trtllm":
        raise ValueError(
            "Flashinfer allreduce quantization fusion is not supported for "
            "multi-node allreduce with 'trtllm' backend. Please use 'mnnvl' "
            "backend instead."
        )

    # Reuse the non-quant workspace if it was already created with the same
    # backend.
    if _fi_ar_workspace is not None and _fi_ar_workspace.backend == backend:
        _fi_ar_quant_workspace = _fi_ar_workspace
        return _fi_ar_quant_workspace

    if (
        _fi_ar_workspace is not None
        and _fi_ar_workspace.backend == "trtllm"
        and allow_trtllm_fallback
        and backend != "trtllm"
    ):
        _fi_ar_quant_workspace = _fi_ar_workspace
        return _fi_ar_quant_workspace

    _fi_ar_quant_workspace = _create_workspace(
        backend, world_size, rank, max_token_num, hidden_dim, dtype, group
    )
    if _fi_ar_quant_workspace is None and allow_trtllm_fallback and backend != "trtllm":
        logger.warning_once(
            "FlashInfer mnnvl allreduce quantization fusion workspace unavailable "
            "(likely no NVSwitch multicast support); falling back to trtllm "
            "backend for single node."
        )
        backend = "trtllm"
        if _fi_ar_workspace is not None and _fi_ar_workspace.backend == backend:
            _fi_ar_quant_workspace = _fi_ar_workspace
        else:
            _fi_ar_quant_workspace = _create_workspace(
                backend, world_size, rank, max_token_num, hidden_dim, dtype, group
            )

    if _fi_ar_quant_workspace is not None:
        logger.info_once(
            "Initialized FlashInfer Allreduce norm quantization "
            f"fusion workspace with backend={backend}"
        )
    else:
        logger.warning_once(
            "Failed to initialize FlashInfer Allreduce norm quantization "
            f"fusion workspace with backend={backend}"
        )

    return _fi_ar_quant_workspace


# CuTe DSL vector width for bf16 (flashinfer's VEC_BF16); the HT kernel shards
# hidden across ``consumer_threads * _CUTEDSL_VEC * vectors_per_thread`` and
# rejects a hidden it does not divide.
_CUTEDSL_VEC = 8
_CUDA_MAX_BLOCK_THREADS = 1024
_WARP = 32


def _ht_shard_threads(hidden_dim: int, base_preset) -> tuple[int, int] | None:
    """consumer_threads/vectors_per_thread whose shard divides ``hidden_dim``.

    The shipped profiles tile a whole row per shard (8192 = 512 * 8 * 2), so
    keep the tuned vectors_per_thread and scale the thread count; fall back to
    any warp-multiple pair that fits the block limit.
    """
    reserved = (2 + base_preset.reduction_warps) * _WARP
    for vpt in (base_preset.vectors_per_thread, 1, 2, 3, 4):
        threads, rem = divmod(hidden_dim, _CUTEDSL_VEC * vpt)
        if rem or threads % _WARP or threads <= 0:
            continue
        if threads % base_preset.rms_token_groups:
            continue
        if threads + reserved <= _CUDA_MAX_BLOCK_THREADS:
            return threads, vpt
    return None


def _moe_finalize_cutedsl_config(
    tp_size: int, hidden_dim: int, top_k: int, dtype: torch.dtype
):
    """A CuTe DSL config covering this shape, or None if none can be built.

    Profiles are keyed on ``(tp_size, hidden_size, top_k, dtype)`` and the
    shipped ones only cover tp 8/16 with hidden 8192 and top_k 10, so anything
    else is derived: the LL collective's cluster spans the TP group, and the HT
    kernel's shard has to divide the hidden size.
    """
    import dataclasses

    from flashinfer.comm.mnnvl_cutedsl_ar import DEFAULT_CONFIG, MNNVLCuteDSLConfig

    for profile in DEFAULT_CONFIG.profiles:
        if (
            profile.tp_size == tp_size
            and profile.hidden_size == hidden_dim
            and profile.top_k == top_k
            and profile.dtype == dtype
        ):
            return DEFAULT_CONFIG

    base = min(
        (p for p in DEFAULT_CONFIG.profiles if p.dtype == dtype),
        key=lambda p: abs(p.tp_size - tp_size),
        default=None,
    )
    if base is None:
        return None

    def retune(routes):
        targets = []
        for target in routes.targets:
            preset = target.preset
            collective = getattr(preset, "collective", None)
            if collective is not None and hasattr(collective, "cluster_size"):
                preset = dataclasses.replace(
                    preset,
                    collective=dataclasses.replace(collective, cluster_size=tp_size),
                )
            if hasattr(preset, "consumer_threads"):
                shard = _ht_shard_threads(hidden_dim, preset)
                if shard is None:
                    return None
                threads, vpt = shard
                preset = dataclasses.replace(
                    preset, consumer_threads=threads, vectors_per_thread=vpt
                )
            targets.append(dataclasses.replace(target, preset=preset))
        return dataclasses.replace(routes, targets=tuple(targets))

    finalize_routes = retune(base.finalize_routes)
    all_reduce_routes = retune(base.all_reduce_routes)
    if finalize_routes is None or all_reduce_routes is None:
        logger.warning_once(
            "No CuTe DSL HT shard divides hidden_size=%d; MoE tail fusion off.",
            hidden_dim,
        )
        return None

    logger.info_once(
        "Derived a CuTe DSL profile for tp=%d hidden=%d top_k=%d %s from the "
        "shipped tp=%d profile; its M-range boundaries are not tuned for this "
        "shape.",
        tp_size,
        hidden_dim,
        top_k,
        dtype,
        base.tp_size,
    )
    return MNNVLCuteDSLConfig(
        profiles=(
            dataclasses.replace(
                base,
                tp_size=tp_size,
                hidden_size=hidden_dim,
                top_k=top_k,
                finalize_routes=finalize_routes,
                all_reduce_routes=all_reduce_routes,
            ),
        )
    )


@cache
def has_fi_ar_moe_finalize_backend() -> bool:
    """Whether flashinfer ships the mnnvl CuTe DSL allreduce the fused tail needs.

    A layer asks before declaring that its experts may leave the MoE output
    unfinalized, so a build without the backend never gets that far.
    """
    try:
        from flashinfer.comm.mnnvl_cutedsl_ar import (  # noqa: F401
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )
    except ImportError:
        logger.warning_once(
            "FlashInfer has no mnnvl CuTe DSL allreduce; MoE tail fusion needs "
            "flashinfer >= 0.6.18."
        )
        return False
    return True


def _create_moe_finalize_cutedsl_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
    *,
    top_k: int,
    routed_scaling_factor: float,
    rms_eps: float,
    weight_bias: float,
    include_shared_expert: bool,
):
    """Build the mnnvl CuTe DSL workspace, or None if the shape is unsupported.

    Its config resolves a StaticProfile that has to match
    ``(tp_size, hidden_size, top_k, dtype)`` exactly; the shipped profiles cover
    tp 8/16 with hidden_size 8192, top_k 10, bf16, so anything else raises and
    lands the caller back on finalizing inside the MoE kernel.
    """
    if not has_fi_ar_moe_finalize_backend():
        return None
    from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

    try:
        config = _moe_finalize_cutedsl_config(world_size, hidden_dim, top_k, dtype)
        if config is None:
            return None
        return MNNVLCuteDSLAllReduceFusionWorkspace(
            world_size,
            rank,
            max_token_num,
            hidden_dim,
            dtype,
            group=group,
            top_k=top_k,
            rms_eps=rms_eps,
            routed_scaling_factor=routed_scaling_factor,
            weight_bias=weight_bias,
            include_shared_expert=include_shared_expert,
            config=config,
        )
    except Exception as e:
        logger.warning_once(
            "No mnnvl CuTe DSL workspace for tp=%d hidden=%d top_k=%d %s: %s",
            world_size,
            hidden_dim,
            top_k,
            dtype,
            e,
        )
        return None


def get_fi_ar_moe_finalize_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
    *,
    top_k: int,
    routed_scaling_factor: float,
    rms_eps: float,
    weight_bias: float,
    include_shared_expert: bool,
):
    """
    Return the workspace for the MoE finalize + allreduce + norm fusion.

    mnnvl CuTe DSL is the one supported backend, so this does not follow
    VLLM_FLASHINFER_ALLREDUCE_BACKEND and never reuses the general all-reduce
    workspaces: theirs is a different class, compiled without the fusion's
    semantics. Whether the fusion runs at all is VLLM_ENABLE_MOE_TAIL_FUSION's
    call, checked by the caller.
    """
    global _fi_ar_moe_finalize_workspace
    if _fi_ar_moe_finalize_workspace is not None:
        return _fi_ar_moe_finalize_workspace

    if not fi_ar_available:
        return None

    _fi_ar_moe_finalize_workspace = _create_moe_finalize_cutedsl_workspace(
        world_size,
        rank,
        max_token_num,
        hidden_dim,
        dtype,
        group,
        top_k=top_k,
        rms_eps=rms_eps,
        routed_scaling_factor=routed_scaling_factor,
        weight_bias=weight_bias,
        include_shared_expert=include_shared_expert,
    )
    if _fi_ar_moe_finalize_workspace is not None:
        logger.info_once(
            "Initialized FlashInfer MoE finalize fusion workspace "
            "(mnnvl-cutedsl, top_k=%d hidden=%d %s)",
            top_k,
            hidden_dim,
            dtype,
        )
    return _fi_ar_moe_finalize_workspace


_fi_ar_workspace_lock = threading.Lock()


def destroy_fi_ar_workspace():
    global _fi_ar_workspace, _fi_ar_quant_workspace, _fi_ar_moe_finalize_workspace
    with _fi_ar_workspace_lock:
        destroyed: list[int] = []
        for workspace in (
            _fi_ar_workspace,
            _fi_ar_quant_workspace,
            _fi_ar_moe_finalize_workspace,
        ):
            if workspace is None or id(workspace) in destroyed:
                continue
            workspace.destroy()
            destroyed.append(id(workspace))

        _fi_ar_workspace = _fi_ar_quant_workspace = None
        _fi_ar_moe_finalize_workspace = None
        _fi_ar_workspace_groups.clear()


def _fi_ar_workspaces_for_group(group: ProcessGroup) -> list[Any]:
    workspaces: list[Any] = []
    for workspace in (
        _fi_ar_workspace,
        _fi_ar_quant_workspace,
        _fi_ar_moe_finalize_workspace,
    ):
        if workspace is not None and not any(w is workspace for w in workspaces):
            workspaces.append(workspace)

    group_workspaces = []
    for workspace in workspaces:
        if workspace is None:
            continue
        workspace_group = _fi_ar_workspace_groups.get(id(workspace))
        if workspace_group is None:
            raise RuntimeError(
                "FlashInfer all-reduce workspace process group was not retained"
            )
        if workspace_group is group:
            group_workspaces.append(workspace)
    return group_workspaces


def checkpoint_prepare_fi_ar_workspaces(group: ProcessGroup) -> None:
    for workspace in _fi_ar_workspaces_for_group(group):
        workspace.checkpoint_prepare()


def checkpoint_restore_fi_ar_workspaces(group: ProcessGroup) -> None:
    for workspace in _fi_ar_workspaces_for_group(group):
        workspace.checkpoint_restore(TorchDistBackend(group=group))


atexit.register(destroy_fi_ar_workspace)


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
    ):
        self.disabled = True

        if not fi_ar_available:
            logger.info(
                "FlashInfer All Reduce is disabled because flashinfer is not available"
            )
            return

        if not current_platform.is_cuda():
            logger.info(
                "FlashInfer All Reduce is disabled because it requires CUDA platform"
            )
            return

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = device
        if self.world_size == 1:
            return

        # Use the same threshold as the allreduce-rms fusion pass
        # TODO: tune the threshold
        MiB = 1024 * 1024
        max_workspace_size = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(
            self.world_size, None
        )
        if not max_workspace_size:
            logger.warning(
                "FlashInfer All Reduce is disabled because it "
                "is not supported for world_size=%d.",
                self.world_size,
            )
            return
        self.max_workspace_size = max_workspace_size * MiB
        self.max_num_tokens = 0
        self.disabled = False

    def _ensure_workspace(self, hidden_dim: int, dtype: torch.dtype) -> bool:
        """Ensure the all reduce workspace is initialized."""
        if self.max_num_tokens == 0:
            element_size = torch.tensor([], dtype=dtype, device="cpu").element_size()
            self.max_num_tokens = self.max_workspace_size // (hidden_dim * element_size)
        workspace = get_fi_ar_workspace(
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=self.max_num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            group=self.group,
        )
        if workspace is None:
            self.disabled = True
            return False
        return True

    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda:
            return False

        if not input_tensor.is_contiguous():
            return False

        if len(input_tensor.shape) != 2:
            return False

        num_tokens, hidden_dim = input_tensor.shape
        if not self.max_num_tokens:
            element_size = torch.tensor([], dtype=input_tensor.dtype).element_size()
            self.max_num_tokens = self.max_workspace_size // (hidden_dim * element_size)

        if num_tokens > self.max_num_tokens:
            return False

        return self._ensure_workspace(hidden_dim, input_tensor.dtype)

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = input_tensor.shape
        workspace = get_fi_ar_workspace(
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=self.max_num_tokens,
            hidden_dim=hidden_dim,
            dtype=input_tensor.dtype,
            group=self.group,
        )
        return flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
        )

    def destroy(self):
        if not self.disabled:
            destroy_fi_ar_workspace()
