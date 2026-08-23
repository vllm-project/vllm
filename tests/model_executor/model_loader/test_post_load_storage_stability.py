# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry-wide post-load storage stability.

Weight reload reruns ``process_weights_after_loading`` (PWAL). Registered
parameters and buffers are protected by copy-back; anything else a PWAL
pass binds -- kernel attributes, bare layer attributes, tensors captured
inside callables -- can be silently rebound, leaving captured CUDA graphs
pointing at freed or stale memory (#48312 category 1).

The unit of that bug is PWAL idempotency, not reload. A reload is
restore-to-meta -> load -> PWAL -> copy-back, and only PWAL rebinds runtime
storage. So running PWAL twice over a synthetic layer reproduces the
rebinding **without a checkpoint, an engine, or a second weight load** --
which is what makes this affordable as a per-commit check rather than a
production gate.

Coverage is driven off the kernel registry, so a newly registered backend
is checked the day it lands: nothing to implement, nothing to declare, it
simply must not drift.

Known scope limits, deliberately not papered over:
  * Storage allocated on the FIRST FORWARD rather than during post-load is
    invisible here. The MoE permute scratch that produced a reproduced
    illegal-memory-access on H200 is exactly that shape; see
    ``test_reload_lazy_storage.py`` for the variant that covers it.
  * Bugs arising from the copy-back interaction (parameter-alias buffers,
    unloaded non-persistent buffers) need the real layerwise path and are
    not modelled by a synthetic layer.
  * ``object.__new__`` bypasses ``can_implement``, so a pass says the
    post-load path is stable, not that the kernel is deployable in that
    configuration.
"""

import functools
import os
import tempfile

import pytest
import torch

from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel, MPLinearLayerConfig)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

DEVICE = torch.device("cuda" if current_platform.is_cuda_alike() else "cpu")
SIZE_K = 256
SIZE_N = 256
GROUP_SIZE = 128
PACK_FACTOR = 8


@pytest.fixture(scope="module")
def dist_init():
    """Single-rank parallel state.

    vLLM's weight parameter classes query the TP group at construction, so
    the synthetic layer cannot be built without it. Defined locally rather
    than taken from tests/conftest.py so this file also runs against an
    installed wheel, where the repo-root conftest chain is not importable.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (cleanup_dist_env_and_memory,
                                  init_distributed_environment,
                                  initialize_model_parallel)

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)  # FileStore opens the path itself; leaving this open leaks
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl" if current_platform.is_cuda_alike() else "gloo",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def _collect(value, path: str, out: dict[str, int], depth: int) -> None:
    """data_ptr census over attributes, containers, partial bindings and
    closure cells -- the four ways a rebindable tensor hides."""
    if depth < 0:
        return
    if isinstance(value, torch.Tensor):
        if value.numel():
            out[path] = value.data_ptr()
        return
    if isinstance(value, functools.partial):
        for i, arg in enumerate(value.args):
            _collect(arg, f"{path}.args[{i}]", out, depth - 1)
        for key, arg in (value.keywords or {}).items():
            _collect(arg, f"{path}.kw[{key}]", out, depth - 1)
        return
    if callable(value) and getattr(value, "__closure__", None):
        for i, cell in enumerate(value.__closure__):
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            _collect(contents, f"{path}.closure[{i}]", out, depth - 1)
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _collect(item, f"{path}[{i}]", out, depth - 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect(item, f"{path}[{key!r}]", out, depth - 1)


# nn.Module's own registries. Registered parameters and buffers are exactly
# what reload copy-back restores, so they are out of scope here: this test
# is about the storage that escapes that protection. Walking them would
# also report the synthetic reload's fresh parameters as drift.
_MANAGED_CONTAINERS = ("_parameters", "_buffers", "_modules",
                       "_non_persistent_buffers_set")


def tensor_slots(*holders, depth: int = 3) -> dict[str, int]:
    out: dict[str, int] = {}
    for holder in holders:
        tag = type(holder).__name__
        for name, value in list(vars(holder).items()):
            if name.startswith("__") or name in _MANAGED_CONTAINERS:
                continue
            _collect(value, f"{tag}.{name}", out, depth)
    return out


def _drifted(before: dict[str, int], after: dict[str, int]) -> list[str]:
    return sorted(path for path, ptr in before.items()
                  if after.get(path) != ptr)


def make_config(has_g_idx: bool) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(SIZE_K, SIZE_N),
        partition_weight_shape=(SIZE_K, SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=GROUP_SIZE,
        zero_points=False,
        has_g_idx=has_g_idx,
    )


def build_layer(has_g_idx: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size = SIZE_K
    layer.input_size_per_partition = SIZE_K
    layer.output_size = SIZE_N
    layer.output_size_per_partition = SIZE_N
    layer.output_partition_sizes = [SIZE_N]
    layer.params_dtype = torch.float16
    layer.has_bias = False
    return layer


def load_weights(layer: torch.nn.Module, has_g_idx: bool, seed: int) -> None:
    """Bind checkpoint-format parameters, as a real load would.

    Called before each PWAL pass with a different seed: the second pass is
    the reload, and different values make sure a stability pass is not an
    identity no-op.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_groups = SIZE_K // GROUP_SIZE

    qweight = PackedvLLMParameter(
        data=torch.randint(-(2**31), 2**31 - 1,
                           (SIZE_K // PACK_FACTOR, SIZE_N),
                           dtype=torch.int32, generator=gen).to(DEVICE),
        input_dim=0, output_dim=1, packed_dim=0, packed_factor=PACK_FACTOR,
        weight_loader=lambda *a, **k: None,
    )
    scales = GroupQuantScaleParameter(
        data=torch.rand((num_groups, SIZE_N), dtype=torch.float16,
                        generator=gen).to(DEVICE) + 0.01,
        input_dim=0, output_dim=1,
        weight_loader=lambda *a, **k: None,
    )
    layer.register_parameter("qweight", qweight)
    layer.register_parameter("scales", scales)

    if has_g_idx:
        g_idx = BasevLLMParameter(
            data=(torch.arange(SIZE_K, dtype=torch.int32) // GROUP_SIZE
                  ).to(DEVICE),
            weight_loader=lambda *a, **k: None,
        )
        layer.register_parameter("g_idx", g_idx)


def registry_kernels() -> list[type[MPLinearKernel]]:
    """Every kernel any platform can select. Driving off the registry is
    what makes coverage automatic for backends added later."""
    seen: dict[str, type[MPLinearKernel]] = {}
    for kernels in _POSSIBLE_KERNELS.values():
        for kernel_cls in kernels:
            seen.setdefault(kernel_cls.__name__, kernel_cls)
    return sorted(seen.values(), key=lambda k: k.__name__)


@pytest.mark.parametrize("has_g_idx", [False, True],
                         ids=["plain", "act_order"])
@pytest.mark.parametrize("kernel_cls", registry_kernels(),
                         ids=lambda k: k.__name__)
def test_post_load_runtime_storage_is_stable(kernel_cls, has_g_idx,
                                             dist_init):
    config = make_config(has_g_idx)

    ok, reason = kernel_cls.can_implement(config)
    if not ok:
        pytest.skip(f"can_implement: {reason}")
    kernel = kernel_cls(config, "qweight", "scales", None,
                        "g_idx" if has_g_idx else None)

    layer = build_layer(has_g_idx)

    def post_load(seed: int):
        load_weights(layer, has_g_idx, seed)
        try:
            kernel.process_weights_after_loading(layer)
        except (RuntimeError, NotImplementedError, AssertionError,
                OSError, AttributeError) as e:
            pytest.skip(f"post-load needs unavailable device support: {e}")

    post_load(seed=0)
    before = tensor_slots(layer, kernel)
    post_load(seed=1)  # the reload
    after = tensor_slots(layer, kernel)

    drifted = _drifted(before, after)
    assert not drifted, (
        f"{kernel_cls.__name__} rebound runtime storage across a second "
        f"post-load pass: {drifted}. Captured graphs hold the previous "
        "addresses. Allocate through the layer's ReloadArena "
        "(vllm/model_executor/reload_arena.py) so the storage is reused."
    )


# ---------------------------------------------------------------------------
# Canaries. A sweep that only ever passes is indistinguishable from a sweep
# that cannot fail, so both directions are pinned here: the census must go
# red on a kernel that rebinds, and green on the same kernel once it
# allocates through the arena.
# ---------------------------------------------------------------------------


class _DriftingKernel:
    """Post-load that reallocates its workspace every pass -- the shape of
    the reproduced Marlin livelock and CUTLASS stride failures."""

    def process_weights_after_loading(self, layer):
        self.workspace = torch.zeros(16, dtype=torch.int32, device=DEVICE)
        layer.sort_indices = torch.arange(8, dtype=torch.int32, device=DEVICE)


class _ArenaKernel:
    """Same allocations, routed through the layer's arena."""

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.reload_arena import get_reload_arena
        arena = get_reload_arena(layer)
        self.workspace = arena.put(
            "workspace", torch.zeros(16, dtype=torch.int32, device=DEVICE))
        layer.sort_indices = arena.put(
            "sort_indices",
            torch.arange(8, dtype=torch.int32, device=DEVICE))


class _ClosureDriftingKernel:
    """Rebinding hidden inside a kernel-held callable: invisible to a plain
    attribute comparison, which is how the Machete act_perm bug escaped."""

    def process_weights_after_loading(self, layer):
        perm = torch.arange(8, dtype=torch.int32, device=DEVICE)
        self.apply_perm = lambda x: x[:, perm]


def test_canary_census_catches_a_rebinding_kernel():
    layer, kernel = torch.nn.Module(), _DriftingKernel()
    kernel.process_weights_after_loading(layer)
    before = tensor_slots(layer, kernel)
    kernel.process_weights_after_loading(layer)
    drifted = _drifted(before, tensor_slots(layer, kernel))
    assert len(drifted) == 2, drifted
    assert any("workspace" in p for p in drifted)
    assert any("sort_indices" in p for p in drifted)


def test_canary_census_catches_a_closure_captured_rebind():
    layer, kernel = torch.nn.Module(), _ClosureDriftingKernel()
    kernel.process_weights_after_loading(layer)
    before = tensor_slots(layer, kernel)
    kernel.process_weights_after_loading(layer)
    drifted = _drifted(before, tensor_slots(layer, kernel))
    assert drifted and all("closure" in p for p in drifted), drifted


def test_canary_arena_backed_kernel_is_stable():
    layer, kernel = torch.nn.Module(), _ArenaKernel()
    kernel.process_weights_after_loading(layer)
    before = tensor_slots(layer, kernel)
    kernel.process_weights_after_loading(layer)
    assert _drifted(before, tensor_slots(layer, kernel)) == []
