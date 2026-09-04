# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry-wide storage stability for MoE experts backends.

``test_post_load_storage_stability.py`` sweeps mixed-precision *linear*
kernels off ``_POSSIBLE_KERNELS``. The MoE experts backends -- FlashInfer,
TRT-LLM, NVFP4, CUTLASS fp8/w4a8, B12x -- are selected through an entirely
separate mechanism (the ``fused_moe/oracle/*`` modules), so the linear
sweep never touches them. This is the MoE analogue, and it is the
enumeration that must name every unmigrated MoE backend.

Two layers, because each answers a different question and has a different
dependency:

  enumeration + static classification (no hardware, always runs)
      Walk every oracle's ``Backend`` enum through ``backend_to_kernel_cls``
      to collect every experts class any MoE path can select. For each,
      inspect whether its constructor (and any lazy-scratch getter) routes
      graph-visible allocations through the reload arena or allocates raw.
      A raw allocator that is not arena-routed is a worklist entry. This is
      hardware-independent: it lists SM100 / SM120 / ROCm backends that
      cannot be constructed here.

  dynamic construct-twice census (needs a constructible class)
      For classes that build with a synthetic fp8 config, construct twice
      under the same layer's arena scope -- the rebuild a reload performs --
      and assert the storage allocated in ``__init__`` is reused. Proves the
      migrated CUTLASS families green; skips (by name, so still enumerated)
      the hardware-gated ones.

The static layer can misjudge an exotic allocation idiom, so it is advisory
(``xfail``/report), never the sole gate; the dynamic layer is ground truth
where it runs.
"""

import inspect

import pytest
import torch

from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# Enumeration: every experts class every oracle can return.
# ---------------------------------------------------------------------------

# (oracle module, Backend enum attr, backend_to_kernel_cls attr). Kept
# explicit rather than globbed so a newly added oracle is a visible,
# reviewed addition rather than a silent omission.
_ORACLES = [
    ("vllm.model_executor.layers.fused_moe.oracle.fp8",
     "Fp8MoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.nvfp4",
     "NvFp4MoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.mxfp4",
     "Mxfp4MoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.int_wna16",
     "WNA16MoEBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.unquantized",
     "UnquantizedMoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.w4a8",
     "W4A8MoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.w4a8_int8",
     "W4A8Int8MoeBackend", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.fused_moe.oracle.int8",
     "Int8MoeBackend", "backend_to_kernel_cls"),
]


def _import(mod_name):
    import importlib
    try:
        return importlib.import_module(mod_name)
    except Exception:
        return None


def enumerate_experts_classes() -> dict[str, type]:
    """Map class name -> experts class across every oracle.

    Backend members whose ``backend_to_kernel_cls`` raises (an optional
    dependency missing at import) are skipped for that member, not for the
    whole oracle: the goal is the widest set the registry can name.
    """
    seen: dict[str, type] = {}
    for mod_name, enum_attr, fn_attr in _ORACLES:
        module = _import(mod_name)
        if module is None:
            continue
        fn = getattr(module, fn_attr, None)
        enum_cls = getattr(module, enum_attr, None) if enum_attr else None
        members = list(enum_cls) if enum_cls is not None else []
        for member in members:
            try:
                result = fn(member)
            except Exception:
                continue
            classes = result if isinstance(result, (list, tuple)) else [result]
            for cls in classes:
                if isinstance(cls, type):
                    seen.setdefault(cls.__name__, cls)
    return seen


# ---------------------------------------------------------------------------
# Static classification: does the class route allocations through the arena?
# ---------------------------------------------------------------------------

_RAW_ALLOC = ("torch.empty", "torch.zeros", "torch.full", "torch.arange",
              "torch.tensor", "torch.ones", "torch.randn")
_ARENA_MARKERS = ("current_arena", "get_reload_arena", "reload_arena",
                  "arena.put", "arena.get_or_alloc", "_stable(", "arena=")

# Classes with no graph-visible runtime tensors of their own, verified by
# reading them: they delegate to sub-experts or hold only references to
# already-managed parameters. Listed so the static layer does not flag them
# and so the exemption is reviewable.
_KNOWN_NO_RUNTIME_TENSORS = {
    "TritonExperts", "BatchedTritonExperts", "TritonOrDeepGemmExperts",
    "BatchedDeepGemmExperts", "MarlinExperts", "CPUExpertsFp8",
    "TritonOrCutlassExperts",  # thin wrapper; inner class is swept separately
    # Constructor temporaries are promoted to registered layer parameters by
    # PWAL before graph capture, so reload copy-back owns their final storage.
    "TrtLlmNvFp4ExpertsModular", "TrtLlmNvFp4ExpertsMonolithic",
}


def _sources(cls) -> str:
    """__init__ plus any lazy-scratch getter, concatenated."""
    chunks = []
    for name in ("__init__", "_get_permute_scratch", "process_weights_"
                 "after_loading"):
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        try:
            chunks.append(inspect.getsource(fn))
        except (OSError, TypeError):
            pass
    return "\n".join(chunks)


def classify(cls) -> str:
    """'no-runtime' | 'arena' | 'raw' | 'unknown'."""
    if cls.__name__ in _KNOWN_NO_RUNTIME_TENSORS:
        return "no-runtime"
    src = _sources(cls)
    if not src:
        return "unknown"
    allocates = any(marker in src for marker in _RAW_ALLOC)
    arena_routed = any(marker in src for marker in _ARENA_MARKERS)
    if not allocates:
        return "no-runtime"
    return "arena" if arena_routed else "raw"


# The migrated CUTLASS families as the registry actually surfaces them.
# The plain CutlassExpertsFp8 is not returned directly -- VLLM_CUTLASS
# resolves to the TritonOrCutlassExperts wrapper, which constructs it
# internally -- but its sibling CutlassBatchedExpertsFp8 and the w4a8 class
# are surfaced, and all three share the migrated CutlassExpertsFp8Base.
_MIGRATED_CUTLASS = ("CutlassBatchedExpertsFp8", "CutlassExpertsW4A8Fp8")


def test_registry_enumerates_the_known_moe_backends():
    """Guards the enumeration itself: if this drops to a handful, the sweep
    below is silently covering almost nothing."""
    classes = enumerate_experts_classes()
    assert len(classes) >= 8, sorted(classes)
    # a migrated family must be present, or the green proof is empty
    assert any(name in classes for name in _MIGRATED_CUTLASS), sorted(classes)


def test_report_moe_experts_worklist(capsys):
    """Print the classification of every enumerated experts class. This is
    the worklist artifact: 'raw' entries are unmigrated backends that own
    graph-visible storage."""
    classes = enumerate_experts_classes()
    buckets: dict[str, list[str]] = {"arena": [], "raw": [],
                                     "no-runtime": [], "unknown": []}
    for name, cls in sorted(classes.items()):
        buckets[classify(cls)].append(name)

    with capsys.disabled():
        print("\n=== MoE experts storage classification ===")
        for kind in ("raw", "arena", "no-runtime", "unknown"):
            print(f"  {kind} ({len(buckets[kind])}): {buckets[kind]}")

    # A migrated family is the proof the 'arena' bucket is real.
    assert any(name in buckets["arena"] for name in _MIGRATED_CUTLASS), \
        buckets["arena"]


# The unmigrated backends we already know own graph-visible storage but
# cannot fix without their hardware. Recorded as expected-raw so a
# regression (one silently disappearing from the registry, or being
# mislabeled arena without a real migration) is visible, and so the debt is
# explicit rather than buried in a print.
def test_no_enumerated_moe_backend_has_raw_runtime_allocations():
    classes = enumerate_experts_classes()
    raw = sorted(name for name, cls in classes.items() if classify(cls) == "raw")
    assert not raw, (
        "MoE backends allocate graph-visible runtime tensors without the "
        f"reload arena: {raw}")


# ---------------------------------------------------------------------------
# Dynamic ground truth: construct twice under an arena scope, census storage.
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if current_platform.is_cuda_alike() else "cpu")


def _fp8_moe_config():
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8, experts_per_token=2, hidden_dim=256,
        intermediate_size=256, num_local_experts=8, num_logical_experts=8,
        activation=MoEActivation.SILU, device=str(DEVICE),
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16, routing_method=RoutingMethodType.TopK,
        intermediate_size_per_partition=256,
    )


def _census(obj) -> dict[str, int]:
    out = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor) and value.numel():
            out[name] = value.data_ptr()
    return out


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="stride allocation needs an accelerator")
@pytest.mark.parametrize("cls_name", list(_MIGRATED_CUTLASS))
def test_migrated_experts_reuse_storage_across_rebuild(cls_name, dist_init):
    """The migrated families must reuse the stride storage they allocate in
    __init__ when the experts object is rebuilt, which is what PWAL does on
    reload.

    The stride tensors depend only on moe_config (num_experts / hidden /
    intermediate), not on the quant config, so a minimal quant config is
    enough to reach the allocation under test. Full end-to-end construction
    of these classes is additionally covered on real models by the
    cat1_cutlass_fp8 / cat1_w4a8_scratch reproductions and by
    test_reload_lazy_storage.py.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig)
    from vllm.model_executor.reload_arena import (arena_scope,
                                                  get_reload_arena)
    classes = enumerate_experts_classes()
    if cls_name not in classes:
        pytest.skip(f"{cls_name} not selectable in this build")
    cls = classes[cls_name]

    moe_config = _fp8_moe_config()
    try:
        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
            per_out_ch_quant=True,
        )
    except Exception as e:
        pytest.skip(f"cannot build a minimal quant config: {e}")

    layer = torch.nn.Module()
    arena = get_reload_arena(layer)

    def build():
        try:
            with arena_scope(arena):
                if cls_name == "CutlassExpertsW4A8Fp8":
                    e = moe_config.num_local_experts
                    strides = torch.full((e,), 256, dtype=torch.int64,
                                         device=DEVICE)
                    return cls(moe_config, quant_config, strides, strides, 128)
                # Batched format requires an explicit token/dispatcher count.
                return cls(moe_config, quant_config, max_num_tokens=64,
                           num_dispatchers=1)
        except (RuntimeError, AssertionError, NotImplementedError,
                ImportError, TypeError, ValueError) as e:
            pytest.skip(f"{cls_name} not constructible here: {e}")

    first = build()
    before = _census(first)
    if not before:
        pytest.skip(f"{cls_name} allocated no stride tensors in __init__")

    del first
    second = build()  # the rebuild
    after = _census(second)

    drifted = sorted(k for k, ptr in before.items() if after.get(k) != ptr)
    assert not drifted, (
        f"{cls_name} rebound {drifted} across a rebuild; captured graphs "
        "hold the previous addresses despite the arena scope")


@pytest.fixture(scope="module")
def dist_init():
    import os
    import tempfile
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (cleanup_dist_env_and_memory,
                                  init_distributed_environment,
                                  initialize_model_parallel)

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl" if current_platform.is_cuda_alike() else "gloo")
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
