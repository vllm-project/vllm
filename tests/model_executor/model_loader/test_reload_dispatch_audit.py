# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier 3: dataflow discovery of storage no path can reach.

The reload arena covers storage a layer owns (tier 1). The module-level
manifest covers globals reachable by scanning loaded modules (tier 2,
``vllm/model_executor/reload_manifest.py``). Neither can see a tensor that
is referenced from nowhere addressable -- held only in a closure a walk
does not descend into, a container under an unscanned module, or an object
graph the walk bottoms out on -- yet a captured graph bakes its address
just the same.

``TorchDispatchMode`` finds those by dataflow rather than reachability:
anything that flows through an op gets recorded, regardless of who owns it.
That is strictly more discovery power than a walk, at the cost of
intercepting every op, which is why this lives in the test tree and is
deliberately not importable from the serving path. RFC #48312 draws the
same boundary: production does not use dispatch discovery.

Its job is to produce a worklist. Storage it records that tiers 1 and 2 do
not account for is an undeclared site -- either migrate it to the arena or
waive it with a reason.

The recorder cannot compute ``moved``: its paths are op-and-position
counters, not re-resolvable slots. Liveness (``expired``) is all it
offers, which is why it complements the manifest rather than replacing it.
"""

import sys
import types

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from vllm.model_executor.reload_manifest import collect_module_level_tensors
from vllm.platforms import current_platform


class DispatchRecorder(TorchDispatchMode):
    """Records the storage of every tensor argument flowing through any op.

    Keyed by storage pointer so one tensor seen by many ops is one entry;
    the first op to touch it is kept as a provenance hint for triage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[int, tuple[str, StorageWeakRef]] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        flat, _ = tree_flatten((args, kwargs or {}))
        for value in flat:
            if isinstance(value, torch.Tensor) and value.numel():
                storage = value.untyped_storage()
                self.seen.setdefault(
                    storage.data_ptr(),
                    (str(func), StorageWeakRef(storage)))
        return func(*args, **(kwargs or {}))

    def expired(self) -> list[str]:
        """Recorded storage that has since been freed: a captured address
        pointing at it now dangles."""
        return [op for _, (op, ref) in self.seen.items() if ref.expired()]

    def residual(self, *, accounted: set[int]) -> dict[int, str]:
        """Storage seen flowing through ops that no tier accounts for.

        This is the worklist: each entry is a site that must be migrated to
        the arena or explicitly waived.
        """
        return {ptr: op for ptr, (op, _) in self.seen.items()
                if ptr not in accounted}


PREFIX = "vllm.model_executor.fake_dispatch_holder"


@pytest.fixture
def fake_module():
    module = types.ModuleType(PREFIX)
    sys.modules[PREFIX] = module
    yield module
    sys.modules.pop(PREFIX, None)


def test_records_a_tensor_that_no_walk_can_reach():
    """A tensor held only in a closure over a local: invisible to any
    attribute walk of the model, but its address is in the graph."""
    hidden = torch.randn(4)

    class Toy(torch.nn.Module):
        def forward(self, x):
            return x * hidden

    model = Toy()
    assert not list(model.parameters())
    assert not list(model.buffers())
    assert not any(isinstance(v, torch.Tensor) for v in vars(model).values())

    recorder = DispatchRecorder()
    with recorder:
        model(torch.ones(4))

    assert hidden.untyped_storage().data_ptr() in recorder.seen


def test_expired_fires_when_recorded_storage_is_freed(fake_module):
    fake_module.registry = {"scale": torch.randn(4)}

    def forward():
        return torch.ones(4) * fake_module.registry["scale"]

    recorder = DispatchRecorder()
    with recorder:
        forward()

    fake_module.registry["scale"] = torch.randn(4)  # rebind; old storage dies
    assert recorder.expired()


def test_residual_excludes_what_the_other_tiers_account_for(fake_module):
    """The worklist is what dispatch sees minus what tiers 1 and 2 own --
    otherwise every weight in the model would be reported."""
    fake_module.registry = {"scale": torch.randn(4)}
    x = torch.ones(4)

    recorder = DispatchRecorder()
    with recorder:
        x * fake_module.registry["scale"]

    tier2 = {t.untyped_storage().data_ptr() for t in
             collect_module_level_tensors((PREFIX, ),
                                          require_cuda=False).values()}
    assert tier2, "tier 2 should see the module-level registry"

    # with tier 2 accounted for, the registry tensor drops out of the
    # residual; the un-owned operands remain as the worklist
    residual = recorder.residual(accounted=tier2)
    assert fake_module.registry["scale"].untyped_storage().data_ptr() \
        not in residual
    assert x.untyped_storage().data_ptr() in residual


def test_recorder_does_not_perturb_results():
    """Discovery must not change what it observes."""
    a, b = torch.randn(8), torch.randn(8)
    baseline = a * b + a.sum()
    with DispatchRecorder():
        under_mode = a * b + a.sum()
    assert torch.equal(baseline, under_mode)


# ---------------------------------------------------------------------------
# The open question: does dispatch recording compose with real CUDA graph
# capture? The upstream prototype validates on CPU only and says so. If it
# does not compose, dispatch discovery has to run in a separate warmup
# forward rather than at capture -- which changes how it would be wired.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="needs an accelerator")
def test_dispatch_recording_during_cudagraph_capture():
    static_in = torch.ones(8, device="cuda")
    weight = torch.randn(8, device="cuda")

    # cudagraph capture requires a warmup on a side stream
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            static_in * weight
    torch.cuda.current_stream().wait_stream(side)

    recorder = DispatchRecorder()
    graph = torch.cuda.CUDAGraph()
    captured_under_mode = None
    error = None
    try:
        with recorder:
            with torch.cuda.graph(graph):
                static_out = static_in * weight  # noqa: F841
        captured_under_mode = len(recorder.seen)
    except Exception as e:  # noqa: BLE001 - the finding IS whether it raises
        error = f"{type(e).__name__}: {e}"

    # Whatever the outcome, record it explicitly: a silent zero would be
    # indistinguishable from "composes fine but saw nothing".
    print(f"\ncudagraph capture under dispatch mode: "
          f"error={error!r} storages_seen={captured_under_mode}")

    if error is not None:
        pytest.skip(
            "TorchDispatchMode does not compose with cudagraph capture on "
            f"this build ({error}); dispatch discovery must run in a "
            "separate warmup forward instead of at capture")

    assert captured_under_mode, (
        "dispatch mode composed with capture but recorded nothing, so it "
        "cannot be used to discover graph-visible storage at capture time")
    assert weight.untyped_storage().data_ptr() in recorder.seen


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="needs an accelerator")
def test_warmup_forward_discovery_is_a_viable_fallback():
    """Independent of the capture question: recording a warmup forward on
    the same tensors finds the same storage a capture would bake."""
    static_in = torch.ones(8, device="cuda")
    weight = torch.randn(8, device="cuda")

    recorder = DispatchRecorder()
    with recorder:
        static_in * weight

    assert weight.untyped_storage().data_ptr() in recorder.seen
    assert static_in.untyped_storage().data_ptr() in recorder.seen
