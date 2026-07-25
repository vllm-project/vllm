# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from inspect import BoundArguments
import torch

from .audit import LoadEventAudit

__all__ = [
    "LayerTensors",
    "LayerRestoreMetadata",
    "LayerReloadingInfo",
    "LoadManifestGroupScope",
    "LoadManifestScope",
    "LoadManifestReport",
]

# Runtime tensors contain only materialized Tensor values. Restore metadata
# additionally preserves registered ``None`` slots such as ``Linear.bias``
# when a layer was constructed with ``bias=False``.
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
LayerRestoreMetadata = tuple[
    dict[str, torch.Tensor | None],
    dict[str, torch.Tensor | None],
]


@dataclass(frozen=True)
class LoadManifestGroupScope:
    """The current worker's coordinate in one parallel group."""

    rank: int
    world_size: int


@dataclass(frozen=True)
class LoadManifestScope:
    """Rank coordinates attached to a process-local load manifest.

    A manifest is intentionally not unioned across workers: TP, PP and EP
    workers consume different target fragments, while DP replicas own
    independent model state. The controller may collect these coordinates and
    reports, but reconciliation must happen in each worker first.
    """

    global_rank: int | None = None
    global_world_size: int | None = None
    tp: LoadManifestGroupScope | None = None
    pp: LoadManifestGroupScope | None = None
    dp: LoadManifestGroupScope | None = None
    ep: LoadManifestGroupScope | None = None


@dataclass(frozen=True)
class LoadManifestReport:
    """Serializable result of one worker's local reconciliation."""

    scope: LoadManifestScope
    required_event_count: int
    received_event_count: int
    completion_findings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.completion_findings


@dataclass
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerRestoreMetadata

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # Whether this layer currently has online reload wrappers installed.
    # Completion is manifest-driven; copied element counts are diagnostics only.
    is_loading: bool = False
    copied_numel_diagnostic: int = 0
    layer_numel_diagnostic: int | None = None

    # used by `online_process_loader` to buffer args and tensors until ready to load
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # The set of parameter keys a correct load actually consumes, OBSERVED at
    # the first load (record_load_consumption), not predicted from layer
    # structure. This is the ground-truth "required" for completion: tensors
    # never routed through a loader (EP bookkeeping, shared copies) are absent
    # by construction, so no SKIP_TENSORS-style hand-maintained exclusion is
    # needed. It is a BASELINE -- survives reset(), like restore_metadata.
    required_keys: "set[str] | None" = None

    # Keys received during the current reload, accumulated by the online
    # loader wrapper. Transient: cleared by reset() each reload. Completion
    # by set reconciliation is received_keys >= required_keys.
    received_keys: set[str] = field(default_factory=set)

    # Dummy loading has no checkpoint source stream, but it still has to
    # protect the first real RL weight transfer. This rank-local target
    # baseline lists every parameter that must be touched at least once.
    # After the first complete real load it is replaced by the exact observed
    # source-to-target event set above.
    required_target_keys: "set[str] | None" = None
    received_target_keys: set[str] = field(default_factory=set)

    # Independent witnesses used to detect multiple semantic loader calls
    # collapsing to the same LoadReceipt event/target identity.
    load_event_audit: LoadEventAudit = field(default_factory=LoadEventAudit)

    def reset(self, *, preserve_received_keys: bool = False):
        # required_keys is the observed baseline and must survive across
        # reloads, like restore_metadata. A layer can finish its numel-driven
        # processing before the checkpoint iterator is exhausted, so receipts
        # optionally survive that *internal* reset and are cleared only when
        # the whole reload is finalized.
        received_keys = self.received_keys if preserve_received_keys else set()
        received_target_keys = (
            self.received_target_keys if preserve_received_keys else set()
        )
        load_event_audit = (
            self.load_event_audit
            if preserve_received_keys
            else LoadEventAudit()
        )
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            required_keys=self.required_keys,
            received_keys=received_keys,
            required_target_keys=self.required_target_keys,
            received_target_keys=received_target_keys,
            load_event_audit=load_event_audit,
        )

    def can_load(self) -> bool:
        return self.is_loading
