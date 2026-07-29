# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable initial-load baseline for constructing partial update scopes."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from vllm.model_executor.load_receipt import FragmentValue

from .layerwise import get_layerwise_info, get_load_manifest_scope
from .types import LoadEventIdentity, LoadManifestScope


@dataclass(frozen=True)
class WeightUpdateBaselineEvent:
    source_name: str
    target_name: str
    fragment: tuple[tuple[str, FragmentValue], ...]


@dataclass(frozen=True)
class WeightUpdateBaselineGroup:
    module_name: str
    events: tuple[WeightUpdateBaselineEvent, ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(sorted({event.source_name for event in self.events}))


@dataclass(frozen=True)
class WeightUpdateBaselineReport:
    scope: LoadManifestScope
    state: Literal["exact", "provisional", "unavailable"]
    groups: tuple[WeightUpdateBaselineGroup, ...]
    provisional_target_names: tuple[str, ...] = ()
    reason: str | None = None


def _event_sort_key(event: LoadEventIdentity) -> str:
    return event.format()


def get_weight_update_baseline(
    model: torch.nn.Module,
) -> WeightUpdateBaselineReport:
    """Return this worker's immutable first-load completion baseline."""
    groups: list[WeightUpdateBaselineGroup] = []
    provisional_targets: list[str] = []
    has_parameters = any(True for _ in model.parameters())
    has_unaddressable_event = False

    for module_name, module in model.named_modules():
        info = get_layerwise_info(module)
        provisional_targets.extend(
            f"{module_name}.{name}" if module_name else name
            for name in (info.required_target_keys or ())
        )
        events = []
        for event in sorted(info.required_events or (), key=_event_sort_key):
            if event.source_name is None:
                has_unaddressable_event = True
                continue
            target_name = (
                f"{module_name}.{event.target_name}"
                if module_name
                else event.target_name
            )
            events.append(
                WeightUpdateBaselineEvent(
                    source_name=event.source_name,
                    target_name=target_name,
                    fragment=event.fragment.items,
                )
            )
        if events:
            groups.append(
                WeightUpdateBaselineGroup(
                    module_name=module_name,
                    events=tuple(events),
                )
            )

    state: Literal["exact", "provisional", "unavailable"]
    reason: str | None
    if provisional_targets:
        state = "provisional"
        reason = (
            "The model has only a target-side dummy baseline. Run one complete "
            "real base-weight update or enable dummy load probing before "
            "requesting partial checkpoint scopes."
        )
    elif has_unaddressable_event:
        state = "unavailable"
        reason = "The initial-load baseline contains events without source names."
    elif has_parameters and not groups:
        state = "unavailable"
        reason = "No structured initial-load baseline was recorded."
    else:
        state = "exact"
        reason = None

    return WeightUpdateBaselineReport(
        scope=get_load_manifest_scope(),
        state=state,
        groups=tuple(groups),
        provisional_target_names=tuple(sorted(set(provisional_targets))),
        reason=reason,
    )


def aggregate_weight_update_baselines(
    reports: list[WeightUpdateBaselineReport],
) -> dict[str, Any]:
    """Merge rank-local closure constraints into global atomic source groups."""
    non_exact = [report for report in reports if report.state != "exact"]
    all_sources = {
        event.source_name
        for report in reports
        for group in report.groups
        for event in group.events
    }

    parent = {name: name for name in all_sources}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for report in reports:
        for group in report.groups:
            names = group.source_names
            for name in names[1:]:
                union(names[0], name)

    components: dict[str, set[str]] = {}
    for name in all_sources:
        components.setdefault(find(name), set()).add(name)
    atomic_groups = sorted(
        (sorted(names) for names in components.values()),
        key=lambda names: names[0],
    )

    return {
        "ready": not non_exact,
        "reason": (
            None
            if not non_exact
            else "; ".join(
                sorted({report.reason or report.state for report in non_exact})
            )
        ),
        "scope_template": {
            "kind": "base_checkpoint",
            "mode": "partial",
            "source_names": [],
        },
        "source_names": sorted(all_sources),
        # A legal partial scope is a union of these connected components.
        "atomic_source_groups": atomic_groups,
        "atomic_update_scopes": [
            {
                "kind": "base_checkpoint",
                "mode": "partial",
                "source_names": names,
            }
            for names in atomic_groups
        ],
        "workers": [asdict(report) for report in reports],
    }


def get_weight_update_manifest(
    model: torch.nn.Module,
    lora_adapters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return one worker's model-weight and LoRA update manifests."""
    return {
        "model_weights": get_weight_update_baseline(model),
        "lora_adapters": list(lora_adapters or ()),
    }


def aggregate_weight_update_manifests(
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate all worker-local updatable-weight manifests."""
    model_weights = aggregate_weight_update_baselines(
        [manifest["model_weights"] for manifest in manifests]
    )
    adapters: dict[tuple[int, str], dict[str, Any]] = {}
    for worker_index, manifest in enumerate(manifests):
        for local in manifest.get("lora_adapters", ()):
            key = (local["adapter_id"], local["adapter_name"])
            adapter = adapters.setdefault(
                key,
                {
                    "adapter_id": local["adapter_id"],
                    "adapter_name": local["adapter_name"],
                    "rank": local["rank"],
                    "generation": local["generation"],
                    "module_names": set(),
                    "modules": {},
                    "workers": [],
                    "replace_scope_template": local["replace_scope_template"],
                    "patch_scope_template": local["patch_scope_template"],
                    "remove_scope_template": local["remove_scope_template"],
                },
            )
            if adapter["rank"] != local["rank"]:
                raise RuntimeError(
                    "Workers disagree on LoRA rank for adapter "
                    f"{local['adapter_id']} ({local['adapter_name']!r})"
                )
            if adapter["generation"] != local["generation"]:
                raise RuntimeError(
                    "Workers disagree on LoRA generation for adapter "
                    f"{local['adapter_id']} ({local['adapter_name']!r})"
                )
            adapter["module_names"].update(local["module_names"])
            for module in local["modules"]:
                adapter["modules"].setdefault(module["module_name"], []).append(
                    {
                        "worker_index": worker_index,
                        "lora_a": module["lora_a"],
                        "lora_b": module["lora_b"],
                    }
                )
            adapter["workers"].append(
                {
                    "worker_index": worker_index,
                    "module_names": local["module_names"],
                }
            )
    lora_adapters = []
    for adapter in adapters.values():
        adapter["module_names"] = sorted(adapter["module_names"])
        adapter["modules"] = [
            {"module_name": name, "workers": workers}
            for name, workers in sorted(adapter["modules"].items())
        ]
        lora_adapters.append(adapter)
    lora_adapters.sort(
        key=lambda adapter: (adapter["adapter_id"], adapter["adapter_name"])
    )
    return {
        "model_weights": model_weights,
        "lora_adapters": lora_adapters,
    }


__all__ = [
    "WeightUpdateBaselineEvent",
    "WeightUpdateBaselineGroup",
    "WeightUpdateBaselineReport",
    "aggregate_weight_update_baselines",
    "aggregate_weight_update_manifests",
    "get_weight_update_baseline",
    "get_weight_update_manifest",
]
