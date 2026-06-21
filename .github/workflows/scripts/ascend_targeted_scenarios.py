# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "ascend_targeted_scenarios.json"
)


@dataclass(frozen=True)
class TargetedScenarioRegistry:
    scenario_aliases: dict[str, str]
    group_scenarios: dict[str, str]

    @property
    def supported_scenarios(self) -> list[str]:
        return sorted(set(self.scenario_aliases.values()))

    @property
    def supported_groups(self) -> list[str]:
        return sorted(self.group_scenarios)

    def resolve_scenario(self, name: str) -> str | None:
        return self.scenario_aliases.get(name)

    def resolve_group(self, name: str) -> str | None:
        return self.group_scenarios.get(name)


def load_targeted_scenario_registry(
    path: Path = DEFAULT_REGISTRY_PATH,
) -> TargetedScenarioRegistry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    groups = payload.get("groups")
    if not isinstance(scenarios, dict) or not isinstance(groups, dict):
        raise ValueError(
            "targeted scenario registry requires object keys: scenarios, groups"
        )

    scenario_aliases: dict[str, str] = {}
    for scenario_id, config in scenarios.items():
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("targeted scenario id must be a non-empty string")
        if not isinstance(config, dict):
            raise ValueError(f"targeted scenario {scenario_id!r} must be an object")
        aliases = config.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(item, str) for item in aliases
        ):
            raise ValueError(
                f"targeted scenario {scenario_id!r} aliases must be a string array"
            )
        for alias in [scenario_id, *aliases]:
            if not alias:
                raise ValueError(
                    f"targeted scenario {scenario_id!r} has an empty alias"
                )
            existing = scenario_aliases.get(alias)
            if existing and existing != scenario_id:
                raise ValueError(
                    f"targeted scenario alias {alias!r} maps to both "
                    f"{existing!r} and {scenario_id!r}"
                )
            scenario_aliases[alias] = scenario_id

    group_scenarios: dict[str, str] = {}
    for group_id, config in groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("targeted scenario group id must be a non-empty string")
        if not isinstance(config, dict):
            raise ValueError(f"targeted scenario group {group_id!r} must be an object")
        scenario_id = config.get("scenario")
        if (
            not isinstance(scenario_id, str)
            or scenario_id not in scenario_aliases.values()
        ):
            raise ValueError(
                f"targeted scenario group {group_id!r} references unknown "
                f"scenario {scenario_id!r}"
            )
        group_scenarios[group_id] = scenario_id

    return TargetedScenarioRegistry(
        scenario_aliases=scenario_aliases,
        group_scenarios=group_scenarios,
    )
