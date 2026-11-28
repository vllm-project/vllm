# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Catalog (C2): auto-discovered BYOG workloads and suites."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tools.byog.interfaces import BenchmarkRunner, EvalRunner
from tools.byog.schemas.execution import (
    SuiteDefinition,
    WorkloadDefinition,
    WorkloadKind,
)

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parent


class CatalogError(ValueError):
    """Raised when catalog YAML discovery, validation, or plugin loading fails."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CatalogError(f"Failed to load catalog file '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogError(f"Catalog file '{path}' must be a YAML mapping.")
    return raw


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix in (".yaml", ".yml")
        ),
        key=lambda p: p.name,
    )


class Catalog:
    """In-memory registry populated by scanning `workloads/` and `suites/`."""

    def __init__(
        self,
        workloads: dict[str, WorkloadDefinition],
        suites: dict[str, SuiteDefinition],
    ):
        self._workloads = dict(workloads)
        self._suites = dict(suites)

    @classmethod
    def discover(cls, catalog_dir: Path | None = None) -> Catalog:
        root = catalog_dir or DEFAULT_CATALOG_DIR
        workloads: dict[str, WorkloadDefinition] = {}
        suites: dict[str, SuiteDefinition] = {}

        for path in _yaml_files(root / "workloads"):
            try:
                defn = WorkloadDefinition.model_validate(_load_yaml(path))
            except ValidationError as exc:
                raise CatalogError(f"Invalid workload '{path}': {exc}") from exc
            if path.stem != defn.name:
                raise CatalogError(
                    f"Workload '{path.name}' stem must match name '{defn.name}'."
                )
            workloads[defn.name] = defn

        for path in _yaml_files(root / "suites"):
            try:
                suite = SuiteDefinition.model_validate(_load_yaml(path))
            except ValidationError as exc:
                raise CatalogError(f"Invalid suite '{path}': {exc}") from exc
            if path.stem != suite.name:
                raise CatalogError(
                    f"Suite '{path.name}' stem must match name '{suite.name}'."
                )
            suites[suite.name] = suite

        return cls(workloads=workloads, suites=suites)

    def get_workload(self, name: str) -> WorkloadDefinition:
        if name not in self._workloads:
            known = ", ".join(sorted(self._workloads)) or "<none>"
            raise CatalogError(f"Unknown workload '{name}'. Known: {known}")
        return self._workloads[name]

    def get_suite(self, name: str) -> SuiteDefinition:
        if name not in self._suites:
            known = ", ".join(sorted(self._suites)) or "<none>"
            raise CatalogError(f"Unknown suite '{name}'. Known: {known}")
        return self._suites[name]

    def list_workloads(self) -> list[WorkloadDefinition]:
        return sorted(self._workloads.values(), key=lambda w: (w.kind.value, w.name))

    def list_suites(self) -> list[SuiteDefinition]:
        return sorted(self._suites.values(), key=lambda s: s.name)


def instantiate_plugin(
    entrypoint: str, expected_kind: WorkloadKind | str | None = None
) -> BenchmarkRunner | EvalRunner:
    """Import and instantiate `module.path:ClassName`, checking its base class."""
    mod_name, cls_name = entrypoint.split(":", 1)
    try:
        instance = getattr(importlib.import_module(mod_name), cls_name)()
    except Exception as exc:
        raise CatalogError(f"Failed to load plugin '{entrypoint}': {exc}") from exc

    kind = (
        expected_kind.value
        if isinstance(expected_kind, WorkloadKind)
        else expected_kind
    )
    if kind == WorkloadKind.BENCHMARK.value and not isinstance(
        instance, BenchmarkRunner
    ):
        raise CatalogError(f"Plugin '{entrypoint}' must subclass BenchmarkRunner.")
    if kind == WorkloadKind.EVAL.value and not isinstance(instance, EvalRunner):
        raise CatalogError(f"Plugin '{entrypoint}' must subclass EvalRunner.")
    if not isinstance(instance, (BenchmarkRunner, EvalRunner)):
        raise CatalogError(
            f"Plugin '{entrypoint}' must subclass BenchmarkRunner or EvalRunner."
        )
    return instance
