# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2: Auto-discovered catalog for BYOG workloads and suites."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tools.byog.executor.interfaces import BenchmarkRunner, EvalRunner
from tools.byog.schemas.catalog import (
    SuiteDefinition,
    WorkloadDefinition,
    WorkloadKind,
)
from tools.byog.schemas.spec import BenchmarkSelection, EvalSelection

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parent


class CatalogError(ValueError):
    """Raised when catalog YAML discovery, validation, or plugin loading fails."""


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CatalogError(f"Failed to read catalog file '{path}': {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CatalogError(f"Malformed YAML in catalog file '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogError(
            f"Catalog file '{path}' must contain a YAML mapping, got "
            f"{type(raw).__name__}."
        )
    return raw


def _iter_yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    ]
    return sorted(files, key=lambda p: p.name)


class Catalog:
    """In-memory registry of auto-discovered workloads and suites."""

    def __init__(
        self,
        workloads: dict[str, WorkloadDefinition],
        suites: dict[str, SuiteDefinition],
        catalog_dir: Path | None = None,
    ):
        self._workloads = dict(workloads)
        self._suites = dict(suites)
        self.catalog_dir = catalog_dir or DEFAULT_CATALOG_DIR

    @classmethod
    def discover(cls, catalog_dir: Path | None = None) -> Catalog:
        """Scan `workloads/*.y*ml` and `suites/*.y*ml` without manual registration."""
        root = catalog_dir or DEFAULT_CATALOG_DIR
        workloads: dict[str, WorkloadDefinition] = {}
        suites: dict[str, SuiteDefinition] = {}

        for path in _iter_yaml_files(root / "workloads"):
            raw = _load_yaml_mapping(path)
            try:
                defn = WorkloadDefinition.model_validate(raw)
            except ValidationError as exc:
                raise CatalogError(
                    f"Invalid workload catalog entry '{path}': {exc}"
                ) from exc
            if path.stem != defn.name:
                raise CatalogError(
                    f"Workload file '{path.name}' stem '{path.stem}' does not match "
                    f"declared name '{defn.name}'."
                )
            if defn.name in workloads:
                raise CatalogError(
                    f"Duplicate workload name '{defn.name}' found in '{path}'."
                )
            workloads[defn.name] = defn

        for path in _iter_yaml_files(root / "suites"):
            raw = _load_yaml_mapping(path)
            try:
                suite = SuiteDefinition.model_validate(raw)
            except ValidationError as exc:
                raise CatalogError(
                    f"Invalid suite catalog entry '{path}': {exc}"
                ) from exc
            if path.stem != suite.name:
                raise CatalogError(
                    f"Suite file '{path.name}' stem '{path.stem}' does not match "
                    f"declared name '{suite.name}'."
                )
            # Validate every suite entry against selection schemas upfront.
            for idx, entry in enumerate(suite.benchmarks):
                try:
                    BenchmarkSelection.model_validate(entry)
                except ValidationError as exc:
                    raise CatalogError(
                        f"Invalid benchmark entry #{idx} in suite '{path}': {exc}"
                    ) from exc
            for idx, entry in enumerate(suite.evals):
                try:
                    EvalSelection.model_validate(entry)
                except ValidationError as exc:
                    raise CatalogError(
                        f"Invalid eval entry #{idx} in suite '{path}': {exc}"
                    ) from exc
            if suite.name in suites:
                raise CatalogError(
                    f"Duplicate suite name '{suite.name}' found in '{path}'."
                )
            suites[suite.name] = suite

        return cls(workloads=workloads, suites=suites, catalog_dir=root)

    def get_workload(self, name: str) -> WorkloadDefinition:
        if name not in self._workloads:
            known = ", ".join(sorted(self._workloads)) or "<none>"
            raise CatalogError(
                f"Unknown workload '{name}'. Available workloads: {known}"
            )
        return self._workloads[name]

    def get_suite(self, name: str) -> SuiteDefinition:
        if name not in self._suites:
            known = ", ".join(sorted(self._suites)) or "<none>"
            raise CatalogError(f"Unknown suite '{name}'. Available suites: {known}")
        return self._suites[name]

    def list_workloads(
        self, kind: WorkloadKind | None = None
    ) -> list[WorkloadDefinition]:
        items = list(self._workloads.values())
        if kind is not None:
            items = [w for w in items if w.kind == kind]
        return sorted(items, key=lambda w: (w.kind.value, w.name))

    def list_suites(self) -> list[SuiteDefinition]:
        return sorted(self._suites.values(), key=lambda s: s.name)


def instantiate_plugin(
    entrypoint: str,
    expected_kind: WorkloadKind | str | None = None,
) -> BenchmarkRunner | EvalRunner:
    """Import and instantiate a plugin from `module.path:ClassName`."""
    if ":" not in entrypoint:
        raise CatalogError(
            f"Invalid plugin entrypoint '{entrypoint}': expected 'module:ClassName'."
        )
    mod_name, cls_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(mod_name)
    except Exception as exc:
        raise CatalogError(
            f"Failed to import plugin module '{mod_name}' for '{entrypoint}': {exc}"
        ) from exc
    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise CatalogError(
            f"Plugin class '{cls_name}' not found in '{mod_name}'."
        ) from exc
    try:
        instance = cls()
    except Exception as exc:
        raise CatalogError(
            f"Failed to instantiate plugin '{entrypoint}': {exc}"
        ) from exc

    if expected_kind is not None:
        kind_val = (
            expected_kind.value
            if isinstance(expected_kind, WorkloadKind)
            else str(expected_kind)
        )
        if kind_val == WorkloadKind.BENCHMARK.value and not isinstance(
            instance, BenchmarkRunner
        ):
            raise CatalogError(
                f"Plugin '{entrypoint}' must subclass BenchmarkRunner for "
                "kind='benchmark'."
            )
        if kind_val == WorkloadKind.EVAL.value and not isinstance(instance, EvalRunner):
            raise CatalogError(
                f"Plugin '{entrypoint}' must subclass EvalRunner for kind='eval'."
            )
    elif not isinstance(instance, (BenchmarkRunner, EvalRunner)):
        raise CatalogError(
            f"Plugin '{entrypoint}' must subclass BenchmarkRunner or EvalRunner."
        )
    return instance
