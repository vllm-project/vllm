# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect incomplete identities returned by weight-loader receipts."""

import enum
import inspect
from dataclasses import dataclass, field
from typing import TypeAlias

import torch

from vllm.model_executor.load_receipt import (
    LoadCollisionPolicy,
    LoadReceipt,
)

AuditScalar: TypeAlias = str | int | float | bool | None
AuditValue: TypeAlias = AuditScalar | tuple[AuditScalar, ...]

_EXCLUDED_ARGUMENTS = {
    "self",
    "param",
    "loaded_weight",
    "return_success",
}
_MAX_FINDINGS = 20


def _normalize_argument(value: object) -> AuditValue | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return f"{value.__class__.__qualname__}.{value.name}"
    if isinstance(value, tuple):
        normalized: list[AuditScalar] = []
        for item in value:
            normalized_item = _normalize_argument(item)
            if isinstance(normalized_item, tuple):
                return None
            if normalized_item is None and item is not None:
                return None
            normalized.append(normalized_item)
        return tuple(normalized)
    return None


def _stable_arguments(
    args: inspect.BoundArguments,
) -> tuple[tuple[str, AuditValue], ...]:
    values = []
    for name, value in args.arguments.items():
        if name in _EXCLUDED_ARGUMENTS or isinstance(value, torch.Tensor):
            continue
        normalized = _normalize_argument(value)
        if normalized is not None or value is None:
            values.append((name, normalized))
    return tuple(values)


def _loaded_tensor_metadata(
    args: inspect.BoundArguments,
) -> tuple[tuple[int, ...] | None, str | None]:
    loaded_weight = args.arguments.get("loaded_weight")
    if not isinstance(loaded_weight, torch.Tensor):
        return None, None
    return tuple(loaded_weight.shape), str(loaded_weight.dtype)


def _loader_id(loader: object) -> str:
    unwrapped = inspect.unwrap(loader)  # type: ignore[arg-type]
    module = getattr(unwrapped, "__module__", type(unwrapped).__module__)
    qualname = getattr(unwrapped, "__qualname__", type(unwrapped).__qualname__)
    return f"{module}.{qualname}"


def _declared_schema(loader: object) -> tuple[str, ...] | None:
    current = loader
    while current is not None:
        schema = getattr(current, "load_receipt_fragment_arguments", None)
        if schema is not None:
            return tuple(schema)
        current = getattr(current, "__wrapped__", None)
    return None


@dataclass(frozen=True)
class LoadCallWitness:
    """Stable, receipt-independent description of one loader invocation."""

    loader_id: str
    source_key: str | None
    param_name: str
    stable_arguments: tuple[tuple[str, AuditValue], ...]
    loaded_shape: tuple[int, ...] | None
    loaded_dtype: str | None


@dataclass
class LoadEventAudit:
    """Per-layer collision detector for one first-load or reload pass."""

    event_witnesses: dict[str, tuple[LoadCallWitness, bool]] = field(
        default_factory=dict
    )
    target_owners: dict[
        str, tuple[LoadCallWitness, LoadCollisionPolicy]
    ] = field(default_factory=dict)
    loader_schemas: dict[str, tuple[str, ...]] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    _finding_keys: set[tuple[object, ...]] = field(default_factory=set)
    _reported: bool = False

    def observe(
        self,
        *,
        param_name: str,
        receipt: LoadReceipt,
        args: inspect.BoundArguments,
        loader: object,
        source_key: str | None,
    ) -> None:
        fragment = receipt.fragment.format()
        target_key = (
            param_name if not fragment else f"{param_name}[{fragment}]"
        )
        event_key = (
            target_key
            if source_key is None
            else f"{source_key}=>{target_key}"
        )
        loaded_shape, loaded_dtype = _loaded_tensor_metadata(args)
        witness = LoadCallWitness(
            loader_id=_loader_id(loader),
            source_key=source_key,
            param_name=param_name,
            stable_arguments=_stable_arguments(args),
            loaded_shape=loaded_shape,
            loaded_dtype=loaded_dtype,
        )

        previous_event = self.event_witnesses.get(event_key)
        if previous_event is not None:
            previous_witness, previous_consumed = previous_event
            if previous_witness != witness:
                self._report_collision(
                    "EVENT_KEY_COLLISION",
                    event_key,
                    receipt,
                    previous_witness,
                    witness,
                )
            if previous_consumed != receipt.consumed:
                self._report_collision(
                    "STATUS_CONFLICT",
                    event_key,
                    receipt,
                    previous_witness,
                    witness,
                )
        else:
            self.event_witnesses[event_key] = (witness, receipt.consumed)

        if receipt.consumed:
            previous_owner = self.target_owners.get(target_key)
            if (
                previous_owner is not None
                and previous_owner[0] != witness
                and previous_owner[1] is LoadCollisionPolicy.UNIQUE
                and receipt.collision_policy is LoadCollisionPolicy.UNIQUE
            ):
                self._report_collision(
                    "TARGET_ALIAS_COLLISION",
                    target_key,
                    receipt,
                    previous_owner[0],
                    witness,
                )
            else:
                self.target_owners[target_key] = (
                    witness,
                    receipt.collision_policy,
                )

        declared_schema = _declared_schema(loader)
        actual_fields = tuple(name for name, _ in receipt.fragment.items)
        if declared_schema is not None:
            expected_fields = tuple(
                name
                for name in declared_schema
                if args.arguments.get(name) is not None
            )
        else:
            expected_fields = actual_fields
        if declared_schema is not None and actual_fields != expected_fields:
            self._report(
                (
                    "RECEIPT_SCHEMA_MISMATCH",
                    witness.loader_id,
                    expected_fields,
                    actual_fields,
                ),
                f"RECEIPT_SCHEMA_MISMATCH: loader={witness.loader_id}, "
                f"expected_fields={expected_fields!r}, "
                f"actual_fields={actual_fields!r}",
            )
        schema = declared_schema or actual_fields
        previous_schema = self.loader_schemas.get(witness.loader_id)
        if previous_schema is not None and previous_schema != schema:
            self._report(
                (
                    "SCHEMA_DRIFT",
                    witness.loader_id,
                    previous_schema,
                    schema,
                ),
                f"SCHEMA_DRIFT: loader={witness.loader_id}, "
                f"first_schema={previous_schema!r}, schema={schema!r}",
            )
        else:
            self.loader_schemas[witness.loader_id] = schema

    def _report_collision(
        self,
        kind: str,
        key: str,
        receipt: LoadReceipt,
        first: LoadCallWitness,
        second: LoadCallWitness,
    ) -> None:
        first_args = dict(first.stable_arguments)
        second_args = dict(second.stable_arguments)
        differing = tuple(
            sorted(
                name
                for name in first_args.keys() | second_args.keys()
                if first_args.get(name) != second_args.get(name)
            )
        )
        receipt_fields = {name for name, _ in receipt.fragment.items}
        missing_fields = tuple(
            name for name in differing if name not in receipt_fields
        )
        self._report(
            (kind, key, first, second),
            f"{kind}: key={key!r}, loader={second.loader_id}, "
            f"first_source={first.source_key!r}, "
            f"second_source={second.source_key!r}, "
            f"differing_arguments={differing!r}, "
            f"possible_missing_receipt_fields={missing_fields!r}",
        )

    def _report(self, finding_key: tuple[object, ...], message: str) -> None:
        if finding_key in self._finding_keys:
            return
        self._finding_keys.add(finding_key)
        if len(self.findings) < _MAX_FINDINGS:
            self.findings.append(message)

    def take_findings(self) -> tuple[str, ...]:
        """Return findings once, even if a layer is finalized in stages."""
        if self._reported:
            return ()
        self._reported = True
        return tuple(self.findings)


__all__ = [
    "LoadCallWitness",
    "LoadEventAudit",
]
