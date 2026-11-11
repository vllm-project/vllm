# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB placement snapshots and per-engine weight-cache writer bindings."""

import secrets
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.model_executor.models.interfaces import get_mixture_of_experts_model

WEIGHT_CACHE_EPLB_BINDING_ATTR = "_vllm_weight_cache_eplb_binding"


@dataclass(frozen=True)
class EplbPlacement:
    """A complete, immutable global physical-to-logical expert placement."""

    version: int
    physical_to_logical_map: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("EPLB placement version must be a non-negative integer")
        if not isinstance(self.physical_to_logical_map, tuple):
            raise ValueError("EPLB placement map must be a tuple of tuples")
        width: int | None = None
        for row in self.physical_to_logical_map:
            if not isinstance(row, tuple) or not row:
                raise ValueError("EPLB placement layers must be non-empty tuples")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("EPLB placement map must be rectangular")
            if any(type(expert) is not int or expert < 0 for expert in row):
                raise ValueError(
                    "EPLB placement logical expert IDs must be non-negative integers"
                )

    @classmethod
    def initial(cls, model: Any) -> "EplbPlacement":
        """Build version zero exactly as the runtime's initial EPLB state does."""
        moe_model = get_mixture_of_experts_model(model)
        if moe_model is None:
            return cls(version=0, physical_to_logical_map=())
        from vllm.distributed.eplb.eplb_state import EplbState

        row = tuple(
            EplbState.build_initial_global_physical_to_logical_map(
                moe_model.num_routed_experts,
                moe_model.num_redundant_experts,
            )
        )
        return cls(
            version=0,
            physical_to_logical_map=tuple(row for _ in range(moe_model.num_moe_layers)),
        )

    @classmethod
    def from_json_dict(cls, value: Any) -> "EplbPlacement":
        """Validate and decode the JSON representation used on the D2D plane."""
        if not isinstance(value, dict) or set(value) != {
            "version",
            "physical_to_logical_map",
        }:
            raise ValueError(
                "EPLB placement must contain exactly version and "
                "physical_to_logical_map"
            )
        rows = value["physical_to_logical_map"]
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("EPLB placement map must be a JSON array of arrays")
        try:
            return cls(
                version=value["version"],
                physical_to_logical_map=tuple(tuple(row) for row in rows),
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid EPLB placement: {e}") from e

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "physical_to_logical_map": [
                list(row) for row in self.physical_to_logical_map
            ],
        }

    def validate_successor(self, new_placement: "EplbPlacement") -> None:
        """Check the version, shape, and logical-ID domain of a committed map."""
        if not isinstance(new_placement, EplbPlacement):
            raise ValueError("EPLB commit requires an EplbPlacement")
        if new_placement.version != self.version + 1:
            raise ValueError(
                f"EPLB commit version must be {self.version + 1}, "
                f"got {new_placement.version}"
            )
        old_rows = self.physical_to_logical_map
        new_rows = new_placement.physical_to_logical_map
        if len(new_rows) != len(old_rows) or any(
            len(new_row) != len(old_row)
            for new_row, old_row in zip(new_rows, old_rows, strict=True)
        ):
            raise ValueError("EPLB commit must preserve the placement map shape")
        if not old_rows:
            raise ValueError("Cannot commit an EPLB placement without MoE layers")
        logical_experts = set(range(max(max(row) for row in old_rows) + 1))
        if any(set(row) != logical_experts for row in new_rows):
            raise ValueError(
                "EPLB commit must retain every logical expert in every layer"
            )


class WeightCacheEplbBinding:
    """Per-rank model binding for the single synchronous EPLB writer."""

    def __init__(
        self,
        placement: EplbPlacement,
        *,
        token: str | None = None,
        held_socket: socket.socket | None = None,
        request: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self._placement = placement
        self._token = token
        self._held_socket = held_socket
        self._request = request

    @property
    def version(self) -> int:
        return self._placement.version

    @property
    def placement(self) -> EplbPlacement:
        return self._placement

    def _command(self, cmd: str, **fields: Any) -> dict[str, Any]:
        if self._request is None or self._token is None:
            raise RuntimeError("This model has no active weight-cache EPLB writer")
        response = self._request({"cmd": cmd, "token": self._token, **fields})
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Weight cache daemon rejected EPLB {cmd}: "
                f"{response.get('message', response)}"
            )
        return response

    def begin(self, version: int) -> None:
        """Mark weights dirty before starting any expert movement."""
        self._command("eplb_begin", version=version)

    def commit(self, placement: EplbPlacement) -> None:
        """Publish a fully moved next-version placement and mark weights clean."""
        self._command("eplb_commit", placement=placement)
        self._placement = placement

    def abort(self) -> None:
        """Fail closed after an incomplete movement; old weights may be mixed."""
        self._command("eplb_abort")

    def freeze(self) -> None:
        """Prevent writer mutations while D2D copy leases are active."""
        self._command("eplb_freeze")

    def unfreeze(self) -> None:
        """Resume writing after all D2D copy leases have been acknowledged."""
        self._command("eplb_unfreeze")

    def can_unfreeze(self) -> bool:
        """Whether the daemon currently has no active D2D copy lease."""
        response = self._command("eplb_can_unfreeze")
        return bool(response["can_unfreeze"])

    def detach(self) -> None:
        """Cleanly release the writer so a replacement engine can hot-restart."""
        try:
            if self._request is not None and self._token is not None:
                self._command("eplb_detach")
        finally:
            if self._held_socket is not None:
                self._held_socket.close()
                self._held_socket = None
            self._token = None
            self._request = None


def bind_weight_cache_eplb(
    model: Any,
    placement: EplbPlacement,
    *,
    token: str | None = None,
    held_socket: socket.socket | None = None,
    request: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> WeightCacheEplbBinding:
    """Attach a placement and optional writer session directly to a model."""
    binding = WeightCacheEplbBinding(
        placement, token=token, held_socket=held_socket, request=request
    )
    setattr(model, WEIGHT_CACHE_EPLB_BINDING_ATTR, binding)
    return binding


def get_weight_cache_eplb_binding(model: Any) -> WeightCacheEplbBinding | None:
    """Return the weight-cache binding without treating it as an exported flag."""
    binding = getattr(model, WEIGHT_CACHE_EPLB_BINDING_ATTR, None)
    if binding is not None and not isinstance(binding, WeightCacheEplbBinding):
        raise TypeError(f"{WEIGHT_CACHE_EPLB_BINDING_ATTR} has an invalid value")
    return binding


def new_writer_token() -> str:
    return secrets.token_urlsafe(32)
