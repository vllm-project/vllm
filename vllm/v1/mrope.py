# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MROPE_POSITION_STATE_KEY = "mrope_position_state"
_MROPE_POSITION_STATE_VERSION = 1


@dataclass(frozen=True)
class MRoPEPositionState:
    positions: list[list[int]]
    delta: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": _MROPE_POSITION_STATE_VERSION,
            "positions": self.positions,
            "delta": self.delta,
        }

    @classmethod
    def from_kv_transfer_params(
        cls,
        kv_transfer_params: Mapping[str, Any] | None,
        prompt_len: int,
    ) -> "MRoPEPositionState | None":
        if not kv_transfer_params:
            return None
        payload = kv_transfer_params.get(MROPE_POSITION_STATE_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError(f"{MROPE_POSITION_STATE_KEY} must be an object")
        version = payload.get("version")
        if type(version) is not int or version != _MROPE_POSITION_STATE_VERSION:
            raise ValueError(f"Unsupported {MROPE_POSITION_STATE_KEY} version")

        positions = payload.get("positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError(
                f"{MROPE_POSITION_STATE_KEY}.positions must be a non-empty list"
            )
        if any(
            not isinstance(row, list)
            or len(row) != prompt_len
            or any(type(position) is not int for position in row)
            for row in positions
        ):
            raise ValueError(
                f"{MROPE_POSITION_STATE_KEY}.positions must contain integer rows "
                f"of prompt length {prompt_len}"
            )

        delta = payload.get("delta")
        if type(delta) is not int:
            raise ValueError(f"{MROPE_POSITION_STATE_KEY}.delta must be an integer")
        return cls(positions=positions, delta=delta)
