# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common PP field names; models may also register custom string names."""

from enum import StrEnum

AUX_HIDDEN_STATES_PREFIX = "aux_hidden_states_"


class PPTransportField(StrEnum):
    HIDDEN_STATES = "hidden_states"
    RESIDUAL = "residual"
    POSITIONS = "pp_positions"

    @staticmethod
    def aux_hidden_state(index: int, *, prefix: str = AUX_HIDDEN_STATES_PREFIX) -> str:
        """Return the field name of an indexed auxiliary hidden state."""
        return f"{prefix}{index}"
