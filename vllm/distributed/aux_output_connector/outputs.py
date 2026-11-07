# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level auxiliary outputs carried from the scheduler to clients."""

from __future__ import annotations

from dataclasses import dataclass, fields

from vllm.distributed.aux_output_connector.hidden_states import (
    HiddenStateCaptureResult,
)


@dataclass
class AuxOutputs:
    """Settled auxiliary outputs for one request.

    The engine layer only stores, merges and forwards this container; new
    auxiliary outputs add fields here instead of to engine output types.
    """

    hidden_state_capture: HiddenStateCaptureResult | None = None
    hidden_capture_skip_reason: str | None = None

    def combine(self, older: AuxOutputs | None) -> AuxOutputs:
        """Return this output with fields it does not set taken from older."""
        if older is None:
            return self
        return AuxOutputs(
            **{
                f.name: value
                if (value := getattr(self, f.name)) is not None
                else getattr(older, f.name)
                for f in fields(self)
            }
        )
