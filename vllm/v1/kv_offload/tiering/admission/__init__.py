# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policies for tiering store/promotion submissions.

An admission policy is the single submission gate for cascade stores
(primary -> secondary) and promotions (secondary -> primary). It wraps the
per-tier backpressure detectors (see
vllm/v1/kv_offload/tiering/backpressure.py) behind one interface so future
policies (e.g. global load/store contention, multi-tier pin budgets) can be
added without touching the manager.
"""

from vllm.v1.kv_offload.tiering.admission.always import AlwaysAdmitPolicy
from vllm.v1.kv_offload.tiering.admission.backpressure import (
    BackpressureAdmissionPolicy,
)
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy
__all__ = [
    "AlwaysAdmitPolicy",
    "BackpressureAdmissionPolicy",
    "TieringAdmissionPolicy",
]
