# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policies for tiering store and promotion submissions."""

from vllm.v1.kv_offload.tiering.admission.always import AlwaysAdmitPolicy
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy
from vllm.v1.kv_offload.tiering.admission.factory import AdmissionPolicyFactory

__all__ = [
    "AdmissionPolicyFactory",
    "AlwaysAdmitPolicy",
    "TieringAdmissionPolicy",
]
