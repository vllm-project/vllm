# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policy that admits every transfer."""

from collections.abc import Collection

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy


class AlwaysAdmitPolicy(TieringAdmissionPolicy):
    """Admit all cascades and promotions.

    The manager default: preserves pre-admission-policy behavior exactly.
    """

    def should_admit(
        self,
        keys: Collection[OffloadKey],
        tier_idx: int,
        is_promotion: bool,
    ) -> bool:
        return True
