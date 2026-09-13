# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillAdmissionLimits:
    """Per-step limits produced by a prefill admission policy."""

    max_scheduled_running: int
    max_new_prefills: int | None
    defer_running_prefills: bool


class PrefillAdmissionPolicy:
    """Controls bounded prefill waves without evicting resident decodes."""

    def __init__(self, max_num_seqs: int, prefill_admission_slots: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.prefill_admission_slots = min(
            prefill_admission_slots, max_num_seqs
        )

    @property
    def enabled(self) -> bool:
        return self.prefill_admission_slots > 0

    @property
    def max_num_resident_reqs(self) -> int:
        # Deferred decodes retain their KV cache while a prefill wave runs.
        return self.max_num_seqs + self.prefill_admission_slots

    def get_limits(
        self,
        *,
        num_resident_decodes: int,
        num_running_prefills: int,
        num_waiting_prefills: int,
        resident_prefill_capacity: int,
        defer_prefills: bool,
    ) -> PrefillAdmissionLimits:
        if not self.enabled:
            return PrefillAdmissionLimits(self.max_num_seqs, None, False)

        # Preserve a full decode batch so it can keep using decode-optimized
        # execution. Open a bounded prefill wave only after decode residency
        # naturally drops below max_num_seqs.
        decode_batch_is_full = num_resident_decodes >= self.max_num_seqs
        if defer_prefills or decode_batch_is_full:
            num_reserved_slots = 0
        else:
            num_reserved_slots = min(
                self.prefill_admission_slots - num_running_prefills,
                num_waiting_prefills,
                resident_prefill_capacity,
            )
            num_reserved_slots = max(0, num_reserved_slots)

        return PrefillAdmissionLimits(
            max_scheduled_running=self.max_num_seqs - num_reserved_slots,
            max_new_prefills=num_reserved_slots,
            defer_running_prefills=decode_batch_is_full,
        )