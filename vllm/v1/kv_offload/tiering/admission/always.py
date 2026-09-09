# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult


class AlwaysAdmitPolicy(TieringAdmissionPolicy):
    """Admits every job unconditionally."""

    def should_admit(self, job: JobMetadata) -> bool:
        return True

    def on_admitted(self, job: JobMetadata) -> None:
        return

    def on_completed(self, job: JobMetadata, result: JobResult) -> None:
        return

    def reset(self) -> None:
        return
