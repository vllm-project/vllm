# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return 16 if backend == "flashkda" else None
