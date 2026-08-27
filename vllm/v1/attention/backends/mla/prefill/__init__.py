# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
    from vllm.v1.attention.backends.mla.prefill.selector import get_mla_prefill_backend

__all__ = [
    "MLAPrefillBackend",
    "MLAPrefillBackendEnum",
    "get_mla_prefill_backend",
    "register_mla_prefill_backend",
]


def __getattr__(name: str):
    if name == "MLAPrefillBackend":
        from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

        return MLAPrefillBackend
    if name == "get_mla_prefill_backend":
        from vllm.v1.attention.backends.mla.prefill.selector import (
            get_mla_prefill_backend,
        )

        return get_mla_prefill_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
