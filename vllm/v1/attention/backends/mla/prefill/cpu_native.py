# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert MLA prefill backend for the CPU platform.

CPU's MLA attention impl (``CPUMLAImpl``) fully overrides ``forward_mha`` with a
kernel that attends directly against the paged latent KV cache (covering both
fresh prefill and cached-prefix continuation in one pass), so it never calls
into a pluggable ``MLAPrefillBackend``'s ``run_prefill_new_tokens``/
``run_prefill_context_chunk``. This class exists only to satisfy
``MLACommonMetadataBuilder``'s structural requirement that
``MLAAttention.prefill_backend`` be a real, cloneable object.
"""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


class CPUNativeMLAPrefillBackend(MLAPrefillBackend):
    """Placeholder MLA prefill backend for CPU; never actually invoked."""

    @staticmethod
    def get_name() -> str:
        return "CPU_NATIVE"

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError(
            "CPUNativeMLAPrefillBackend.run_prefill_new_tokens is unreachable: "
            "CPUMLAImpl.forward_mha fully overrides the dense-MHA prefill path "
            "and never calls the pluggable prefill backend."
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError(
            "CPUNativeMLAPrefillBackend.run_prefill_context_chunk is "
            "unreachable: CPUMLAImpl.forward_mha fully overrides the "
            "dense-MHA prefill path and never calls the pluggable prefill "
            "backend."
        )
