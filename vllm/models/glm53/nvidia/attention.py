# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA (SM100 TRT-LLM) with fused MTP draft decodes."""

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseTRTLLMBackend,
    FlashInferMLASparseTRTLLMMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices


class Glm53FlashInferMLASparseMetadataBuilder(FlashInferMLASparseTRTLLMMetadataBuilder):
    """The TRT-LLM sparse MLA builder that updates draft decodes in place."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        self.supports_draft_decode_metadata_update = (
            self.dcp_world_size == 1
            and config.parallel_config.prefill_context_parallel_size == 1
            and config.attention_config.hisparse_config is None
        )

    def update_draft_decode_metadata(
        self, metadata: FlashInferMLASparseMetadata
    ) -> None:
        """Recompute the per-token request map of a draft decode in place."""
        num_tokens = metadata.num_decode_tokens
        if num_tokens:
            compute_token_to_req_indices(
                metadata.query_start_loc,
                metadata.req_id_per_token,
                num_tokens,
                num_tokens,
            )


class Glm53FlashInferMLASparseBackend(FlashInferMLASparseTRTLLMBackend):
    """``FLASHINFER_MLA_SPARSE`` with
    :class:`Glm53FlashInferMLASparseMetadataBuilder`.
    """

    @staticmethod
    def get_builder_cls() -> type[Glm53FlashInferMLASparseMetadataBuilder]:
        return Glm53FlashInferMLASparseMetadataBuilder


register_backend(
    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
    f"{__name__}.{Glm53FlashInferMLASparseBackend.__qualname__}",
)
