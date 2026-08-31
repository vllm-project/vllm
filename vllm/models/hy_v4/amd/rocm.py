# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadataBuilder,
)


class HYV4ROCMAiterMLASparseImpl(ROCMAiterMLASparseImpl):
    """Use the generic sink-capable AITER implementation with HY V4 metadata."""


class HYV4ROCMAiterMLASparseMetadataBuilder(ROCMAiterMLASparseMetadataBuilder):
    supports_draft_decode_metadata_update = True
    use_persistent_mla_metadata = False


class HYV4ROCMAiterMLASparseBackend(ROCMAiterMLASparseBackend):
    @staticmethod
    def get_builder_cls() -> type[HYV4ROCMAiterMLASparseMetadataBuilder]:
        return HYV4ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HYV4ROCMAiterMLASparseImpl]:
        return HYV4ROCMAiterMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
