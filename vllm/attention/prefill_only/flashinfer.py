from typing import Type

from vllm.attention.prefill_only.flash_attn import (
    PrefillOnlyFlashAttentionBackend, PrefillOnlyFlashAttentionImpl)


class PrefillOnlyFlashInferBackend(PrefillOnlyFlashAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def get_impl_cls() -> Type["PrefillOnlyFlashInferImpl"]:
        return PrefillOnlyFlashInferImpl


class PrefillOnlyFlashInferImpl(PrefillOnlyFlashAttentionImpl):
    # Because prefill only models do not involve kv cache,
    # When using Flashinfer backend in prefill only models,
    # you are actually using FLASH ATTN backend
    pass
