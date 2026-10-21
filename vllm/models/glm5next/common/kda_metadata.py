# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


class Glm5NextKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> int:
        # Decode kernels read device offsets within the allocated MTP window.
        # Mixed prefill/decode graphs retain GDN's existing restrictions.
        return vllm_config.num_speculative_tokens + 1


class Glm5NextKDAAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KDA"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDAMetadataBuilder]:
        return Glm5NextKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True
