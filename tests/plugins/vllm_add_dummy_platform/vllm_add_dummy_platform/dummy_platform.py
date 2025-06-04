# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.cuda import CudaPlatform


class DummyPlatform(CudaPlatform):
    device_name = "DummyDevice"

    def get_attn_backend_cls(self, backend_name, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        return "vllm_add_dummy_platform.dummy_attention_backend.DummyAttentionBackend"  # noqa E501
