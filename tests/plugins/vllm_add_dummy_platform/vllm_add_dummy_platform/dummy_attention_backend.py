# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.attention.backends.flash_attn import FlashAttentionBackend


class DummyAttentionBackend(FlashAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
