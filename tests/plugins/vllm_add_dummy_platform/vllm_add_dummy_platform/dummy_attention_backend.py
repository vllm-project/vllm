# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.attention.backends.abstract import AttentionBackend


class DummyAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
