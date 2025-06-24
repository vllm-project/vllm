# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class VerifyAndUpdateConfig:

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        raise NotImplementedError


class Qwen3ForSequenceClassification(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        is_original_qwen3_reranker = getattr(config,
                                             "is_original_qwen3_reranker",
                                             False)

        if not is_original_qwen3_reranker:
            return

        tokens = getattr(config, "classifier_from_token", None)
        assert tokens is not None and len(tokens) == 2, \
            ("Try loading the original Qwen3 Reranker?, see: "
             "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/qwen3_reranker.py")
        config.num_labels = 1
