# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from transformers import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces_base import VllmModelForPooling
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)

ROBOMETER_SPECIAL_TOKENS = (
    "<|split_token|>",
    "<|reward_token|>",
    "<|pref_token|>",
    "<|sim_token|>",
    "<|prog_token|>",
)


def ensure_robometer_special_tokens(tokenizer: object) -> None:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    if not callable(get_vocab) or not callable(add_special_tokens):
        return

    vocab = get_vocab()
    missing_tokens = [token for token in ROBOMETER_SPECIAL_TOKENS if token not in vocab]
    if missing_tokens:
        add_special_tokens({"additional_special_tokens": missing_tokens})


class RFMProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        model_config = self.ctx.model_config
        processor_name = model_config.tokenizer or "Qwen/Qwen3-VL-4B-Instruct"
        tokenizer = self.ctx.tokenizer
        if tokenizer is not None:
            ensure_robometer_special_tokens(tokenizer)

        processor = Qwen3VLProcessor.from_pretrained(
            processor_name,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer=tokenizer,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        ensure_robometer_special_tokens(processor.tokenizer)
        return processor


class RobometerHead(nn.Module):
    """Robometer progress, success, preference, and similarity heads."""

    def __init__(
        self,
        progress_head: nn.Module,
        success_head: nn.Module,
        preference_head: nn.Module,
        similarity_head: nn.Module,
    ) -> None:
        super().__init__()
        self.progress_head = progress_head
        self.success_head = success_head
        self.preference_head = preference_head
        self.similarity_head = similarity_head

    @property
    def progress_dim(self) -> int:
        progress_out = self.progress_head[-1]
        assert isinstance(progress_out, nn.Linear)
        return progress_out.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.progress_head(hidden_states),
                self.success_head(hidden_states),
                self.preference_head(hidden_states),
                self.similarity_head(hidden_states),
            ),
            dim=-1,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=RFMProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class RFM(Qwen3VLForConditionalGeneration, VllmModelForPooling):
    """Robometer reward model based on Qwen3-VL.

    The pooling output is a token-classification tensor whose last dimension is
    `[progress_logits, success_logit, preference_logit, similarity_logit]`.
    Robometer-4B uses 10 discrete progress bins, so its output dimension is 13.
    """

    is_pooling_model = True
    default_tok_pooling_type = "STEP"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        hf_config = vllm_config.model_config.hf_config
        hidden_size = hf_config.text_config.hidden_size
        progress_bins = getattr(
            hf_config,
            "progress_discrete_bins",
            getattr(hf_config, "num_progress_bins", 10),
        )

        head_size = hidden_size // 2
        params_dtype = vllm_config.model_config.head_dtype
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size, head_size, dtype=params_dtype),
            nn.LayerNorm(head_size, dtype=params_dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_size, progress_bins, dtype=params_dtype),
        )
        self.success_head = nn.Sequential(
            nn.Linear(hidden_size, head_size, dtype=params_dtype),
            nn.LayerNorm(head_size, dtype=params_dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_size, 1, dtype=params_dtype),
        )
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_size, head_size, dtype=params_dtype),
            nn.LayerNorm(head_size, dtype=params_dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_size, 1, dtype=params_dtype),
        )
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_size, head_size, dtype=params_dtype),
            nn.LayerNorm(head_size, dtype=params_dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_size, 1, dtype=params_dtype),
        )
        self.robometer_head = RobometerHead(
            self.progress_head,
            self.success_head,
            self.preference_head,
            self.similarity_head,
        )
        self.frame_pool_attn = nn.Linear(
            hidden_size,
            1,
            bias=False,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_classify(
            pooler_config, classifier=self.robometer_head
        )
