# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional XPU setup for the upstream MiniCPM5 model."""

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .specialized_decode.config import requested


def configure_xpu_model(model, config: VllmConfig) -> None:
    extra = config.additional_config
    options = extra if isinstance(extra, dict) else {}
    if options.get("minicpm5_replicated_embedding", False):
        replicate_embedding(model.model, config)
    if requested(config) or options.get("minicpm5_dspark_split_kv", False) is not False:
        model.get_model_state_cls = _get_xpu_model_state_cls


def _get_xpu_model_state_cls():
    from .model_state import MiniCPM5XpuModelState

    return MiniCPM5XpuModelState


def replicate_embedding(model, config: VllmConfig) -> None:
    original = model.embed_tokens
    if (
        get_tensor_model_parallel_world_size() != 2
        or model.config.tie_word_embeddings
        or config.lora_config is not None
        or not isinstance(original, VocabParallelEmbedding)
        or original.weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Replicated MiniCPM5 embedding requires TP2, untied FP16/BF16 "
            "embedding weights, and no LoRA"
        )
    with torch.device(original.weight.device):
        model.embed_tokens = VocabParallelEmbedding(
            original.num_embeddings,
            original.embedding_dim,
            params_dtype=original.weight.dtype,
            org_num_embeddings=original.org_vocab_size,
            padding_size=original.padding_size,
            quant_config=model.quant_config,
            disable_tp=True,
        )
