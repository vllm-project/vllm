# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EuroBERT model (https://huggingface.co/EuroBERT).

EuroBERT is a multilingual encoder that is architecturally a bidirectional
Llama (RMSNorm, GQA, SwiGLU MLP and RoPE), so it reuses the Llama building
blocks. Bidirectional (encoder-only) attention is enabled by setting
``is_causal=False`` on the config in ``EuroBertModelConfig``
(``vllm/model_executor/models/config.py``).
"""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


@default_pooling_type(seq_pooling_type="MEAN")
class EuroBertModel(as_embedding_model(LlamaForCausalLM)):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # EuroBERT checkpoints ship a masked-LM head (`lm_head`) that the
        # embedding model does not use.
        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])
        return loader.load_weights(weights)
