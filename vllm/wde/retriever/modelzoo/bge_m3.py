# Adapted from
# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/modeling.py
# FlagEmbedding is licensed under the MIT License.

from typing import Optional
import torch
from torch import nn
from vllm.wde.encode_only.modelzoo.xlm_roberta import (XLMRobertaModel,
                                                       XLMRobertaConfig,
                                                       LoadWeightsMixin)
from vllm.wde.encode_only.layers.attention import (EncodeOnlyAttentionMetadata,
                                                   EncodeOnlyAttentionBackend)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class BGEM3Model(nn.Module, LoadWeightsMixin):
    _ignore_weights_keys = [
        "roberta.pooler.dense.weight", "roberta.pooler.dense.bias"
    ]

    prefix = "roberta."

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: EncodeOnlyAttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None,
                 sentence_pooling_method="cls",
                 normalized=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.sentence_pooling_method = sentence_pooling_method
        assert self.sentence_pooling_method == 'cls'
        self.normalized = normalized
        self.roberta = XLMRobertaModel(config, attn_backend, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: EncodeOnlyAttentionMetadata,
    ) -> torch.Tensor:

        sequence_output = self.roberta(
            input_ids,
            positions,
            attn_metadata,
        )

        seq_start_loc = attn_metadata.seq_start_loc

        dense_vecs = sequence_output[seq_start_loc[:-1]]

        if self.normalized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)

        return dense_vecs
