# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from vllm.config import CompilationMode
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_native_mtp_attention_registers_after_target_attention():
    from vllm.model_executor.models import nemotron_h, nemotron_h_mtp

    static_forward_context = {}

    class StaticContextAttention(_StubModule):
        def __init__(self, *args, prefix: str = "", **kwargs):
            super().__init__()
            static_forward_context[prefix] = self

    config = NemotronHConfig(
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=1,
        hybrid_override_pattern="*",
        mtp_hybrid_override_pattern="*",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_nextn_predict_layers=1,
    )
    model_config = SimpleNamespace(hf_config=config)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=None,
        quant_config=None,
        parallel_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    with (
        patch.object(
            nemotron_h, "get_tensor_model_parallel_world_size", return_value=1
        ),
        patch.object(nemotron_h, "Attention", StaticContextAttention),
        patch.object(nemotron_h, "QKVParallelLinear", _StubModule),
        patch.object(nemotron_h, "RowParallelLinear", _StubModule),
        patch.object(nemotron_h, "RMSNorm", _StubModule),
        patch.object(nemotron_h_mtp, "VocabParallelEmbedding", _StubModule),
        patch.object(nemotron_h_mtp, "ColumnParallelLinear", _StubModule),
        patch.object(nemotron_h_mtp, "ParallelLMHead", _StubModule),
        patch.object(nemotron_h_mtp, "LogitsProcessor", _StubModule),
        patch.object(nemotron_h_mtp, "RMSNorm", _StubModule),
    ):
        nemotron_h.NemotronHAttentionDecoderLayer(
            config=config,
            layer_idx=0,
            model_config=model_config,
            prefix="model.layers.0",
        )
        nemotron_h_mtp.NemotronHMTP(
            vllm_config=vllm_config,
            prefix="draft_model",
        )

    assert list(static_forward_context) == [
        "model.layers.0.mixer.attn",
        "draft_model.mtp.layers.0.mixer.attn",
    ]
