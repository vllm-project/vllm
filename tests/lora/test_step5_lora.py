# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch.nn import functional as F

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.step5 import (
    Step5FusedQKVIndexerWithLoRA,
    Step5SparseIndexerWWithLoRA,
)
from vllm.lora.utils import from_layer
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.step5.nvidia import model as step5_model
from vllm.models.step5.nvidia.model import (
    STEP5_PACKED_MODULES_MAPPING,
    Step5FusedQKVIndexerLinear,
    Step5SparseIndexerIndexTPLinear,
)

CONFIG = LoRAConfig(max_lora_rank=16, lora_dtype=torch.float32)


def _set_tp(monkeypatch, tp, rank):
    for module in (linear, parameter, step5_model):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)


@pytest.mark.parametrize("index_kv_heads", [1, 4])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_fused_qkv_indexer_lora_b_matches_weight_loader(
    monkeypatch, tp, index_kv_heads
):
    rows = {"q": 32, "k": 16, "v": 16, "index_q": 32}
    rows |= {"index_k": 4 * index_kv_heads, "index_z": 4 * index_kv_heads}
    rows["index_g"] = 8
    full = {shard: torch.randn(n, 16) for shard, n in rows.items()}
    for rank in range(tp):
        _set_tp(monkeypatch, tp, rank)
        base = Step5FusedQKVIndexerLinear(
            16,
            4,
            8,
            4,
            indexer_q_output_size=32,
            indexer_kv_output_size=4 * index_kv_heads,
            gate_output_size=8,
            proxy_dim=4,
            index_tp_size=4,
            params_dtype=torch.float32,
        )
        for shard, weight in full.items():
            base.weight.weight_loader(base.weight, weight, shard)
        layer = from_layer(
            base, 1, CONFIG, STEP5_PACKED_MODULES_MAPPING["qkv_indexer_proj"]
        )
        assert isinstance(layer, Step5FusedQKVIndexerWithLoRA)
        lora_b = torch.cat(layer.slice_lora_b(list(full.values())))
        torch.testing.assert_close(lora_b, base.weight.data, atol=0, rtol=0)
        monkeypatch.setattr(layer, "_apply_lora_to_output", lambda x, out: out)
        x = torch.randn(3, 16)
        torch.testing.assert_close(
            torch.cat(layer(x), dim=-1), F.linear(x, base.weight), atol=0, rtol=0
        )


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_sparse_indexer_w_lora_b_matches_weight_loader(monkeypatch, tp):
    full = torch.randn(16, 32)
    for rank in range(tp):
        _set_tp(monkeypatch, tp, rank)
        base = Step5SparseIndexerIndexTPLinear(
            32, 16, params_dtype=torch.float32, index_tp_size=4
        )
        base.weight.weight_loader(base.weight, full)
        layer = from_layer(base, 1, CONFIG, [])
        assert isinstance(layer, Step5SparseIndexerWWithLoRA)
        torch.testing.assert_close(
            layer.slice_lora_b(full), base.weight.data, atol=0, rtol=0
        )
