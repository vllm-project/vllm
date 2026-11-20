# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V3.2 checkpoint loading with cross-layer indexer sharing."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, DeepseekV32Config


@pytest.fixture(scope="module")
def shared_indexer_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("deepseek_v32_shared_indexer")
    config = DeepseekV32Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=256,
        n_group=1,
        topk_group=1,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=64,
        max_position_embeddings=1024,
        first_k_dense_replace=1,
    )
    config.architectures = ["DeepseekV32ForCausalLM"]
    config.index_topk_freq = 2
    config.index_skip_topk_offset = 1

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)

    weights = {}
    for name, tensor in model.state_dict().items():
        if ".experts.gate_up_proj" in name:
            prefix = name.removesuffix("gate_up_proj")
            for expert, fused in enumerate(tensor):
                gate, up = fused.chunk(2)
                weights[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                weights[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif ".experts.down_proj" in name:
            prefix = name.removesuffix("down_proj")
            for expert, down in enumerate(tensor):
                weights[f"{prefix}{expert}.down_proj.weight"] = down.contiguous()
        else:
            weights[name] = tensor.contiguous()

    config.save_pretrained(path)
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


def _indexer_state(worker):
    layers = worker.model_runner.model.model.layers
    indexers = [layer.self_attn.indexer for layer in layers]
    return (
        [indexer is not None for indexer in indexers],
        indexers[0].wq_b.weight.detach().cpu(),
    )


def test_loads_checkpoint_weights_for_shared_indexer_layers(
    vllm_runner, shared_indexer_checkpoint
):
    """Drop checkpoint indexer weights for layers that reuse an earlier top-k."""
    weights = load_file(Path(shared_indexer_checkpoint) / "model.safetensors")
    expected = weights["model.layers.0.self_attn.indexer.wq_b.weight"]
    assert "model.layers.1.self_attn.indexer.wq_b.weight" in weights

    with vllm_runner(
        shared_indexer_checkpoint,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
    ) as runner:
        states = runner.llm.collective_rpc(_indexer_state)

    assert len(states) == 1
    has_indexer, loaded = states[0]
    assert has_indexer == [True, False]
    torch.testing.assert_close(loaded, expected, rtol=0, atol=0)
