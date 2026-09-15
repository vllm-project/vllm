# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import BoundedContextCudaGraph


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.cpu_test
def test_bounded_context_cudagraph_replay_pads_inert_slots():
    manager = BoundedContextCudaGraph(
        torch.device("cpu"), torch.float32, hidden_size=2, max_num_tokens=2
    )

    class FakeGraph:
        replay_count = 0

        def replay(self):
            self.replay_count += 1

    graph = FakeGraph()
    manager.graphs[2] = graph  # type: ignore[assignment]
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([5, 6])
    slots = torch.tensor([9, PAD_SLOT_ID])

    assert manager.replay(states, positions, [slots, slots], eligible=True)
    assert graph.replay_count == 1
    assert manager.replay_count == 1
    assert manager.fallback_count == 0
    torch.testing.assert_close(manager.context_states[:2], states)
    torch.testing.assert_close(manager.context_positions[:2], positions)
    assert manager.context_slot_mapping.tolist() == [9, PAD_SLOT_ID]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("eligible", "num_tokens", "slot_mapping"),
    [
        (False, 2, torch.tensor([1, 2])),
        (True, 5, torch.tensor([1, 2, 3, 4, 5])),
        (True, 2, [torch.tensor([1, 2]), torch.tensor([1, 2])]),
        (True, 2, None),
    ],
)
def test_bounded_context_cudagraph_falls_back(eligible, num_tokens, slot_mapping):
    manager = BoundedContextCudaGraph(
        torch.device("cpu"), torch.float32, hidden_size=2, max_num_tokens=4
    )
    graph = SimpleNamespace(
        replay=lambda: pytest.fail("ineligible input must not replay")
    )
    manager.graphs = {  # type: ignore[assignment]
        num_tokens: graph for num_tokens in range(1, manager.max_num_tokens + 1)
    }

    assert not manager.replay(
        torch.zeros(num_tokens, 2),
        torch.arange(num_tokens),
        slot_mapping,
        eligible=eligible,
    )
    assert manager.replay_count == 0
    assert manager.fallback_count == 1


@pytest.mark.cpu_test
def test_bounded_context_cudagraph_clear_drops_pool_ownership():
    manager = BoundedContextCudaGraph(
        torch.device("cpu"), torch.float32, hidden_size=2, max_num_tokens=1
    )
    manager.graphs[1] = SimpleNamespace(replay=lambda: None)  # type: ignore[assignment]
    manager.graph_pools[1] = (1, 2)
    manager.replay_count = 3
    manager.fallback_count = 4

    manager.clear()

    assert manager.graphs == {}
    assert manager.graph_pools == {}
    assert manager.replay_count == 0
    assert manager.fallback_count == 0


@pytest.mark.cpu_test
def test_context_cache_pointer_array_tracks_reallocation():
    owner = SimpleNamespace()
    layers = [
        SimpleNamespace(kv_cache=torch.empty(2, 3, 4)),
        SimpleNamespace(kv_cache=torch.empty(2, 3, 4)),
    ]
    first = dspark_mla.K3DSparkModel._get_context_kv_cache_ptrs(owner, layers)
    first_values = first.tolist()

    layers[0].kv_cache = torch.empty(4, 3, 4)
    second = dspark_mla.K3DSparkModel._get_context_kv_cache_ptrs(owner, layers)

    assert second is not first
    assert second.tolist() == [layer.kv_cache.data_ptr() for layer in layers]
    assert second.tolist() != first_values


@pytest.mark.cpu_test
def test_context_cache_layout_rechecks_after_reallocation():
    owner = SimpleNamespace()
    layers = [
        SimpleNamespace(kv_cache=torch.empty(2, 3, 4)),
        SimpleNamespace(kv_cache=torch.empty(2, 3, 4)),
    ]
    assert dspark_mla.K3DSparkModel._has_uniform_block_layout(owner, layers)

    layers[1].kv_cache = torch.empty(2, 5, 4)
    assert not dspark_mla.K3DSparkModel._has_uniform_block_layout(owner, layers)


@pytest.mark.cpu_test
def test_context_precompute_split_reuses_eager_projection():
    states = torch.randn(3, 4)
    projected = torch.randn(3, 6)
    positions = torch.arange(3)
    slots = torch.arange(3)
    calls: list[tuple[torch.Tensor, ...]] = []
    owner = SimpleNamespace(
        project_context_kv=lambda actual: calls.append((actual,)) or projected,
        store_projected_context_kv=lambda *args: calls.append(args),
    )

    dspark_mla.K3DSparkModel._precompute_fused_context_kv(
        owner, states, positions, slots
    )

    assert calls[0][0] is states
    assert calls[1][0] is projected
    assert calls[1][1] is positions
    assert calls[1][2] is slots


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.float8_e8m0fnu, torch.float32]
)
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_module", "shard_id"),
    [
        ("mtp.0.attn.wq_a.scale", "model.layers.0.attn.fused_wqa_wkv", 0),
        ("mtp.0.attn.wkv.scale", "model.layers.0.attn.fused_wqa_wkv", 1),
        ("mtp.0.main_proj.scale", "model.main_proj", None),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            "model.layers.0.ffn.shared_experts.gate_up_proj",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            "model.layers.0.ffn.shared_experts.down_proj",
            None,
        ),
    ],
)
def test_v41_dspark_loads_linear_scales(
    monkeypatch, scale_dtype, checkpoint_name, runtime_module, shard_id
):
    """Checkpoint ``.scale`` maps to the quant method's scale parameter and
    loads untouched. MXFP8 block-scale expansion lives in the KMxfp8Static
    loader (see tests/quantization/test_modelopt.py), not in load_weights."""
    from vllm.models.deepseek_v41.nvidia import dspark

    mxfp8 = scale_dtype != torch.float32
    scale_name = "weight_scale" if mxfp8 else "weight_scale_inv"
    runtime_name = f"{runtime_module}.{scale_name}"
    raw = torch.tensor([[120, 127], [128, 130]], dtype=torch.uint8)
    checkpoint_scale = raw.view(scale_dtype) if mxfp8 else raw.float()
    param = nn.Parameter(torch.empty_like(checkpoint_scale), requires_grad=False)
    shards = []

    def load_scale(param, weight, *args):
        shards.append(args)
        assert weight.dtype == checkpoint_scale.dtype
        param.copy_(weight)

    param.weight_loader = load_scale
    draft = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, n_routed_experts=1),
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32] if mxfp8 else [128, 128]
        ),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))],
            confidence_head=None,
        ),
        named_parameters=lambda: [(runtime_name, param)],
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )

    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(
        draft, [(checkpoint_name, checkpoint_scale)]
    )

    assert loaded == {runtime_name}
    assert shards == [() if shard_id is None else (shard_id,)]
    torch.testing.assert_close(param, checkpoint_scale)
