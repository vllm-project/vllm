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
from vllm.models.deepseek_v4.nvidia import dspark as dsv4_dspark
from vllm.models.kimi_k3.common import dspark_mla as common_dspark_mla
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel
from vllm.platforms import current_platform


def test_nvidia_dspark_binding_uses_multi_head_latent_attention():
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    assert dspark_mla.MultiHeadLatentAttention is MultiHeadLatentAttention


def test_default_mla_hooks_return_their_inputs():
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    kv = torch.zeros(2, 4)
    k_pe = torch.zeros(2, 1, 2)
    slots = torch.zeros(2, dtype=torch.int64)
    q = torch.zeros(2, 2, 4)
    ql = torch.zeros(2, 2, 4)
    q_pe = torch.zeros(2, 2, 2)
    layer = SimpleNamespace(
        kv_cache_dtype="auto",
        impl=SimpleNamespace(supports_quant_query_input=False),
    )

    out_kv, out_k_pe, out_slots = MLAAttention._prepare_kv_cache_update(
        layer, kv, k_pe, slots, None
    )
    assert out_kv is kv
    assert out_k_pe is k_pe
    assert out_slots is slots

    out_q, out_mha_k_pe = MLAAttention._prepare_mha_inputs(layer, q, k_pe)
    assert out_q is q
    assert out_mha_k_pe is k_pe

    formed = MLAAttention._form_decode_q(layer, ql, q_pe, kv, k_pe, kv, None, 2)
    assert formed[0] is ql
    assert formed[1] is q_pe


def test_kv_cache_layer_defaults_to_the_attention_module():
    attn = SimpleNamespace(layer_name="model.layers.0.self_attn", mla_attn=object())
    assert K3DSparkModel.kv_cache_layer(SimpleNamespace(), attn) is attn

    class NestedCacheOwner(K3DSparkModel):
        def kv_cache_layer(self, attn):
            return attn.mla_attn

    assert NestedCacheOwner.kv_cache_layer(SimpleNamespace(), attn) is attn.mla_attn


def test_wrapper_uses_mla_attn_cls():
    from vllm.model_executor.layers.mla import (
        MLAModules,
        MultiHeadLatentAttentionWrapper,
    )

    class DummyAttn(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.prefix = kwargs["prefix"]

    class Sub(MultiHeadLatentAttentionWrapper):
        mla_attn_cls = DummyAttn

    modules = MLAModules(
        kv_a_layernorm=nn.Identity(),
        kv_b_proj=nn.Identity(),
        rotary_emb=None,
        o_proj=nn.Identity(),
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=nn.Identity(),
        q_a_layernorm=None,
        q_b_proj=None,
        q_proj=nn.Identity(),
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )
    wrapper = Sub(
        hidden_size=8,
        num_heads=2,
        scale=1.0,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=4,
        mla_modules=modules,
        prefix="model.layers.0.self_attn",
    )
    assert isinstance(wrapper.mla_attn, DummyAttn)
    assert wrapper.mla_attn.prefix == "model.layers.0.self_attn.attn"


def test_dspark_mla_uses_compile_free_model_entrypoint():
    from vllm.models.kimi_k3 import K3DSparkForCausalLM as registered

    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is registered
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 DSpark MLA wrapper is the ROCm draft path",
)
def _bind_mla_wrapper_call(args, kwargs):
    import inspect

    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

    signature = inspect.signature(MultiHeadLatentAttentionWrapper.__init__)
    return signature.bind(None, *args, **kwargs).arguments


def test_k3_dspark_decoder_uses_mla_wrapper(monkeypatch: pytest.MonkeyPatch):
    from vllm.models.kimi_k3.amd import dspark_mla as amd_dspark_mla

    captured: dict = {}

    class DummyLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.reduce_results = True

    class DummyRope(nn.Module):
        pass

    class DummyWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["args"] = args
            captured["kwargs"] = kwargs
            bound = _bind_mla_wrapper_call(args, kwargs)
            self.o_proj = DummyLinear()
            self.rotary_emb = DummyRope()
            self.mla_attn = SimpleNamespace(
                layer_name=f"{bound['prefix']}.attn",
                non_causal_multi_token_decode=kwargs["non_causal_multi_token_decode"],
            )

    monkeypatch.setattr(common_dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(common_dspark_mla, "RMSNorm", DummyLinear)
    monkeypatch.setattr(
        amd_dspark_mla, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(amd_dspark_mla, "MergedColumnParallelLinear", DummyLinear)
    monkeypatch.setattr(amd_dspark_mla, "ColumnParallelLinear", DummyLinear)
    monkeypatch.setattr(amd_dspark_mla, "RowParallelLinear", DummyLinear)
    monkeypatch.setattr(amd_dspark_mla, "RMSNorm", DummyLinear)
    monkeypatch.setattr(amd_dspark_mla, "KimiMLP", DummyLinear)
    monkeypatch.setattr(amd_dspark_mla, "get_rope", lambda *args, **kwargs: DummyRope())
    monkeypatch.setattr(amd_dspark_mla, "KimiK3DSparkMLAWrapper", DummyWrapper)

    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        rms_norm_eps=1e-6,
        intermediate_size=16,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default"},
    )
    vllm_config = SimpleNamespace(cache_config=None)
    layer = amd_dspark_mla.K3DSparkDecoderLayer(
        vllm_config=vllm_config,
        config=config,
        layer_idx=0,
        start_layer_id=61,
        prefix="model",
    )

    bound = _bind_mla_wrapper_call(captured["args"], captured["kwargs"])
    assert isinstance(layer.self_attn, DummyWrapper)
    assert layer.self_attn.o_proj.reduce_results is False
    assert captured["kwargs"]["non_causal_multi_token_decode"] is True
    assert bound["prefix"] == "model.layers.61.self_attn"
    assert bound["mla_modules"].rotary_emb is not None
    assert layer.self_attn.mla_attn.layer_name == "model.layers.61.self_attn.attn"


def test_amd_kv_cache_layer_returns_inner_mla_attn():
    from vllm.models.kimi_k3.amd.dspark_mla import K3DSparkForCausalLM, K3DSparkModel

    inner = SimpleNamespace(layer_name="model.layers.3.self_attn.attn")
    attn = SimpleNamespace(mla_attn=inner)
    assert K3DSparkModel.kv_cache_layer(SimpleNamespace(), attn) is inner

    owner = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attn)],
            kv_cache_layer=lambda module: K3DSparkModel.kv_cache_layer(
                SimpleNamespace(), module
            ),
        )
    )
    assert K3DSparkForCausalLM.get_draft_kv_cache_layer_names(owner) == [
        "model.layers.3.self_attn.attn"
    ]


def _yarn_cast_wrapper():
    """Wrapper whose RoPE returns fp32, matching HIP YaRN."""
    from vllm.model_executor.layers.mla import MLAModules
    from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

    class RecordingAttn(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kv_cache_dtype = "auto"
            self.impl = SimpleNamespace(dcp_world_size=1)
            self.use_pcp = False
            self.hisparse_cache = None
            self.q_pad_num_heads = None
            self.is_aiter_triton_fp4_bmm_enabled = False
            self.is_aiter_triton_fp8_bmm_enabled = False
            self.W_UK_T = None
            self.layer_name = kwargs.get("prefix", "attn")
            self.seen_k_pe = None
            self.seen_q = None

        def forward(self, q, kv_c_normed, k_pe, output_shape=None, **kwargs):
            del kv_c_normed, kwargs
            self.seen_q = q
            self.seen_k_pe = k_pe
            return torch.zeros(output_shape, dtype=q.dtype)

    class Fp32Rope(nn.Module):
        def forward(self, positions, q_pe, k_pe):
            del positions
            return q_pe.to(torch.float32), torch.ones_like(k_pe, dtype=torch.float32)

    class Proj(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width

        def forward(self, x):
            return torch.zeros(x.shape[0], self.width, dtype=torch.bfloat16), None

    class Identity(nn.Module):
        def forward(self, x):
            return x

    class CastWrapper(KimiK3MultiHeadLatentAttentionWrapper):
        mla_attn_cls = RecordingAttn

    modules = MLAModules(
        kv_a_layernorm=Identity(),
        kv_b_proj=Identity(),
        rotary_emb=Fp32Rope(),
        o_proj=Proj(8),
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=Proj(6),
        q_a_layernorm=None,
        q_b_proj=None,
        q_proj=Proj(12),
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )
    return CastWrapper(
        hidden_size=8,
        num_heads=2,
        scale=1.0,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=4,
        mla_modules=modules,
        prefix="model.layers.0.self_attn",
    )


def test_hip_yarn_k_pe_is_cast_back_to_kv_dtype():
    wrapper = _yarn_cast_wrapper()
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)
    positions = torch.zeros(2, dtype=torch.int64)
    wrapper(positions, hidden)
    assert wrapper.mla_attn.seen_q.dtype == torch.bfloat16
    assert wrapper.mla_attn.seen_k_pe.dtype == torch.bfloat16


def test_fused_decode_not_taken_when_rotary_emb_is_set():
    wrapper = _yarn_cast_wrapper()
    wrapper._fused_qk_prep = True

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("identity RoPE fusion must not run after YaRN")

    wrapper._fused_decode = fail_if_called
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)
    positions = torch.zeros(2, dtype=torch.int64)
    wrapper(positions, hidden)
    assert wrapper.mla_attn.seen_k_pe is not None


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
    head.markov_w2.quant_method.process_weights_after_loading(head.markov_w2)

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

    monkeypatch.setattr(common_dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(common_dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(
        common_dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj
    )
    monkeypatch.setattr(common_dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(K3DSparkModel, "decoder_layer_cls", DummyModule)
    monkeypatch.setattr(common_dspark_mla, "DSparkMarkovHead", make_markov_head)

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
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_native_mega_moe=False))],
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


def test_dsv4_context_wkv_weights_are_duplicated_by_draft_layer():
    weights = [
        ("mtp.0.attn.wkv.weight", torch.arange(4)),
        ("mtp.1.attn.wq_a.weight", torch.arange(3)),
        ("mtp.2.attn.wkv.scale", torch.tensor(0.5)),
        ("mtp.3.attn.wkv.weight", torch.arange(2)),
    ]

    duplicated = list(dsv4_dspark._duplicate_context_wkv_weights(weights, 3))

    assert [name for name, _ in duplicated] == [
        "mtp.0.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.attn.wkv.scale",
        "context_wkv_proj.scale",
        "mtp.3.attn.wkv.weight",
    ]
    assert duplicated[1][1].shard_id == 0
    assert duplicated[4][1].shard_id == 2
    assert duplicated[0][1].data_ptr() == duplicated[1][1].data_ptr()
    assert duplicated[3][1].data_ptr() == duplicated[4][1].data_ptr()


def test_dsv4_context_kv_uses_one_stacked_wkv_projection(monkeypatch):
    calls = []
    stacked_output = torch.arange(24, dtype=torch.float32).view(2, 12)

    class StackedProjection:
        def __init__(self):
            self.calls = 0

        def __call__(self, main_x):
            self.calls += 1
            assert main_x.shape == (2, 5)
            return stacked_output

    projection = StackedProjection()
    layers = [
        SimpleNamespace(attn=SimpleNamespace(kv_norm=lambda kv, offset=i: kv + offset))
        for i in range(3)
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(head_dim=4),
        context_wkv_proj=projection,
        layers=layers,
        num_dspark_layers=3,
    )
    slot_mappings = [torch.tensor([0, 1]), None, torch.tensor([4, 5])]
    monkeypatch.setattr(
        dsv4_dspark,
        "_insert_context_kv",
        lambda attn, kv, positions, slots: calls.append(
            (attn, kv.clone(), positions, slots)
        ),
    )

    dsv4_dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model,
        torch.zeros(2, 5),
        torch.tensor([7, 8]),
        slot_mappings,
    )

    assert projection.calls == 1
    assert len(calls) == 2
    assert torch.equal(calls[0][1], stacked_output.view(2, 3, 4)[:, 0])
    assert torch.equal(calls[1][1], stacked_output.view(2, 3, 4)[:, 2] + 2)
    assert calls[0][3] is slot_mappings[0]
    assert calls[1][3] is slot_mappings[2]


@pytest.mark.cpu_test
def test_k3_dspark_mla_kv_cache_spec_groups_with_target_mla():
    """The draft's MLA layers must share a KV cache group with the target's."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size

    vllm_config = SimpleNamespace(
        model_config=None, cache_config=SimpleNamespace(block_size=64)
    )
    target_attn = SimpleNamespace(
        kv_cache_dtype="fp8",
        head_size=576,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
        attn_backend=SimpleNamespace(get_name=lambda: "ROCM_AITER_MLA"),
        _uses_flat_kv_cache=lambda: False,
    )
    draft_attn = SimpleNamespace(
        kv_cache_dtype="fp8", head_size=576, non_causal_multi_token_decode=True
    )
    target_spec = MLAAttention.get_kv_cache_spec(target_attn, vllm_config)
    draft_spec = MultiHeadLatentAttention.get_kv_cache_spec(draft_attn, vllm_config)

    kv_cache_spec = {f"model.layers.{i}.attn": target_spec for i in range(24)}
    kv_cache_spec |= {f"draft.layers.{i}.attn": draft_spec for i in range(5)}

    groups = _get_kv_cache_groups_uniform_page_size(kv_cache_spec)
    assert len(groups) == 1
    assert len(groups[0].layer_names) == 29


def test_fused_qk_rope_concat_requires_fp32_cos_sin():
    from vllm._aiter_ops import rocm_aiter_ops

    if not bool(rocm_aiter_ops.is_fused_qk_rope_concat_and_cache_mla_enabled()):
        pytest.skip("AITER fused_qk_rope_concat_and_cache_mla is not available")

    dummy = torch.zeros(1, 1, 8)
    bf16_table = torch.zeros(4, 32, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="fp32"):
        rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla(
            dummy,
            dummy,
            dummy.view(1, 8),
            dummy.view(1, 8),
            dummy,
            dummy,
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1),
            torch.ones(1),
            torch.zeros(1, dtype=torch.int64),
            bf16_table,
            bf16_table,
            is_neox=False,
        )
