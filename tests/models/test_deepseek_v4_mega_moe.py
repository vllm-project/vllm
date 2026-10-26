# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import DeepseekV4Config

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.model_executor.models.interfaces import (
    get_mixture_of_experts_model,
    is_mixture_of_experts,
)
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE as DeepseekV41MoE
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 MegaMoE requires CUDA",
)
requires_mega_moe_gpu = pytest.mark.skipif(
    not (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
    ),
    reason="DeepSeek V4 MegaMoE requires SM90 or SM100",
)


@pytest.fixture(autouse=True)
def skip_mega_moe_support_check(monkeypatch):
    """Skip the MegaMoE support check; kernels are gated per test."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


@pytest.fixture
def v41_moe_config(dist_init):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=DeepseekV41Config(
                text_config=dict(
                    hidden_size=128,
                    num_hidden_layers=2,
                    n_routed_experts=8,
                    num_experts_per_tok=2,
                    dspark_n_routed_experts=4,
                    dspark_num_experts_per_tok=3,
                    n_shared_experts=1,
                    moe_intermediate_size=128,
                    hidden_act="silu",
                    swiglu_limit=10.0,
                    norm_topk_prob=True,
                    topk_method="noaux_tc",
                    routed_scaling_factor=1.5,
                ),
            ),
        ),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@requires_mega_moe_gpu
@pytest.mark.parametrize("use_cudagraph", [False, True])
@pytest.mark.parametrize("above_threshold", [False, True])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("layer_id,num_experts,top_k", [(0, 384, 6), (2, 128, 3)])
def test_deepseek_v41_moe_routes_without_hash_table(
    v41_moe_config,
    monkeypatch,
    vision,
    layer_id,
    num_experts,
    top_k,
    above_threshold,
    use_cudagraph,
):
    """Main and draft layers select experts by score and preserve image routing."""
    config = v41_moe_config.model_config.hf_config
    config.hidden_size = 256
    config.n_routed_experts = 384
    config.num_experts_per_tok = 6
    config.dspark_n_routed_experts = 128
    num_tokens = (1 if num_experts == 128 else 16) + int(above_threshold)
    config.vision_n_layers = int(vision)
    with (
        set_default_torch_dtype(v41_moe_config.model_config.dtype),
        torch.device("cuda"),
    ):
        moe = DeepseekV41MoE(v41_moe_config, prefix=f"model.layers.{layer_id}.ffn")
        hidden_states = torch.randn(
            num_tokens, config.hidden_size, dtype=torch.bfloat16
        )
        input_ids = (
            torch.tensor(
                [42, IMAGE_SENTINEL_BASE_ID, IMAGE_SENTINEL_BASE_ID, 129257]
            ).repeat((num_tokens + 3) // 4)[:num_tokens]
            if vision
            else None
        )
    assert moe.gate.tid2eid is None
    assert moe.gate.weight.shape == (num_experts, config.hidden_size)
    assert isinstance(moe.experts, DeepGemmMegaMoEExperts)
    assert moe.experts.activation == MegaMoEActivation("swiglu", clamp=10.0)
    assert moe.experts.w13_weight.shape == (num_experts, 256, 128)
    assert moe.experts.top_k == top_k
    assert (config.n_routed_experts, config.num_experts_per_tok) == (384, 6)

    with torch.no_grad():
        moe.gate.weight.normal_(std=0.01)
        moe.gate.e_score_correction_bias.copy_(torch.arange(num_experts, device="cuda"))
        if vision:
            moe.gate.bias_vl.copy_(-moe.gate.e_score_correction_bias)

    scores = torch.nn.functional.softplus(
        torch.mm(hidden_states, moe.gate.weight.t(), out_dtype=torch.float32)
    ).sqrt()
    bias = moe.gate.e_score_correction_bias
    if vision:
        image_mask = input_ids == IMAGE_SENTINEL_BASE_ID
        bias = torch.where(image_mask[:, None], moe.gate.bias_vl, bias)
    expected_ids = (scores + bias).topk(top_k, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= config.routed_scaling_factor / expected_weights.sum(
        dim=-1, keepdim=True
    )

    routed = {}

    def check_routing(x, weights, ids, shared_experts):
        routed["ids"], routed["weights"] = ids, weights
        return x + shared_experts(x)

    monkeypatch.setattr(moe.experts, "forward", check_routing)
    monkeypatch.setattr(moe.shared_experts, "forward", lambda x: 2 * x)
    routing_input_ids = (
        input_ids
        if input_ids is not None
        else torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
    )
    metadata = prepare_mega_gate_routing_metadata(
        routing_input_ids,
        has_hash_routing=False,
        image_sentinel_base_id=IMAGE_SENTINEL_BASE_ID if vision else None,
    )
    if use_cudagraph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                moe(hidden_states, input_ids, metadata)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = moe(hidden_states, input_ids, metadata)
        graph.replay()
        torch.accelerator.synchronize()
    else:
        output = moe(hidden_states, input_ids, metadata)

    ids, weights = routed["ids"], routed["weights"]
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(
        weights,
        expected_weights,
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(output, 3 * hidden_states)


@pytest.mark.parametrize(
    "draft_experts,draft_top_k,expected", [(0, 0, (8, 2)), (4, 3, (4, 3))]
)
def test_deepseek_v41_fused_moe_uses_draft_counts_or_main_defaults(
    v41_moe_config, monkeypatch, draft_experts, draft_top_k, expected
):
    config = v41_moe_config.model_config.hf_config
    config.dspark_n_routed_experts = draft_experts
    config.dspark_num_experts_per_tok = draft_top_k
    v41_moe_config.kernel_config.moe_backend = "auto"
    captured = {}

    def make_experts(**kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.model.FusedMoEFactory", make_experts
    )
    moe = DeepseekV41MoE(v41_moe_config, prefix="model.layers.2.ffn")
    assert moe.gate.weight.shape == (expected[0], 128)
    assert (captured["num_experts"], captured["top_k"]) == expected
    assert captured["hash_indices_table"] is None


def test_deepseek_v4_moe_preserves_configured_hash_layers(v41_moe_config):
    config = v41_moe_config.model_config.hf_config
    config.num_hash_layers = 1
    config.vocab_size = 32
    moe = DeepseekV4MoE(
        v41_moe_config,
        prefix="model.layers.0.ffn",
        num_hash_layers=config.num_hash_layers,
    )
    assert moe.gate.tid2eid.shape == (32, 2)
    assert moe.gate.e_score_correction_bias is None
    with pytest.raises(ValueError, match="hash MoE routing requires input_ids"):
        moe(torch.zeros(1, 128))


@requires_mega_moe_gpu
@pytest.mark.parametrize("num_tokens", [16, 17])
def test_deepseek_v4_mega_gate_hash_routing_correctness(
    v41_moe_config, monkeypatch, num_tokens
):
    config = v41_moe_config.model_config.hf_config
    config.hidden_size = 256
    config.num_hash_layers = 1
    config.vocab_size = 32
    with (
        set_default_torch_dtype(v41_moe_config.model_config.dtype),
        torch.device("cuda"),
    ):
        moe = DeepseekV4MoE(
            v41_moe_config,
            prefix="model.layers.0.ffn",
            num_hash_layers=config.num_hash_layers,
        )
        hidden_states = torch.randn(
            num_tokens, config.hidden_size, dtype=torch.bfloat16
        )
        input_ids = torch.arange(num_tokens)

    token_ids = torch.arange(config.vocab_size, device="cuda")
    fixed_ids = torch.stack(
        (
            token_ids % config.n_routed_experts,
            (token_ids + 3) % config.n_routed_experts,
        ),
        dim=1,
    )
    with torch.no_grad():
        moe.gate.weight.normal_(std=0.01)
        moe.gate.tid2eid.copy_(fixed_ids)

    expected_ids = fixed_ids[input_ids]
    scores = torch.nn.functional.softplus(
        torch.mm(hidden_states, moe.gate.weight.t(), out_dtype=torch.float32)
    ).sqrt()
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= config.routed_scaling_factor / expected_weights.sum(
        dim=-1, keepdim=True
    )

    def check_routing(x, weights, ids, shared_experts):
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-3, atol=1e-4)
        return x + shared_experts(x)

    monkeypatch.setattr(moe.experts, "forward", check_routing)
    monkeypatch.setattr(moe.shared_experts, "forward", torch.zeros_like)
    metadata = prepare_mega_gate_routing_metadata(
        input_ids,
        has_hash_routing=True,
        image_sentinel_base_id=None,
    )
    torch.testing.assert_close(moe(hidden_states, input_ids, metadata), hidden_states)


def test_deepseek_v4_mega_moe_expert_mapping():
    mapping = make_deepseek_v4_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]


def test_deepseek_v4_pwal_hook_finalizes_mhc_broadcast():
    """The loader invokes the model-level PWAL hook for every load format,
    so it must finalize mhc broadcast weights to cover dummy load, which
    skips load_weights()."""
    calls = []
    stub = SimpleNamespace(
        model=SimpleNamespace(
            finalize_mhc_broadcast_weights=lambda: calls.append("mhc"),
        )
    )

    DeepseekV4ForCausalLM.process_weights_after_loading(stub)

    assert calls == ["mhc"]


@pytest.fixture
def v4_dspark_config(dist_init):
    hf_config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=128,
        num_hidden_layers=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_hash_layers=0,
        n_shared_experts=1,
        moe_intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=10.0,
        norm_topk_prob=True,
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        hc_mult=1,
        hc_eps=1e-5,
        rms_norm_eps=1e-5,
        dspark_target_layer_ids=[0],
        dspark_markov_rank=8,
        index_topk=4,
        head_dim=64,
        num_attention_heads=4,
        vocab_size=256,
        n_mtp_layers=2,
        enable_confidence_head=False,
        layer_types=["sliding_attention"] * 2,
        q_lora_rank=32,
        o_lora_rank=32,
        qk_rope_head_dim=16,
        o_groups=4,
        sliding_window=128,
        hc_sinkhorn_iters=1,
    )
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=hf_config,
        max_model_len=2048,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        kernel_config=SimpleNamespace(
            moe_backend="deep_gemm_mega_moe",
            enable_jit_warmup=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=4),
        ),
        attention_config=SimpleNamespace(backend=None),
        cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(draft_model_config=model_config),
        use_v2_model_runner=False,
    )


@requires_mega_moe_gpu
def test_dspark_draft_registers_mixture_of_experts(
    v4_dspark_config, monkeypatch, dist_init
):
    """DSV4 DSpark MoE drafts expose MixtureOfExperts for EPLB registration."""
    hf_config = v4_dspark_config.model_config.hf_config
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.dspark.get_current_vllm_config",
        lambda: v4_dspark_config,
    )

    with (
        set_default_torch_dtype(v4_dspark_config.model_config.dtype),
        torch.device("cuda"),
    ):
        draft = DSparkDeepseekV4ForCausalLM(vllm_config=v4_dspark_config)

    assert is_mixture_of_experts(draft)
    assert get_mixture_of_experts_model(draft) is draft
    assert len(draft.moe_layers) == draft.model.num_dspark_layers == 2
    assert draft.num_routed_experts == hf_config.n_routed_experts
    assert draft.num_redundant_experts == 4
