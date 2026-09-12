# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4MegaMoEExperts,
    DeepseekV4MoE,
    make_deepseek_v4_expert_params_mapping,
)
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_PAD_ID,
    IMAGE_SENTINEL_BASE_ID,
)
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4MoE as DeepseekV41MoE
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 MegaMoE requires CUDA",
)


@pytest.fixture
def v41_moe_config(dist_init):
    return SimpleNamespace(
        model_config=SimpleNamespace(
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


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("layer_id,num_experts,top_k", [(0, 8, 2), (2, 4, 3)])
def test_deepseek_v41_moe_routes_without_hash_table(
    v41_moe_config, monkeypatch, vision, layer_id, num_experts, top_k
):
    """Main and draft layers select experts by score and preserve image routing."""
    config = v41_moe_config.model_config.hf_config
    config.vision_n_layers = int(vision)
    with torch.device("cuda"):
        moe = DeepseekV41MoE(v41_moe_config, prefix=f"model.layers.{layer_id}.ffn")
        hidden_states = torch.randn(4, config.hidden_size)
        input_ids = (
            torch.tensor([42, IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID, 129257])
            if vision
            else None
        )
    assert moe.gate.tid2eid is None
    assert moe.gate.weight.shape == (num_experts, config.hidden_size)
    assert isinstance(moe.experts, DeepseekV4MegaMoEExperts)
    assert moe.experts.w13_weight.shape == (num_experts, 256, 64)
    assert moe.experts.top_k == top_k
    assert (config.n_routed_experts, config.num_experts_per_tok) == (8, 2)

    with torch.no_grad():
        moe.gate.weight.normal_(std=0.01)
        moe.gate.e_score_correction_bias.copy_(torch.arange(num_experts, device="cuda"))
        if vision:
            moe.gate.bias_vl.copy_(-moe.gate.e_score_correction_bias)

    scores = torch.nn.functional.softplus(
        torch.nn.functional.linear(hidden_states, moe.gate.weight)
    ).sqrt()
    bias = moe.gate.e_score_correction_bias
    if vision:
        image_mask = (input_ids == IMAGE_SENTINEL_BASE_ID) | (input_ids == IMAGE_PAD_ID)
        bias = torch.where(image_mask[:, None], moe.gate.bias_vl, bias)
    expected_ids = (scores + bias).topk(top_k, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= config.routed_scaling_factor / expected_weights.sum(
        dim=-1, keepdim=True
    )

    def check_routing(x, weights, ids, *, activation_clamp):
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(weights, expected_weights)
        assert activation_clamp == config.swiglu_limit
        return x.clone()

    monkeypatch.setattr(moe.experts, "forward", check_routing)
    monkeypatch.setattr(moe.shared_experts, "forward", lambda x: 2 * x)
    torch.testing.assert_close(moe(hidden_states, input_ids), 3 * hidden_states)


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


def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
    raw = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)

    decoded = DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float(raw)

    assert torch.equal(decoded.view(torch.int32), raw.to(torch.int32) << 23)
    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 0.5
    assert decoded[2].item() == 1.0
    assert decoded[3].item() == 2.0


@pytest.mark.parametrize("use_kimi", [False, True])
def test_deep_gemm_mega_moe_capture_precedes_eplb(monkeypatch, use_kimi):
    experts_cls = DeepseekV4MegaMoEExperts
    if use_kimi:
        from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts

        experts_cls = KimiK3MegaMoEExperts

    experts = experts_cls.__new__(experts_cls)
    torch.nn.Module.__init__(experts)
    if use_kimi:
        experts.synchronize_first_launch = lambda: None
    experts.prefix = "model.layers.3.ffn.experts"
    experts.max_num_tokens = 4
    experts.capture_fn = None
    experts.get_symm_buffer = lambda: object()
    experts.eplb_state = SimpleNamespace(
        logical_to_physical_map=torch.empty(1),
        expert_load_view=torch.empty(1),
        logical_replica_count=torch.empty(1),
        should_record_tensor=torch.empty(1),
        num_unpadded_tokens_tensors=None,
    )

    topk_ids = torch.tensor([[1, 2], [3, 4]])
    captured: list[tuple[int, torch.Tensor]] = []
    bind_routed_experts_capturer(
        SimpleNamespace(modules=lambda: [experts]),
        SimpleNamespace(capture=lambda layer_id, ids: captured.append((layer_id, ids))),
    )

    class MappingReached(Exception):
        pass

    def map_ids(**kwargs):
        assert captured == [(3, topk_ids)]
        raise MappingReached

    monkeypatch.setattr(
        f"{experts_cls.__module__}.eplb_map_to_physical_and_record",
        map_ids,
    )
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm", lambda: SimpleNamespace()
    )

    with pytest.raises(MappingReached):
        experts(
            torch.empty(2, 8),
            torch.empty(2, 2),
            topk_ids,
            activation_clamp=None,
        )


def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
    )

    nonlocal_weight = torch.ones(128, 64, dtype=torch.uint8)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_weight,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((128, 64), 3, dtype=torch.uint8)
    w3 = torch.full((128, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 64), 11, dtype=torch.uint8)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :128], w1)
    assert torch.equal(experts.w13_weight[0, 128:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


@pytest.mark.parametrize("intermediate_size", [512, 2304])
def test_deepseek_v4_mega_moe_padding_preserves_weights(monkeypatch, intermediate_size):
    """Pad gate/up separately and zero every added down-projection column."""
    experts = DeepseekV4MegaMoEExperts(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=intermediate_size,
    )
    originals = []
    for param in (
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
    ):
        param.data.random_(1, 128)
        originals.append(param.data.clone())
    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(
            transform_sf_into_required_layout=lambda sf, *args: sf,
            transform_weights_for_mega_moe=lambda l1, l2: (l1, l2),
        ),
    )

    experts.finalize_weights()
    padded_size = experts.intermediate_size
    assert padded_size == (intermediate_size + 511) // 512 * 512
    for actual, original in zip(experts._transformed_l1_weights, originals[:2]):
        expected = (
            original
            if actual.dtype == torch.int8
            else (experts._ue8m0_uint8_to_float(original))
        )
        actual = actual.view(expected.dtype).unflatten(1, (2, padded_size))
        assert torch.equal(
            actual[:, :, :intermediate_size],
            expected.unflatten(1, (2, intermediate_size)),
        )
        assert torch.count_nonzero(actual[:, :, intermediate_size:]) == 0
    for actual, original in zip(experts._transformed_l2_weights, originals[2:]):
        expected = (
            original
            if actual.dtype == torch.int8
            else (experts._ue8m0_uint8_to_float(original))
        )
        actual = actual.view(expected.dtype)
        assert torch.equal(actual[..., : original.shape[-1]], expected)
        assert torch.count_nonzero(actual[..., original.shape[-1] :]) == 0

    transformed_l1 = experts._transformed_l1_weights
    experts.finalize_weights()
    assert experts._transformed_l1_weights is transformed_l1


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,block_size,fused,mxfp8",
    [
        pytest.param(128, 512, 128, True, False, id="aligned-block128"),
        pytest.param(128, 512, 128, True, True, id="aligned-block128-mxfp8"),
        pytest.param(5120, 2304, 32, False, False, id="deepseek-v41-flash"),
    ],
)
def test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights(
    monkeypatch, hidden_size, intermediate_size, block_size, fused, mxfp8
):
    """V4.1-Flash's routed padding must preserve the serial shared-expert fallback."""

    class FakeDeepGemm:
        transformed_dims: list[tuple[int, int]] = []
        scale_inputs: list[tuple[int, ...]] = []

        @staticmethod
        def get_symm_buffer_for_mega_moe(*args, num_shared_experts=0, **kwargs):
            return None

        @staticmethod
        def get_block_m_for_mega_moe(*args, **kwargs):
            return 128

        @staticmethod
        def fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            shared_l1_weights=None,
            shared_l2_weights=None,
            **kwargs,
        ):
            return None

        @classmethod
        def transform_sf_into_required_layout(cls, sf, mn, k, *args, **kwargs):
            cls.scale_inputs.append(tuple(sf.shape))
            return torch.empty((sf.shape[0], mn, k // 128), dtype=torch.int32)

        @classmethod
        def transform_weights_for_mega_moe(cls, l1_weights, l2_weights):
            cls.transformed_dims.append((l1_weights[0].dim(), l2_weights[0].dim()))
            if l1_weights[0].dim() == 2:
                return (l1_weights[0].clone(), l1_weights[1]), l2_weights
            return l1_weights, l2_weights

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_shared_experts=1,
    )
    experts._check_runtime_supported = lambda: None

    def fp8_parameter(*shape):
        return torch.nn.Parameter(
            torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )

    def scale_parameter(*shape, dtype=torch.int32):
        return torch.nn.Parameter(torch.ones(*shape, dtype=dtype), requires_grad=False)

    shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=fp8_parameter(2 * intermediate_size, hidden_size),
            weight_block_size=(block_size, block_size),
            weight_scale_inv=scale_parameter(
                2 * intermediate_size // block_size,
                hidden_size // block_size,
                dtype=torch.float8_e8m0fnu,
            ),
        ),
        down_proj=SimpleNamespace(
            weight=fp8_parameter(hidden_size, intermediate_size),
            weight_block_size=(block_size, block_size),
            weight_scale_inv=scale_parameter(
                hidden_size // block_size,
                intermediate_size // block_size,
                dtype=torch.float8_e8m0fnu,
            ),
        ),
    )
    if mxfp8:
        for linear in (shared_experts.gate_up_proj, shared_experts.down_proj):
            linear.weight_block_size = (1, 32)
            linear.weight_scale = torch.nn.Parameter(
                linear.weight_scale_inv.view(torch.uint8)
                .repeat_interleave(128, dim=0)
                .repeat_interleave(4, dim=1),
                requires_grad=False,
            )
            del linear.weight_scale_inv
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm)

    original_gate_up_ptr = shared_experts.gate_up_proj.weight.data_ptr()
    original_down_ptr = shared_experts.down_proj.weight.data_ptr()
    experts.finalize_weights(shared_experts)

    assert FakeDeepGemm.scale_inputs[-2:] == [
        (1, 2 * intermediate_size, hidden_size // 32),
        (1, hidden_size, intermediate_size // 32),
    ]
    assert experts.has_fused_shared_experts is fused
    if not fused:
        assert experts.intermediate_size == 2560
        assert FakeDeepGemm.transformed_dims == [(3, 3)]
        assert experts.num_shared_experts == 0
        assert shared_experts.gate_up_proj.weight.data_ptr() == original_gate_up_ptr
        assert shared_experts.down_proj.weight.data_ptr() == original_down_ptr
        assert shared_experts.gate_up_proj.weight.shape == (
            2 * intermediate_size,
            hidden_size,
        )
        assert shared_experts.down_proj.weight.shape == (hidden_size, intermediate_size)
        return

    assert FakeDeepGemm.transformed_dims == [(3, 3), (2, 2)]
    assert shared_experts.gate_up_proj.weight.data_ptr() != original_gate_up_ptr
    assert (
        experts._transformed_shared_l1_weights[0].data_ptr()
        == shared_experts.gate_up_proj.weight.data_ptr()
    )
    assert (
        experts._transformed_shared_l2_weights[0].data_ptr()
        == shared_experts.down_proj.weight.data_ptr()
    )


@pytest.mark.parametrize("fused", [False, True])
def test_deepseek_v4_mega_moe_does_not_double_add_fused_shared_expert(
    monkeypatch, fused
):
    class FakeGate(torch.nn.Module):
        tid2eid = None
        e_score_correction_bias = None

        def forward(self, hidden_states):
            return torch.empty(hidden_states.shape[0], 2), None

    class FakeExperts(torch.nn.Module):
        has_fused_shared_experts = fused

        def forward(self, hidden_states, *args, **kwargs):
            return torch.ones_like(hidden_states)

    class FakeSharedExperts(torch.nn.Module):
        calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.full_like(hidden_states, 2)

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 1
    moe.renormalize = True
    moe.hash_indices_dtype = torch.int64
    moe.routed_scaling_factor = 1.0
    moe.swiglu_limit = 10.0
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.model.fused_topk_bias",
        lambda **kwargs: (
            torch.ones(kwargs["hidden_states"].shape[0], 1),
            torch.zeros(kwargs["hidden_states"].shape[0], 1, dtype=torch.int64),
        ),
    )

    output = moe(torch.zeros(2, 128))

    expected = 1 if fused else 3
    assert torch.all(output == expected)
    assert moe.shared_experts.calls == (0 if fused else 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, :32] = 0
    hidden_states[1, 32:64] = 1.0e-6
    hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_weights = topk_weights.clone()

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


@pytest.mark.parametrize("shared_block_m", [8, 32, 96, 128, 192])
def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout(shared_block_m):
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = shared_block_m + 7
    hidden_size = 256
    top_k = 8
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_block_m)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    aligned_block_m = ((shared_block_m + 127) // 128) * 128
    num_shared_rows = ((num_tokens + shared_block_m - 1) // shared_block_m) * (
        aligned_block_m
    )
    ref_shared_x_sf = torch.zeros(
        num_shared_rows,
        hidden_size // 128,
        dtype=torch.int32,
        device=device,
    )
    for token_id in range(num_tokens):
        m_in_block = token_id % shared_block_m
        transposed_m = (
            (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
        )
        shared_row = token_id // shared_block_m * aligned_block_m + transposed_m
        ref_shared_x_sf[shared_row].copy_(ref_x_sf[token_id])

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_shared_storage = torch.full(
        (hidden_size // 128, num_shared_rows),
        -1,
        dtype=torch.int32,
        device=device,
    )
    fused_shared_x_sf = fused_shared_storage.t()
    fused_topk_idx = torch.empty_like(topk_ids, dtype=torch.int64)
    fused_topk_weights = torch.empty_like(topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        shared_x_sf=fused_shared_x_sf,
        shared_block_m=shared_block_m,
    )
    torch.accelerator.synchronize()

    populated = ref_shared_x_sf != 0
    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_shared_x_sf[populated], ref_shared_x_sf[populated])


def test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast():
    """The loader invokes the model-level PWAL hook for every load format,
    so it must finalize megamoe + mhc broadcast weights to cover dummy
    load, which skips load_weights()."""
    calls = []
    stub = SimpleNamespace(
        model=SimpleNamespace(
            finalize_mega_moe_weights=lambda: calls.append("mega_moe"),
            finalize_mhc_broadcast_weights=lambda: calls.append("mhc"),
        )
    )

    DeepseekV4ForCausalLM.process_weights_after_loading(stub)

    assert calls == ["mega_moe", "mhc"]


def test_deepseek_v4_drafter_pwal_hooks_finalize_mega_moe():
    """MTP/DSpark drafters load as their own top-level models, so each needs
    its own PWAL hook now that the megamoe forward no longer finalizes
    weights lazily on first use."""
    calls = []
    mtp = SimpleNamespace(finalize_mega_moe_weights=lambda: calls.append("mtp"))
    DeepSeekV4MTP.process_weights_after_loading(mtp)

    dspark = SimpleNamespace(_finalize_moe=lambda: calls.append("dspark"))
    DSparkDeepseekV4ForCausalLM.process_weights_after_loading(dspark)

    assert calls == ["mtp", "dspark"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_masks_padding():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    is_padding = torch.tensor(
        [False, True, False, False, True, False, True],
        device=device,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_idx[is_padding] = -1
    ref_topk_weights = topk_weights.clone()
    ref_topk_weights[is_padding] = 0.0

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )
