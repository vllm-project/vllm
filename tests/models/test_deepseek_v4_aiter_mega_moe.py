# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.models.deepseek_v4.amd import mega_moe as mega_moe_module
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
)
from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP, DeepseekV4MoE
from vllm.models.deepseek_v4.amd.rocm import (
    apply_e8m0_linear,
    apply_pre_quantized_block_scaled_mm,
    prepare_e8m0_linears,
)
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.torch_utils import aux_stream, set_default_torch_dtype

HIDDEN = 64
INTER = 64


@pytest.fixture
def e8m0_config(default_vllm_config, dist_init, monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8_MXSCALE", "1")
    LlamaConfig(
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=64,
        max_position_embeddings=2048,
    ).save_pretrained(tmp_path)
    default_vllm_config.model_config = ModelConfig(
        model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
    )


@pytest.fixture
def e8m0_mlp(e8m0_config):
    torch.manual_seed(621)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        layer = DeepseekV4MLP(
            512,
            512,
            "silu",
            quant_config=Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
            ),
        )
    weights = []
    scales = []
    for _ in range(3):
        weights.append(
            (torch.randn(512, 512, device="cuda") * 16).to(torch.float8_e4m3fn)
        )
        scales.append(torch.exp2(torch.randint(-10, -7, (4, 4), device="cuda").float()))
    for name, values in (("weight", weights), ("weight_scale_inv", scales)):
        merged = getattr(layer.gate_up_proj, name)
        for shard in (0, 1):
            merged.weight_loader(merged, values[shard], shard)
        down = getattr(layer.down_proj, name)
        down.weight_loader(down, values[2])
    for linear in (layer.gate_up_proj, layer.down_proj):
        linear.quant_method.process_weights_after_loading(linear)
    layer.prepare_gateup_preshuffle()
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield (
        layer,
        torch.cat([w.view(torch.uint8) for w in weights[:2]]).view(torch.float8_e4m3fn),
        torch.cat(scales[:2]),
        weights[2],
        scales[2],
    )
    torch.backends.cuda.matmul.allow_tf32 = before


def _e8m0_mlp_reference(x, layer, gate_weight, gate_scale, down_weight, down_scale):
    from aiter import gemm_a8w8_blockscale
    from aiter.ops.quant import per_group_quant_hip

    shape = x.shape
    if x.numel() == 0:
        return x.new_empty((*shape[:-1], down_weight.shape[0]))

    def matmul(value, weight, scale):
        m, k = value.shape
        n = weight.shape[0]
        quantized, encoded = per_group_quant_hip(
            value,
            quant_dtype=torch.float8_e4m3fn,
            group_size=128,
            transpose_scale=True,
            scale_type=torch.float8_e8m0fnu,
        )
        decoded = torch.exp2(
            encoded.view(torch.uint8).float().reshape(k // 128, m).T - 127
        )
        lhs = (
            quantized.float().reshape(m, k // 128, 128) * decoded[..., None]
        ).reshape(m, k)
        rhs = (
            weight.float().reshape(n // 128, 128, k // 128, 128)
            * scale[:, None, :, None]
        ).reshape(n, k)
        reference = (lhs @ rhs.T).bfloat16()
        baseline = gemm_a8w8_blockscale(
            quantized, weight, decoded.contiguous(), scale, dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            baseline.float(), reference.float(), atol=1e-2, rtol=1e-2
        )
        # Requantization can amplify small GEMM rounding differences. Use the
        # established FP8 arithmetic for the complete layer comparison.
        return baseline

    gate = matmul(x.reshape(-1, shape[-1]), gate_weight, gate_scale)
    activated = layer.act_fn(gate)
    result = matmul(activated, down_weight, down_scale)
    return result.reshape(*shape[:-1], result.shape[-1])


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
@pytest.mark.parametrize("num_tokens", [0, 7, 48, 56, 84, 112, 512])
@pytest.mark.parametrize("offset", [0, 1])
def test_e8m0_mlp_matches_dequantized_reference(e8m0_mlp, num_tokens, offset):
    """The real MLP must consume E8M0 bytes correctly for empty and offset inputs."""
    layer, *weights = e8m0_mlp
    torch.manual_seed(37 + num_tokens)
    data = torch.randn(num_tokens * 512 + offset, device="cuda", dtype=torch.bfloat16)
    x = data[offset:].reshape(num_tokens, 512)
    expected = _e8m0_mlp_reference(x, layer, *weights)
    actual = layer(x)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
def test_e8m0_mlp_preserves_leading_dimensions(e8m0_mlp):
    layer, *weights = e8m0_mlp
    x = torch.randn(2, 7, 512, device="cuda", dtype=torch.bfloat16)
    expected = _e8m0_mlp_reference(x, layer, *weights)
    torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
def test_e8m0_mlp_capture_reuses_packed_weights(e8m0_mlp):
    """Preparing twice and replaying with new inputs must preserve the real weights."""
    layer, *weights = e8m0_mlp
    layer.prepare_gateup_preshuffle()
    x = torch.randn(112, 512, device="cuda", dtype=torch.bfloat16)
    layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x)
    x.mul_(-0.5).add_(0.3)
    graph.replay()
    expected = _e8m0_mlp_reference(x, layer, *weights)
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
def test_e8m0_mlp_compiles_with_encoded_scales(e8m0_mlp):
    """The functional quantizer must expose valid E8M0 outputs to compilation."""
    layer, *weights = e8m0_mlp
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(layer, fullgraph=True)
    actual = compiled(x)
    expected = _e8m0_mlp_reference(x, layer, *weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture
def e8m0_query_linears(e8m0_config, request):
    torch.manual_seed(712)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(1536, eps=1e-6)
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        linears = [
            ColumnParallelLinear(
                1536, 2048, bias=False, return_bias=False, quant_config=config
            ),
            ReplicatedLinear(
                1536, 1024, bias=False, return_bias=False, quant_config=config
            ),
        ]
    weights, scales = [], []
    for index, linear in enumerate(linears):
        n = linear.weight.shape[0]
        weight = (torch.randn(n, 1536, device="cuda") * 16).to(torch.float8_e4m3fn)
        scale = torch.exp2(
            torch.randint(-10, -7, (n // 128, 12), device="cuda").float()
        )
        if index == 1 and getattr(request, "param", None) == "inexact":
            scale[0, 0] *= 0.75
        for name, value in (("weight", weight), ("weight_scale_inv", scale)):
            parameter = getattr(linear, name)
            parameter.weight_loader(parameter, value)
        linear.quant_method.process_weights_after_loading(linear)
        weights.append(weight)
        scales.append(scale)
    return norm, linears, weights, scales


def _e8m0_query_outputs(x, norm, linears):
    quantized, scale = rocm_aiter_ops.rmsnorm_e8m0_quant(
        x, norm.weight, norm.variance_epsilon
    )
    return tuple(
        apply_pre_quantized_block_scaled_mm(linear, quantized, scale)
        for linear in linears
    )


def _check_e8m0_query_outputs(x, norm, linears, weights, scales, actual):
    import aiter

    m, k = x.shape
    quantized = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    encoded = torch.empty((m, k // 128), device="cuda", dtype=torch.float8_e8m0fnu)
    if m:
        aiter.rmsnorm_quant(
            quantized, x, encoded, norm.weight, norm.variance_epsilon, 128, True
        )
    actual_q, actual_s = rocm_aiter_ops.rmsnorm_e8m0_quant(
        x, norm.weight, norm.variance_epsilon
    )
    torch.testing.assert_close(
        actual_q.view(torch.uint8), quantized.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual_s.view(torch.uint8), encoded.view(torch.uint8), atol=0, rtol=0
    )
    scale = torch.exp2(encoded.view(torch.uint8).float().reshape(k // 128, m).T - 127)
    dequantized = (
        quantized.float().reshape(m, k // 128, 128) * scale[..., None]
    ).reshape(m, k)
    if m:
        normalized = x.float()
        normalized = (
            normalized
            * torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + norm.variance_epsilon
            )
            * norm.weight.float()
        )
        limit = (
            normalized.abs() * 0.0625 + scale.repeat_interleave(128, -1) * 2**-10 + 5e-5
        )
        assert not bool(((dequantized - normalized).abs() > limit).any())
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for out, weight, scale in zip(actual, weights, scales):
            n = weight.shape[0]
            rhs = (
                weight.float().reshape(n // 128, 128, k // 128, 128)
                * scale[:, None, :, None]
            ).reshape(n, k)
            assert out.dtype == torch.bfloat16
            torch.testing.assert_close(
                out.float(), dequantized @ rhs.T, atol=1e-2, rtol=1e-2
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
@pytest.mark.parametrize("num_tokens", [0, 7, 56, 84, 112, 512])
@pytest.mark.parametrize("offset", [0, 1])
@torch.inference_mode()
def test_e8m0_qnorm_feeds_both_real_query_linears(
    e8m0_query_linears, num_tokens, offset
):
    """Both coupled consumers must read encoded scales from the strided Q view."""
    norm, linears, weights, scales = e8m0_query_linears
    assert prepare_e8m0_linears(linears)
    data = torch.randn(num_tokens * 2048 + offset, device="cuda", dtype=torch.bfloat16)
    x = data[offset:].reshape(num_tokens, 2048)[:, :1536]
    actual = _e8m0_query_outputs(x, norm, linears)
    _check_e8m0_query_outputs(x, norm, linears, weights, scales, actual)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
@pytest.mark.parametrize("mode", ["capture", "compile"])
@torch.inference_mode()
def test_e8m0_qnorm_captures_and_compiles(e8m0_query_linears, mode):
    norm, linears, weights, scales = e8m0_query_linears
    assert prepare_e8m0_linears(linears)
    assert prepare_e8m0_linears(linears)
    x = torch.randn(84, 2048, device="cuda", dtype=torch.bfloat16)[:, :1536]
    if mode == "compile":
        forward = torch.compile(_e8m0_query_outputs, fullgraph=True)
        actual = forward(x, norm, linears)
    else:
        _e8m0_query_outputs(x, norm, linears)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _e8m0_query_outputs(x, norm, linears)
        x.mul_(-0.5).add_(0.3)
        graph.replay()
    _check_e8m0_query_outputs(x, norm, linears, weights, scales, actual)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
@pytest.mark.parametrize("e8m0_query_linears", ["inexact"], indirect=True)
@torch.inference_mode()
def test_e8m0_qnorm_keeps_both_consumers_on_fallback(e8m0_query_linears):
    """An incompatible second consumer must leave the first linear usable too."""
    _, linears, _, _ = e8m0_query_linears
    x = torch.randn(7, 1536, device="cuda", dtype=torch.bfloat16)
    expected = [linear(x) for linear in linears]
    assert not prepare_e8m0_linears(linears)
    for linear, before in zip(linears, expected):
        torch.testing.assert_close(linear(x), before, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="Requires gfx950 E8M0 GEMMs",
)
@pytest.mark.parametrize("shape", [(0, 1536), (7, 1536), (2, 7, 1536)])
@torch.inference_mode()
def test_e8m0_qnorm_draft_projection_preserves_dimensions(e8m0_query_linears, shape):
    """The draft's BF16 input projection must preserve all leading dimensions."""
    _, linears, weights, scales = e8m0_query_linears
    linear = linears[1]
    assert prepare_e8m0_linears([linear])
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = apply_e8m0_linear(linear, x)
    expected_shape = (*shape[:-1], 1024)
    assert out.shape == expected_shape
    assert out.dtype == torch.bfloat16
    if x.numel() == 0:
        return
    flat = x.reshape(-1, 1536)
    quantized, encoded = rocm_aiter_ops.group_e8m0_quant(flat, transpose_scale=True)
    scale = torch.exp2(
        encoded.view(torch.uint8).float().reshape(12, flat.shape[0]).T - 127
    )
    lhs = (quantized.float().reshape(-1, 12, 128) * scale[..., None]).reshape(-1, 1536)
    rhs = (
        weights[1].float().reshape(8, 128, 12, 128) * scales[1][:, None, :, None]
    ).reshape(1024, 1536)
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = (lhs @ rhs.T).reshape(expected_shape)
        torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


def _make_experts(
    monkeypatch,
    *,
    ep_rank=0,
    ep_size=1,
    num_experts=2,
    intermediate_size=INTER,
    max_num_batched_tokens=3000,
    swiglu_limit=None,
    fuse_shared_expert=False,
):
    monkeypatch.setattr(
        mega_moe_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=ep_rank, world_size=ep_size, cpu_group="ep"
        ),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    )
    return DeepseekV4AiterMegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        swiglu_limit=swiglu_limit,
        fuse_shared_expert=fuse_shared_expert,
    )


@pytest.mark.cpu_test
def test_experts_load_into_local_slots(monkeypatch):
    model = torch.nn.Module()
    model.ffn = torch.nn.Module()
    model.ffn.experts = _make_experts(monkeypatch, ep_rank=1, ep_size=2, num_experts=4)
    params = dict(model.named_parameters())
    mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=4,
        routed_experts_prefix="",
    )

    shapes = {
        ("w1", "weight"): (INTER, HIDDEN // 2),
        ("w3", "weight"): (INTER, HIDDEN // 2),
        ("w2", "weight"): (HIDDEN, INTER // 2),
        ("w1", "weight_scale"): (INTER, HIDDEN // 32),
        ("w3", "weight_scale"): (INTER, HIDDEN // 32),
        ("w2", "weight_scale"): (HIDDEN, INTER // 32),
    }
    shard_value = {"w1": 1, "w3": 2, "w2": 3}
    loaded = {}
    for expert in range(4):
        for (shard, suffix), shape in shapes.items():
            name = f"ffn.experts.{expert}.{shard}.{suffix}"
            weight = torch.full(
                shape, 10 * expert + shard_value[shard], dtype=torch.uint8
            )
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                if param.weight_loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded[name] = mapped
                    break

    assert sorted({n.split(".")[2] for n in loaded}) == ["2", "3"]
    assert len(loaded) == 2 * len(shapes)
    for slot, expert in enumerate((2, 3)):
        for suffix in ("weight", "weight_scale"):
            w13 = params[f"ffn.experts.w13_{suffix}"][slot]
            w2 = params[f"ffn.experts.w2_{suffix}"][slot]
            assert torch.all(w13[:INTER] == 10 * expert + 1)
            assert torch.all(w13[INTER:] == 10 * expert + 2)
            assert torch.all(w2 == 10 * expert + 3)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, expected",
    [([4, 128], "decode"), ([4, 129], "prefill")],
)
def test_runtime_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, expected
):
    experts = _make_experts(monkeypatch)
    experts._runtimes = [(128, "decode"), (4096, "prefill")]
    monkeypatch.setattr(mega_moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        ),
    )
    # Every rank must take the same runtime, so a small local batch still gets
    # the prefill runtime when another DP rank is over the decode capacity.
    assert experts.select_runtime(4) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "moe_backend, targets, fused",
    [
        ("aiter_mega_moe", None, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp8"}, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp4"}, True),
        ("aiter", {"*shared_experts*": "mxfp4"}, False),
    ],
)
def test_mega_moe_shared_expert_fusion_requires_mxfp4_target(
    moe_backend, targets, fused
):
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    quant_config._resolved_expert_dtype = "fp4"
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}
    if targets is not None:
        quant_config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=targets)
        )
    compatible, _ = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.3.ffn.experts",
        "model.layers.3.ffn.shared_experts",
        moe_backend=moe_backend,
    )
    assert compatible is fused


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_shared_expert_route_appended(monkeypatch):
    experts = _make_experts(
        monkeypatch, ep_rank=3, ep_size=8, num_experts=384, fuse_shared_expert=True
    )
    ids = torch.randint(0, 384, (37, 2), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(37, 2, device="cuda")
    out_weights, out_ids = experts.append_shared_expert(weights, ids)

    shared = torch.full((37, 1), 3 * 49 + 48, device="cuda", dtype=torch.int32)
    expected_ids = torch.cat([torch.where(ids >= 0, ids + ids // 48, ids), shared], 1)
    torch.testing.assert_close(out_ids, expected_ids)
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones(37, 1, device="cuda")], 1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("shared_stream", [False, True])
def test_mega_moe_routing_matches_reference(
    monkeypatch, hash_routing, num_tokens, shared_stream
):
    num_experts, top_k, vocab = 16, 4, 64
    scaling = 2.5
    device = "cuda"
    torch.manual_seed(0)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(num_experts, HIDDEN, device=device) * 0.1
            self.e_score_correction_bias = (
                None if hash_routing else torch.randn(num_experts, device=device)
            )
            self.tid2eid = (
                torch.stack(
                    [
                        torch.randperm(num_experts, device=device)[:top_k]
                        for _ in range(vocab)
                    ]
                ).to(torch.int32)
                if hash_routing
                else None
            )
            self.bias_vl = None

        def forward(self, x):
            return x.float() @ self.weight.t(), None

    captured = {}

    def experts(x, topk_weights, topk_ids):
        captured["weights"], captured["ids"] = topk_weights, topk_ids
        return 2 * x

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.experts = experts
    moe.shared_experts = lambda x: 3 * x
    moe.shared_experts_stream = aux_stream() if shared_stream else None
    moe.shared_experts_events = (
        (torch.cuda.Event(), torch.cuda.Event()) if shared_stream else None
    )
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = top_k
    moe.renormalize = True
    moe.routed_scaling_factor = scaling
    moe.image_sentinel_lo = 0

    x = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device)
    out = moe._forward_mega_moe(x, input_ids)

    ids, weights = captured["ids"], captured["weights"]
    assert ids.dtype == torch.int32 and ids.is_contiguous()
    assert weights.dtype == torch.float32 and weights.is_contiguous()

    scores = torch.nn.functional.softplus(x.float() @ moe.gate.weight.t()).sqrt()
    if hash_routing:
        expected_ids = moe.gate.tid2eid[input_ids].long()
    else:
        expected_ids = (scores + moe.gate.e_score_correction_bias).topk(top_k).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= scaling / expected_weights.sum(dim=-1, keepdim=True)

    order = ids.long().sort(dim=-1).indices
    expected_order = expected_ids.sort(dim=-1).indices
    torch.testing.assert_close(
        ids.long().gather(1, order), expected_ids.gather(1, expected_order)
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        expected_weights.gather(1, expected_order),
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(out, 5 * x)
