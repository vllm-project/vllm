# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 on the DeepGEMM NVFP4 MegaMoE layer: gating, the checkpoint
adapter, and the fused block against a dequantized reference."""

import gc
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)
from vllm.models.minimax_m3.nvidia.model import (
    MEGA_MOE_FC2_INPUT_SCALE,
    _mega_moe_expert_params_mapping,
    _mega_moe_expert_weight,
    _use_mega_moe,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="NVFP4 MegaMoE requires CUDA"
)


def _vllm_config(moe_backend: str, moe_quant_algo: str | None):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        quant_config=SimpleNamespace(moe_quant_algo=moe_quant_algo),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
    )


@pytest.mark.parametrize(
    ("moe_backend", "moe_quant_algo", "expected"),
    [
        ("auto", "NVFP4", False),
        ("deep_gemm_mega_moe", "NVFP4", True),
        ("deep_gemm_mega_moe", None, NotImplementedError),
    ],
)
def test_mega_moe_is_gated_on_nvfp4_checkpoints(moe_backend, moe_quant_algo, expected):
    """MegaMoE is opt-in and refuses checkpoints whose experts are not NVFP4."""
    vllm_config = _vllm_config(moe_backend, moe_quant_algo)
    if expected is NotImplementedError:
        with pytest.raises(NotImplementedError, match="NVFP4 routed experts"):
            _use_mega_moe(vllm_config)
    else:
        assert _use_mega_moe(vllm_config) is expected


def test_mega_moe_expert_weight_follows_the_qat_recipe():
    """Compressed-tensors quantization scales become ModelOpt dequantization
    scales; the FC1 input scale is dropped (per-token) and the FC2 one is the
    fixed QAT scale whatever the checkpoint holds."""
    prefix = "layers.3.block_sparse_moe.experts.7"
    packed = torch.randint(0, 255, (4, 8), dtype=torch.uint8)
    block_scale = torch.rand(4, 1).to(torch.float8_e4m3fn)

    assert _mega_moe_expert_weight(f"{prefix}.w1.weight_packed", packed) == (
        f"{prefix}.w1.weight",
        packed,
    )
    assert _mega_moe_expert_weight(f"{prefix}.w3.weight_scale", block_scale) == (
        f"{prefix}.w3.weight_scale",
        block_scale,
    )
    name, scale = _mega_moe_expert_weight(
        f"{prefix}.w2.weight_global_scale", torch.tensor(448.0)
    )
    assert name == f"{prefix}.w2.weight_scale_2"
    assert torch.equal(scale, torch.tensor(1 / 448, dtype=torch.float32))
    for shard_id in ("w1", "w3"):
        assert (
            _mega_moe_expert_weight(
                f"{prefix}.{shard_id}.input_global_scale", torch.tensor(1.0)
            )
            is None
        )
    name, scale = _mega_moe_expert_weight(
        f"{prefix}.w2.input_global_scale", torch.tensor(1.0)
    )
    assert name == f"{prefix}.w2.input_scale"
    assert scale.item() == MEGA_MOE_FC2_INPUT_SCALE


def _load_checkpoint(moe, tensors: dict[str, torch.Tensor]) -> None:
    """``MiniMaxM3Model.load_weights`` for one MegaMoE block's tensors."""
    experts = moe.experts
    mapping = _mega_moe_expert_params_mapping(experts.num_experts)
    for name, tensor in tensors.items():
        if ".shared_experts." in name:
            name = name.replace(".gate_proj.", ".w1.").replace(".up_proj.", ".w3.")
            experts.load_shared_expert_weight(name, tensor)
            continue
        converted = _mega_moe_expert_weight(name, tensor)
        if converted is None:
            continue
        name, tensor = converted
        param_name, weight_name, expert_id, shard_id = next(
            entry for entry in mapping if entry[1] in name
        )
        name = name.replace(weight_name, param_name)
        param = getattr(experts, name.rpartition(".")[2])
        param.weight_loader(param, tensor, name, shard_id=shard_id, expert_id=expert_id)


def _mega_moe_block(num_experts=8, top_k=2, hidden_size=512, intermediate_size=256):
    """An SM100 MiniMax-M3.1 MoE block on MegaMoE with an MXFP8 shared expert."""
    if not current_platform.is_device_capability_family(100):
        pytest.skip("DeepGEMM NVFP4 MegaMoE requires SM100")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MoE

    config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        n_shared_experts=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
        use_routing_bias=False,
        hidden_act="swigluoai",
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        moe_quant_algo="NVFP4",
    )
    vllm_config = VllmConfig(
        kernel_config=KernelConfig(moe_backend="deep_gemm_mega_moe")
    )
    vllm_config.scheduler_config.max_num_batched_tokens = 256
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        moe = MiniMaxM3MoE(
            config,
            layer_id=0,
            quant_config=quant_config,
            prefix="layers.0.block_sparse_moe",
            use_mega_moe=True,
        )
    assert isinstance(moe.experts, DeepGemmMegaMoEExperts)
    return moe, config


@pytest.mark.parametrize("fused", [True, False])
def test_mega_moe_allocates_the_shared_expert_once(dist_init, monkeypatch, fused):
    """The block builds a real shared MLP only when MegaMoE does not fuse it
    (e.g. an older DeepGEMM); a fused layer owns the only copy, and building
    the block leaves no discarded weights for the garbage collector."""
    if not fused:
        monkeypatch.setattr(
            DeepGemmMegaMoEExperts,
            "_kernel_fuses_shared_experts",
            lambda self, spec: False,
        )
    gc.collect()
    gc.disable()
    try:
        moe, _ = _mega_moe_block()
        allocated = torch.accelerator.memory_allocated()
        gc.collect()
    finally:
        gc.enable()
    assert torch.accelerator.memory_allocated() == allocated
    assert moe.experts.has_fused_shared_experts is fused
    assert (moe.shared_experts is None) is fused
    assert all(param.is_cuda for param in moe.parameters())


@pytest.mark.parametrize("num_tokens", [1, 45, 200])
def test_mega_moe_block_matches_the_qat_reference(dist_init, num_tokens):
    """A MiniMax-M3.1 MoE block on MegaMoE fuses its MXFP8 shared expert and,
    loaded from a compressed-tensors checkpoint, reproduces the QAT recipe:
    per-token FC1 input scales and the fixed FC2 input scale on the unweighted
    SwiGLU output. Routing weights scale the BF16 expert outputs in combine."""
    from vllm.third_party.deep_gemm.utils import (
        cast_back_from_fp8,
        per_token_cast_to_fp8,
    )
    from vllm.third_party.deep_gemm.utils.nvfp4 import (
        cast_back_from_nvfp4,
        per_token_cast_to_nvfp4,
    )

    torch.manual_seed(0)
    moe, config = _mega_moe_block()
    assert moe.experts.has_fused_shared_experts and moe.shared_experts is None
    num_experts, top_k = config.num_local_experts, config.num_experts_per_tok
    hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
    clamp, alpha, beta = config.swiglu_limit, config.swiglu_alpha, config.swiglu_beta

    def nvfp4(n, k):
        weight = torch.randn(n, k, device="cuda") * k**-0.5
        global_scale = weight.abs().max().item() / (6 * 448)
        packed, sf = per_token_cast_to_nvfp4(weight, global_scale)
        dequant = cast_back_from_nvfp4(packed, sf, global_scale)
        return (
            packed.view(torch.uint8),
            sf.view(torch.float8_e4m3fn),
            torch.tensor(1 / global_scale),
            dequant,
        )

    def mxfp8(n, k):
        weight = torch.randn(n, k, device="cuda") * k**-0.5
        fp8, sf = per_token_cast_to_fp8(weight, use_ue8m0=True, gran_k=32)
        ue8m0 = (sf.view(torch.int32) >> 23).to(torch.uint8)
        return fp8, ue8m0, cast_back_from_fp8(fp8, sf, gran_k=32)

    prefix = "layers.0.block_sparse_moe"
    checkpoint: dict[str, torch.Tensor] = {}
    ref_w13, ref_w2 = [], []
    for e in range(num_experts):
        w13 = nvfp4(2 * intermediate_size, hidden_size)
        w2 = nvfp4(hidden_size, intermediate_size)
        for shard_id, rows, (packed, sf, global_scale, _) in (
            ("w1", slice(0, intermediate_size), w13),
            ("w3", slice(intermediate_size, None), w13),
            ("w2", slice(None), w2),
        ):
            name = f"{prefix}.experts.{e}.{shard_id}"
            checkpoint[f"{name}.weight_packed"] = packed[rows]
            checkpoint[f"{name}.weight_scale"] = sf[rows]
            checkpoint[f"{name}.weight_global_scale"] = global_scale
            # A placeholder the block must ignore.
            checkpoint[f"{name}.input_global_scale"] = torch.tensor(0.25)
        ref_w13.append(w13[3])
        ref_w2.append(w2[3])
    shared_ref = {}
    for proj, (n, k) in (
        ("gate_proj", (intermediate_size, hidden_size)),
        ("up_proj", (intermediate_size, hidden_size)),
        ("down_proj", (hidden_size, intermediate_size)),
    ):
        fp8, ue8m0, shared_ref[proj] = mxfp8(n, k)
        checkpoint[f"{prefix}.shared_experts.{proj}.weight"] = fp8
        checkpoint[f"{prefix}.shared_experts.{proj}.weight_scale"] = ue8m0
    _load_checkpoint(moe, checkpoint)

    # Rows span six decades, which one tensor-wide input scale cannot cover.
    x = (
        torch.randn(num_tokens, hidden_size, device="cuda")
        * torch.logspace(-4, 2, num_tokens, device="cuda")[:, None]
    ).to(torch.bfloat16)
    scores = torch.randn(num_tokens, num_experts, device="cuda").sigmoid()
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights *= config.routed_scaling_factor / topk_weights.sum(-1, keepdim=True)
    out = moe.experts(x, topk_weights, topk_ids)

    def activation(h):
        gate, up = h.to(torch.bfloat16).float().chunk(2, dim=-1)
        gate = gate.clamp(max=clamp)
        up = up.clamp(-clamp, clamp)
        return gate * torch.sigmoid(alpha * gate) * (up + beta)

    def fp8_round_trip(h):
        return cast_back_from_fp8(
            *per_token_cast_to_fp8(h, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True),
            gran_k=32,
            use_packed_ue8m0=True,
        )

    x_scales = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 2688
    x_ref = cast_back_from_nvfp4(*per_token_cast_to_nvfp4(x.float() / x_scales))
    x_ref *= x_scales
    ref = torch.zeros(num_tokens, hidden_size, device="cuda")
    for e in range(num_experts):
        rows, slots = torch.where(topk_ids == e)
        if rows.numel() == 0:
            continue
        mid = activation(x_ref[rows] @ ref_w13[e].T)
        mid = cast_back_from_nvfp4(
            *per_token_cast_to_nvfp4(mid, MEGA_MOE_FC2_INPUT_SCALE),
            MEGA_MOE_FC2_INPUT_SCALE,
        )
        expert_out = (mid @ ref_w2[e].T).to(torch.bfloat16).float()
        ref.index_add_(0, rows, topk_weights[rows, slots, None] * expert_out)
    shared_w13 = torch.cat([shared_ref["gate_proj"], shared_ref["up_proj"]])
    shared_mid = fp8_round_trip(activation(fp8_round_trip(x) @ shared_w13.T))
    ref += (shared_mid @ shared_ref["down_proj"].T).to(torch.bfloat16).float()

    assert torch.isfinite(out).all()
    # Per row: a missing per-token scale would only show in the small rows.
    row_error = (out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert row_error.max() < 0.02, row_error
