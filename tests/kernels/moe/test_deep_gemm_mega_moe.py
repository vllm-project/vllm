# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MegaMoE experts: kernel-layout weight loading, shared-expert fusion,
input staging, and numerics against a dequantized reference for every routed
format, activation and shared-expert fusion in the feature matrix."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_BF16,
    MEGA_MOE_MXFP4,
    MEGA_MOE_WEIGHT_FORMATS,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_megamoe_inputs,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepGEMM MegaMoE requires CUDA"
)


@pytest.fixture(autouse=True)
def sm100_mega_moe_layout(monkeypatch):
    """Build the SM100 kernel layout on any GPU; kernels are gated per test."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _make_experts(**kwargs) -> DeepGemmMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    kwargs.setdefault("activation", MegaMoEActivation())
    return DeepGemmMegaMoEExperts(vllm_config, **kwargs)


def test_deep_gemm_mega_moe_capture_precedes_eplb(monkeypatch):
    experts = DeepGemmMegaMoEExperts.__new__(DeepGemmMegaMoEExperts)
    torch.nn.Module.__init__(experts)
    experts._synchronize_first_launch = lambda: None
    experts.prefix = "model.layers.3.ffn.experts"
    experts.max_num_tokens = 4
    experts.num_shared_experts = 0
    experts.capture_fn = None
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
        f"{DeepGemmMegaMoEExperts.__module__}.eplb_map_to_physical_and_record",
        map_ids,
    )

    with pytest.raises(MappingReached):
        experts(torch.empty(2, 8), torch.empty(2, 2), topk_ids)


def test_deep_gemm_mega_moe_weight_loader_uses_ep_expert_ownership():
    experts = _make_experts(
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

    interleaved = experts.w13_weight[0].view(torch.uint8).view(16, 2, 8, 64)
    assert torch.equal(interleaved[:, 0], w1.view(16, 8, 64))
    assert torch.equal(interleaved[:, 1], w3.view(16, 8, 64))
    assert torch.equal(experts.w2_weight[0].view(torch.uint8), w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


def _reference_mega_moe_layout(w13, s13, w2, s2):
    """CPU copy of DeepGEMM's transform_weights_for_mega_moe on packed scales."""

    def interleave(t):
        g, n, k = t.shape
        halves = t.view(g, 2, n // 16, 8, k)
        return torch.stack([halves[:, 0], halves[:, 1]], dim=2).reshape(g, n, k)

    def utccp(sf):
        g, mn, k = sf.shape
        return sf.reshape(g, -1, 4, 32, k).transpose(2, 3).reshape(g, mn, k)

    return (
        interleave(w13),
        utccp(interleave(s13.view(torch.int32))),
        w2,
        utccp(s2.view(torch.int32)),
    )


def _make_mega_moe_experts(num_local_experts, num_logical_experts):
    return _make_experts(
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=0,
        top_k=1,
        hidden_size=256,
        intermediate_size=384,
        num_logical_experts=num_logical_experts,
    )


def _load_mega_moe_checkpoint_shuffled(experts, num_logical_experts, seed):
    hidden, inter = experts.hidden_size, experts.intermediate_size
    shapes = {
        "w13_weight": (num_logical_experts, 2 * inter, hidden // 2),
        "w13_weight_scale": (num_logical_experts, 2 * inter, hidden // 32),
        "w2_weight": (num_logical_experts, hidden, inter // 2),
        "w2_weight_scale": (num_logical_experts, hidden, inter // 32),
    }
    generator = torch.Generator().manual_seed(seed)
    ckpt = {
        name: torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        for name, shape in shapes.items()
    }
    shards = []
    for expert_id in range(num_logical_experts):
        for name, tensor in ckpt.items():
            if name.startswith("w2_"):
                shards.append((name, tensor[expert_id], "w2", expert_id))
            else:
                shards.append((name, tensor[expert_id, :inter], "w1", expert_id))
                shards.append((name, tensor[expert_id, inter:], "w3", expert_id))
    for i in torch.randperm(len(shards), generator=generator).tolist():
        name, loaded_weight, shard_id, expert_id = shards[i]
        if name.endswith("scale"):
            loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
        assert experts.weight_loader(
            getattr(experts, name),
            loaded_weight,
            f"experts.{name}",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
    return ckpt


def _expected_mega_moe_layout(ckpt, num_local_experts, num_logical_experts):
    physical = torch.arange(num_local_experts) % num_logical_experts
    ckpt = {name: tensor[physical] for name, tensor in ckpt.items()}
    expected = _reference_mega_moe_layout(*ckpt.values())
    if is_deep_gemm_supported():
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        w13, s13, w2, s2 = (t.cuda() for t in ckpt.values())
        hidden, inter = w2.shape[1], w2.shape[2] * 2
        scales = [
            deep_gemm.transform_sf_into_required_layout(
                (sf.to(torch.int32) << 23).view(torch.float32),
                mn,
                k,
                (1, 32),
                num_local_experts,
            )
            for sf, mn, k in ((s13, 2 * inter, hidden), (s2, hidden, inter))
        ]
        (l1, l1_sf), (l2, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13.view(torch.int8), scales[0]), (w2.view(torch.int8), scales[1])
        )
        for actual, reference in zip((l1, l1_sf, l2, l2_sf), expected):
            assert torch.equal(actual.cpu().view(reference.dtype), reference)
    return expected


def _assert_mega_moe_layout(experts, expected):
    actual = (*experts.l1_weights, *experts.l2_weights)
    for tensor, reference in zip(actual, expected):
        assert torch.equal(tensor.view(reference.dtype), reference)


def test_deep_gemm_mega_moe_loads_shards_into_kernel_layout():
    """Shards in any order, including EPLB replicas, land in DeepGEMM's layout."""
    experts = _make_mega_moe_experts(num_local_experts=4, num_logical_experts=3)
    ckpt = _load_mega_moe_checkpoint_shuffled(experts, 3, seed=0)
    _assert_mega_moe_layout(experts, _expected_mega_moe_layout(ckpt, 4, 3))
    # DeepGEMM's MN-major scale strides.
    assert experts.w13_weight_scale.stride() == (768 * 2, 1, 768)
    assert experts.w2_weight_scale.stride() == (256 * 3, 1, 256)


def test_deep_gemm_mega_moe_reload_overwrites_kernel_layout_in_place():
    """A second load must update the live kernel tensors without reallocating."""
    experts = _make_mega_moe_experts(num_local_experts=2, num_logical_experts=2)
    _load_mega_moe_checkpoint_shuffled(experts, 2, seed=0)
    data_ptrs = [param.data_ptr() for param in experts.parameters()]
    ckpt = _load_mega_moe_checkpoint_shuffled(experts, 2, seed=1)
    assert [param.data_ptr() for param in experts.parameters()] == data_ptrs
    _assert_mega_moe_layout(experts, _expected_mega_moe_layout(ckpt, 2, 2))


@pytest.mark.parametrize("intermediate_size", [512, 2304])
def test_deep_gemm_mega_moe_preserves_checkpoint_dimensions(intermediate_size):
    """Keep native widths so V4.1 routed and shared experts can fuse."""
    experts = _make_experts(
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=intermediate_size,
    )

    assert experts.intermediate_size == intermediate_size
    assert experts.w13_weight.shape == (2, 2 * intermediate_size, 64)
    assert experts.w13_weight_scale.shape == (2, 2 * intermediate_size, 1)
    assert experts.w2_weight.shape == (2, 128, intermediate_size // 2)
    assert experts.w2_weight_scale.shape == (2, 128, intermediate_size // 128)
    transformed = (*experts.l1_weights, *experts.l2_weights)
    for actual, param in zip(transformed, experts.parameters()):
        assert actual.data_ptr() == param.data_ptr()


def _shared_mlp(
    hidden_size,
    shared_size,
    weight_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float8_e8m0fnu,
    scale_block=(1, 32),
):
    def linear(n, k):
        scale = None
        if scale_dtype is not None:
            scale = torch.empty(n // scale_block[0], k // scale_block[1]).to(
                scale_dtype
            )
        return SimpleNamespace(
            weight=torch.empty(n, k, dtype=weight_dtype), weight_scale_inv=scale
        )

    return SimpleNamespace(
        gate_up_proj=linear(2 * shared_size, hidden_size),
        down_proj=linear(hidden_size, shared_size),
    )


def test_deep_gemm_mega_moe_loads_shared_experts_into_kernel_layout(monkeypatch):
    """Shared expert shards land directly in the MegaMoE kernel layout."""
    monkeypatch.setattr(
        DeepGemmMegaMoEExperts, "_kernel_fuses_shared_experts", lambda self: True
    )
    experts = _make_experts(
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=256,
        intermediate_size=384,
        shared_experts=_shared_mlp(256, 384),
    )
    assert experts.has_fused_shared_experts
    hidden, inter = 256, 384
    assert experts.has_fused_shared_experts
    hidden, inter = 256, 384
    generator = torch.Generator().manual_seed(42)

    def rand(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)

    w1, w3, w2 = rand(inter, hidden), rand(inter, hidden), rand(hidden, inter)
    s1, s3, s2 = (
        rand(inter, hidden // 32),
        rand(inter, hidden // 32),
        rand(hidden, inter // 32),
    )
    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    # The down shard arrives as `.w2.` (MTP, DSpark) or `.down_proj.` (mapped).
    for name, data in [
        ("shared_experts.w1.weight", w1.view(fp8)),
        ("shared_experts.w3.weight", w3.view(fp8)),
        ("shared_experts.down_proj.weight", w2.view(fp8)),
        ("shared_experts.w1.weight_scale", s1.view(e8m0)),
        ("shared_experts.w3.weight_scale", s3.view(e8m0)),
        ("shared_experts.w2.weight_scale", s2.view(e8m0)),
    ]:
        loaded = experts.load_shared_expert_weight(f"ffn.{name}", data)
        # Weight tracking checks this name against the registered parameters.
        assert loaded in {f"ffn.experts.{n}" for n, _ in experts.named_parameters()}

    def l1_halves():
        return experts.shared_l1_weight.view(torch.uint8).view(-1, 2, 8, hidden)

    # Every tensor matches DeepGEMM's batched transform of the checkpoint.
    expected = _reference_mega_moe_layout(
        *(t.unsqueeze(0) for t in (torch.cat([w1, w3]), torch.cat([s1, s3]), w2, s2))
    )
    actual = (
        experts.shared_l1_weight,
        experts.shared_l1_weight_scale,
        experts.shared_l2_weight,
        experts.shared_l2_weight_scale,
    )
    for tensor, reference in zip(actual, expected):
        assert torch.equal(tensor.view(reference.dtype), reference[0])
    # A reload of one shard overwrites in place and leaves the other half alone.
    ptr = experts.shared_l1_weight.data_ptr()
    w1b = rand(inter, hidden)
    experts.load_shared_expert_weight("shared_experts.w1.weight", w1b.view(fp8))
    assert experts.shared_l1_weight.data_ptr() == ptr
    assert torch.equal(l1_halves()[:, 0], w1b.view(-1, 8, hidden))
    assert torch.equal(l1_halves()[:, 1], w3.view(-1, 8, hidden))
    with pytest.raises(ValueError):
        experts.load_shared_expert_weight("shared_experts.gate_up_proj.weight", w1)


def _fake_deep_gemm(with_shared: bool):
    def old_buffer(group): ...

    def old_kernel(y): ...

    def new_buffer(group, num_shared_experts=0): ...

    def new_kernel(y, shared_l1_weights=None, shared_l2_weights=None): ...

    kernel = new_kernel if with_shared else old_kernel
    return SimpleNamespace(
        get_symm_buffer_for_mega_moe=new_buffer if with_shared else old_buffer,
        fp8_fp4_mega_moe=kernel,
        bf16_mega_moe=kernel,
        get_block_m_for_mega_moe=lambda *args: 0,
    )


@pytest.mark.parametrize(
    "weight_format,weight_dtype,scale_dtype,scale_block,shared_size,api,fused",
    [
        (
            MEGA_MOE_MXFP4,
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            (128, 128),
            128,
            True,
            True,
        ),
        (MEGA_MOE_MXFP4, torch.float8_e4m3fn, torch.uint8, (1, 32), 256, True, True),
        (
            MEGA_MOE_MXFP4,
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            (128, 128),
            128,
            False,
            False,
        ),
        (MEGA_MOE_MXFP4, torch.bfloat16, None, None, 128, True, False),
        (
            MEGA_MOE_MXFP4,
            torch.float8_e4m3fn,
            torch.float32,
            (128, 128),
            128,
            True,
            False,
        ),
        (MEGA_MOE_MXFP4, torch.float8_e4m3fn, torch.uint8, (1, 16), 128, True, False),
        # A TP-sharded shared MLP is not a whole multiple of the routed expert.
        (MEGA_MOE_MXFP4, torch.float8_e4m3fn, torch.uint8, (1, 32), 64, True, False),
        (MEGA_MOE_BF16, torch.bfloat16, None, None, 128, True, True),
        (MEGA_MOE_BF16, torch.float8_e4m3fn, torch.uint8, (1, 32), 128, True, False),
    ],
)
def test_deep_gemm_mega_moe_decides_shared_fusion_at_construction(
    monkeypatch,
    weight_format,
    weight_dtype,
    scale_dtype,
    scale_block,
    shared_size,
    api,
    fused,
):
    """Fuse only shared experts in the format's shared precision, shaped as whole
    routed experts, and only when the installed kernel takes them."""
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm", lambda: _fake_deep_gemm(api)
    )
    experts = _make_experts(
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        weight_format=weight_format,
        shared_experts=_shared_mlp(
            256, shared_size, weight_dtype, scale_dtype, scale_block
        ),
    )

    assert experts.num_shared_experts == (shared_size // 128 if fused else 0)
    assert hasattr(experts, "shared_l1_weight") == fused


def test_deep_gemm_mega_moe_rejects_unsupported_activation():
    with pytest.raises(NotImplementedError, match="situ"):
        _make_experts(
            num_experts=2,
            num_local_experts=2,
            experts_start_idx=0,
            top_k=1,
            hidden_size=128,
            intermediate_size=128,
            weight_format=MEGA_MOE_BF16,
            activation=MegaMoEActivation("situ"),
        )
    with pytest.raises(ValueError, match="SiTU takes no clamp"):
        MegaMoEActivation("situ", clamp=10.0)


@pytest.mark.parametrize("fused", [False, True])
def test_deep_gemm_mega_moe_adds_unfused_shared_expert_once(monkeypatch, fused):
    """The fusion decided at init fixes the launch; forward adds only an
    unfused shared expert, and refuses one the layer already fused."""
    monkeypatch.setattr(
        DeepGemmMegaMoEExperts, "_kernel_fuses_shared_experts", lambda self: True
    )
    launches = []

    def kernel(y, l1, l2, symm_buffer, **kwargs):
        launches.append(kwargs)
        y.fill_(1)

    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(
            fp8_fp4_mega_moe=kernel, get_block_m_for_mega_moe=lambda *args: 8
        ),
    )
    module = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
    monkeypatch.setattr(f"{module}.prepare_megamoe_inputs", lambda *a, **k: None)
    monkeypatch.setattr(f"{module}.get_ep_group", lambda: SimpleNamespace(world_size=1))
    shared = _shared_mlp(128, 128 if fused else 64)
    experts = _make_experts(
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        shared_experts=shared,
    )
    assert experts.has_fused_shared_experts == fused
    symm_buffer = SimpleNamespace(
        x=torch.empty(4, 128),
        x_sf=torch.empty(4, 1),
        topk_idx=torch.empty(4, 1),
        topk_weights=torch.empty(4, 1),
        shared_l1_acts_sf=torch.empty(128, 1),
        num_max_tokens_per_rank=4,
    )
    experts.get_symm_buffer = lambda: symm_buffer
    experts._synchronize_first_launch = lambda: None

    def shared_experts(x):
        return torch.full_like(x, 2)

    hidden_states = torch.zeros(2, 128, dtype=torch.bfloat16)
    topk = (torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64))
    if fused:
        with pytest.raises(ValueError, match="pass shared_experts=None"):
            experts(hidden_states, *topk, shared_experts)
        output = experts(hidden_states, *topk)
    else:
        output = experts(hidden_states, *topk, shared_experts)

    assert torch.all(output == (1 if fused else 3))
    assert ("shared_l1_weights" in launches[-1]) == fused


@pytest.mark.parametrize("num_tokens", [7, 63, 64, 65, 127, 128, 135, 434, 16384])
@pytest.mark.parametrize("hidden_size", [256, 5120])
def test_prepare_megamoe_inputs_is_bitwise_exact(num_tokens, hidden_size):
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
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
def test_prepare_megamoe_inputs_stages_shared_scale_tma_layout(shared_block_m):
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
    assert torch.all(fused_shared_x_sf[~populated] == -1)


@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("nonfinite_padding", [False, True])
@pytest.mark.parametrize("num_tokens", [7, 65, 135])
def test_prepare_megamoe_inputs_masks_padding(nonfinite_padding, num_tokens, quantize):
    """Padded tokens are unrouted; BF16 dispatch copies the hidden states."""
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
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
    is_padding = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    is_padding[1::3] = True
    is_padding[-1] = True
    if nonfinite_padding:
        topk_weights[is_padding] = float("nan")

    if quantize:
        ref_x, ref_x_sf = per_token_cast_to_fp8(
            hidden_states,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )
    else:
        ref_x, ref_x_sf = hidden_states, None
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_idx[is_padding] = -1
    ref_topk_weights = topk_weights.clone()
    ref_topk_weights[is_padding] = 0.0

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = None if ref_x_sf is None else torch.empty_like(ref_x_sf)
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
        hidden_quant=kMxfp8Dynamic if quantize else None,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    if quantize:
        assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


NUM_EXPERTS, TOP_K, HIDDEN, INTER, NUM_TOKENS = 8, 2, 512, 256, 48

ACTIVATIONS = {
    "swiglu": MegaMoEActivation(),
    "swiglu_clamp": MegaMoEActivation(clamp=0.5),
    "swiglu_alpha_beta": MegaMoEActivation(alpha=1.702, beta=1.0),
    "situ": MegaMoEActivation("situ", alpha=2.0, beta=3.0),
}


def _quantize(w: torch.Tensor, fmt: MegaMoEWeightFormat, shared: bool):
    """Return checkpoint ``(weight, scale)`` tensors and the dequantized weight."""
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    spec = fmt.shared if shared else fmt.routed
    if spec.scale_group_k is None:
        return w, None, w.float()
    if spec.values_per_element == 2:
        q, sf = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
        deq = cast_back_from_fp4(q, sf, gran_k=32)
    else:
        q, sf = per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
        deq = cast_back_from_fp8(q, sf, gran_k=32)
    return q, (sf.view(torch.int32) >> 23).to(torch.uint8), deq


def _dispatch(x: torch.Tensor, fmt: MegaMoEWeightFormat) -> torch.Tensor:
    """What the kernel computes on: MXFP8-rounded or BF16 activations."""
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp8,
        per_token_cast_to_fp8,
    )

    if fmt.hidden_quant is None:
        return x.bfloat16().float()
    return cast_back_from_fp8(
        *per_token_cast_to_fp8(x.bfloat16(), use_ue8m0=True, gran_k=32), gran_k=32
    )


def _activate(h: torch.Tensor, act: MegaMoEActivation) -> torch.Tensor:
    gate, up = h.bfloat16().float().chunk(2, dim=-1)
    if act.name == "situ":
        gate = act.alpha * torch.tanh(gate / act.alpha) * torch.sigmoid(gate)
        up = act.beta * torch.tanh(up / act.beta) if act.beta else up
        return gate * up
    if act.clamp is not None:
        gate, up = gate.clamp(max=act.clamp), up.clamp(-act.clamp, act.clamp)
    return gate * torch.sigmoid(act.alpha * gate) * (up + act.beta)


def _expert(x, w13, w2, weight, fmt, act):
    """Reference expert: weighted activation is re-dispatched before L2."""
    a = _activate(x @ w13.t(), act) * weight[:, None]
    return _dispatch(a, fmt) @ w2.t()


class _ReferenceSharedExpert(torch.nn.Module):
    def __init__(self, w13, w2, fmt, act):
        super().__init__()
        self.w13, self.w2, self.fmt, self.act = w13, w2, fmt, act

    def forward(self, x):
        x = _dispatch(x, self.fmt)
        ones = torch.ones(x.shape[0], device=x.device)
        return _expert(x, self.w13, self.w2, ones, self.fmt, self.act).bfloat16()


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)
@pytest.mark.parametrize("fuse_shared", [False, True])
@pytest.mark.parametrize("act_name", list(ACTIVATIONS))
@pytest.mark.parametrize("fmt_name", list(MEGA_MOE_WEIGHT_FORMATS))
def test_deep_gemm_mega_moe_matches_reference(
    dist_init, fmt_name, act_name, fuse_shared
):
    fmt, act = MEGA_MOE_WEIGHT_FORMATS[fmt_name], ACTIVATIONS[act_name]
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device=device) / fan_in**0.5).bfloat16()

    shared_ckpt = [
        _quantize(randn(*shape, fan_in=k), fmt, shared=True)
        for shape, k in (((INTER, HIDDEN), HIDDEN),) * 2 + (((HIDDEN, INTER), INTER),)
    ]
    shared_mlp = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=torch.cat([shared_ckpt[0][0], shared_ckpt[1][0]]),
            weight_scale_inv=None
            if shared_ckpt[0][1] is None
            else torch.cat([shared_ckpt[0][1], shared_ckpt[1][1]]),
        ),
        down_proj=SimpleNamespace(
            weight=shared_ckpt[2][0], weight_scale_inv=shared_ckpt[2][1]
        ),
    )

    def make_experts():
        with torch.device(device):
            return DeepGemmMegaMoEExperts(
                vllm_config,
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                experts_start_idx=0,
                top_k=TOP_K,
                hidden_size=HIDDEN,
                intermediate_size=INTER,
                activation=act,
                weight_format=fmt,
                shared_experts=shared_mlp if fuse_shared else None,
                prefix=f"model.layers.0.{fmt_name}.{act_name}.{fuse_shared}",
            )

    if act.name not in fmt.activations:
        with pytest.raises(NotImplementedError):
            make_experts()
        return
    experts = make_experts()
    assert experts.has_fused_shared_experts == fuse_shared

    w13 = torch.empty(NUM_EXPERTS, 2 * INTER, HIDDEN, device=device)
    w2 = torch.empty(NUM_EXPERTS, HIDDEN, INTER, device=device)
    for e in range(NUM_EXPERTS):
        for shard_id, rows, fan_in, param, deq_rows in (
            ("w1", INTER, HIDDEN, "w13", w13[e, :INTER]),
            ("w3", INTER, HIDDEN, "w13", w13[e, INTER:]),
            ("w2", HIDDEN, INTER, "w2", w2[e]),
        ):
            k = HIDDEN if param == "w13" else INTER
            weight, scale, deq = _quantize(randn(rows, k, fan_in=fan_in), fmt, False)
            deq_rows.copy_(deq)
            for suffix, tensor in (("weight", weight), ("weight_scale", scale)):
                if tensor is None:
                    continue
                name = f"{param}_{suffix}"
                assert experts.weight_loader(
                    getattr(experts, name),
                    tensor,
                    f"experts.{name}",
                    shard_id=shard_id,
                    expert_id=e,
                    return_success=True,
                )

    shared_w13 = torch.cat([shared_ckpt[0][2], shared_ckpt[1][2]])
    reference_shared = _ReferenceSharedExpert(shared_w13, shared_ckpt[2][2], fmt, act)
    if fuse_shared:
        for (weight, scale, _), shard in zip(shared_ckpt, ("w1", "w3", "w2")):
            experts.load_shared_expert_weight(f"shared_experts.{shard}.weight", weight)
            if scale is not None:
                experts.load_shared_expert_weight(
                    f"shared_experts.{shard}.weight_scale", scale
                )

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device=device).bfloat16()
    scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=device)
    topk_weights, topk_ids = torch.topk(scores.softmax(-1), TOP_K, dim=-1)

    output = experts(
        hidden_states,
        topk_weights,
        topk_ids,
        None if fuse_shared else reference_shared,
    )

    x = _dispatch(hidden_states, fmt)
    expected = reference_shared(hidden_states).float()
    for e in range(NUM_EXPERTS):
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        expected.index_add_(
            0,
            token,
            _expert(x[token], w13[e], w2[e], topk_weights[token, slot], fmt, act),
        )

    assert output.dtype == torch.bfloat16
    assert _calc_diff(output, expected) < 1e-3
    assert F.cosine_similarity(output.float(), expected, dim=-1).min() > 0.99
