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
    MEGA_MOE_NVFP4,
    MEGA_MOE_WEIGHT_FORMATS,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_megamoe_inputs,
    prepare_megamoe_nvfp4_inputs,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kNvfp4DynamicToken,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepGEMM MegaMoE requires CUDA"
)
requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MegaMoE input staging only feeds the SM100 kernels",
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
    kwargs.setdefault("weight_format", MEGA_MOE_MXFP4)
    return DeepGemmMegaMoEExperts(vllm_config, **kwargs)


def test_deep_gemm_mega_moe_capture_precedes_eplb(monkeypatch):
    experts = DeepGemmMegaMoEExperts.__new__(DeepGemmMegaMoEExperts)
    torch.nn.Module.__init__(experts)
    monkeypatch.setattr(experts, "_synchronize_first_launch", lambda: None)
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


def _make_mega_moe_experts(
    num_local_experts, num_logical_experts, weight_format=MEGA_MOE_MXFP4
):
    return _make_experts(
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=0,
        top_k=1,
        hidden_size=256,
        intermediate_size=384,
        num_logical_experts=num_logical_experts,
        weight_format=weight_format,
    )


def _load_mega_moe_checkpoint_shuffled(experts, num_logical_experts, seed):
    hidden, inter = experts.hidden_size, experts.intermediate_size
    routed = experts.weight_format.routed
    group = routed.scale_group_k
    shapes = {
        "w13_weight": (num_logical_experts, 2 * inter, hidden // 2),
        "w13_weight_scale": (num_logical_experts, 2 * inter, hidden // group),
        "w2_weight": (num_logical_experts, hidden, inter // 2),
        "w2_weight_scale": (num_logical_experts, hidden, inter // group),
    }
    generator = torch.Generator().manual_seed(seed)
    ckpt = {
        name: torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        for name, shape in shapes.items()
    }
    if routed.has_global_scale:
        for name in ("w13_weight_scale_2", "w2_weight_scale_2", "w2_input_scale"):
            ckpt[name] = torch.rand(num_logical_experts, generator=generator) + 0.5
    shards = []
    for expert_id in range(num_logical_experts):
        for name, tensor in ckpt.items():
            if name.startswith("w2_"):
                shards.append((name, tensor[expert_id], "w2", expert_id))
            elif tensor.dim() == 1:
                # w1 and w3 share one global scale.
                shards.append((name, tensor[expert_id], "w1", expert_id))
                shards.append((name, tensor[expert_id], "w3", expert_id))
            else:
                shards.append((name, tensor[expert_id, :inter], "w1", expert_id))
                shards.append((name, tensor[expert_id, inter:], "w3", expert_id))
    for i in torch.randperm(len(shards), generator=generator).tolist():
        name, loaded_weight, shard_id, expert_id = shards[i]
        if name.endswith("weight_scale"):
            loaded_weight = loaded_weight.view(routed.scale_dtype)
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
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    ckpt = {name: ckpt[name][physical] for name in names}
    expected = _reference_mega_moe_layout(*ckpt.values())
    if is_deep_gemm_supported():
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        w13, s13, w2, s2 = (t.cuda() for t in ckpt.values())
        hidden, inter = w2.shape[1], w2.shape[2] * 2
        if s13.shape[-1] * 16 == hidden:
            # NVFP4: E4M3 bytes packed four per int32, MN-major.
            scales = [
                sf.view(torch.int32).transpose(1, 2).contiguous().transpose(1, 2)
                for sf in (s13, s2)
            ]
        else:
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


@pytest.mark.parametrize("weight_format", [MEGA_MOE_MXFP4, MEGA_MOE_NVFP4])
def test_deep_gemm_mega_moe_loads_shards_into_kernel_layout(weight_format):
    """Shards in any order, including EPLB replicas, land in DeepGEMM's layout."""
    experts = _make_mega_moe_experts(4, 3, weight_format)
    ckpt = _load_mega_moe_checkpoint_shuffled(experts, 3, seed=0)
    _assert_mega_moe_layout(experts, _expected_mega_moe_layout(ckpt, 4, 3))
    # DeepGEMM's MN-major scale strides.
    group_k = 4 * weight_format.routed.scale_group_k
    assert experts.w13_weight_scale.stride() == (768 * 256 // group_k, 1, 768)
    assert experts.w2_weight_scale.stride() == (256 * 384 // group_k, 1, 256)
    if weight_format.routed.has_global_scale:
        physical = torch.arange(4) % 3
        for name in ("w13_weight_scale_2", "w2_weight_scale_2", "w2_input_scale"):
            assert torch.equal(getattr(experts, name), ckpt[name][physical])
        assert experts.l2_activation_scale == ckpt["w2_input_scale"].max().item()


@pytest.mark.parametrize("weight_format", [MEGA_MOE_MXFP4, MEGA_MOE_NVFP4])
def test_deep_gemm_mega_moe_reload_overwrites_kernel_layout_in_place(weight_format):
    """A second load must update the live kernel tensors without reallocating."""
    experts = _make_mega_moe_experts(2, 2, weight_format)
    _load_mega_moe_checkpoint_shuffled(experts, 2, seed=0)
    data_ptrs = [param.data_ptr() for param in experts.parameters()]
    ckpt = _load_mega_moe_checkpoint_shuffled(experts, 2, seed=1)
    assert [param.data_ptr() for param in experts.parameters()] == data_ptrs
    _assert_mega_moe_layout(experts, _expected_mega_moe_layout(ckpt, 2, 2))


def test_deep_gemm_mega_moe_nvfp4_l2_scale_covers_every_expert():
    """Every rank quantizes L2 inputs with the max input scale over all experts,
    including those it does not own, so EPLB can move experts between ranks."""
    experts = _make_experts(
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        weight_format=MEGA_MOE_NVFP4,
    )
    assert experts.l2_activation_scale == 1.0
    for expert_id, scale in enumerate((0.5, 4.0, 1.0, 2.0)):
        loaded = experts.weight_loader(
            experts.w2_input_scale,
            torch.tensor(scale),
            "experts.w2_input_scale",
            shard_id="w2",
            expert_id=expert_id,
            return_success=True,
        )
        assert loaded == (expert_id >= 2)
    assert experts.l2_activation_scale == 4.0
    assert experts.w2_input_scale.tolist() == [1.0, 2.0]
    # A reload replaces the expert's scale rather than keeping the old max.
    experts.weight_loader(
        experts.w2_input_scale,
        torch.tensor(0.25),
        "experts.w2_input_scale",
        shard_id="w2",
        expert_id=1,
    )
    assert experts.l2_activation_scale == 2.0
    # Global scales are expert weights that EPLB moves.
    assert len(experts.get_expert_weights()) == 7


def test_deep_gemm_mega_moe_nvfp4_rejects_mismatched_l1_scales():
    """The kernel takes one L1 alpha per expert, so w1 and w3 must agree."""
    experts = _make_mega_moe_experts(1, 1, MEGA_MOE_NVFP4)

    def load(shard_id, scale):
        experts.weight_loader(
            experts.w13_weight_scale_2,
            torch.tensor(scale),
            "experts.w13_weight_scale_2",
            shard_id=shard_id,
            expert_id=0,
        )

    load("w1", 0.5)
    load("w3", 0.5)
    # A reload may change the scale while the old pair is still live.
    load("w3", 2.0)
    load("w1", 2.0)
    load("w1", 1.0)
    with pytest.raises(ValueError, match="one L1 scale per expert"):
        load("w3", 4.0)


def test_deep_gemm_mega_moe_nvfp4_meta_load_skips_host_scales():
    """Reload dry runs load meta stand-ins, which must not reach host state."""
    experts = _make_mega_moe_experts(1, 1, MEGA_MOE_NVFP4)
    for name, shard_id in (
        ("w2_input_scale", "w2"),
        ("w13_weight_scale_2", "w1"),
        ("w13_weight_scale_2", "w3"),
    ):
        assert experts.weight_loader(
            getattr(experts, name).to("meta"),
            torch.empty((), device="meta"),
            f"experts.{name}",
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )
    assert experts.l2_activation_scale == 1.0


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
        DeepGemmMegaMoEExperts, "_kernel_fuses_shared_experts", lambda self, spec: True
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

    def new_buffer(group, num_shared_experts=0, shared_dtype=None): ...

    def new_kernel(y, shared_l1_weights=None, shared_l2_weights=None): ...

    kernel = new_kernel if with_shared else old_kernel
    buffer = new_buffer if with_shared else old_buffer
    return SimpleNamespace(
        get_symm_buffer_for_mega_moe=buffer,
        NVFP4SymmBuffer=buffer,
        fp8_fp4_mega_moe=kernel,
        bf16_mega_moe=kernel,
        nvfp4_mega_moe=kernel,
        get_block_m_for_mega_moe=lambda *args: 0,
        get_block_m_for_nvfp4_mega_moe=lambda *args: 0,
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
        (MEGA_MOE_NVFP4, torch.float8_e4m3fn, torch.uint8, (1, 32), 256, True, True),
        (MEGA_MOE_NVFP4, torch.bfloat16, None, None, 128, True, True),
        (MEGA_MOE_NVFP4, torch.bfloat16, None, None, 128, False, False),
        # NVFP4-style E4M3 block scales are not MXFP8.
        (
            MEGA_MOE_NVFP4,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            (1, 16),
            128,
            True,
            False,
        ),
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
    if fused:
        assert experts.shared_spec in weight_format.shared
        assert experts.shared_l1_weight.dtype == weight_dtype


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
        DeepGemmMegaMoEExperts, "_kernel_fuses_shared_experts", lambda self, spec: True
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
    monkeypatch.setattr(experts, "get_symm_buffer", lambda: symm_buffer)
    monkeypatch.setattr(experts, "_synchronize_first_launch", lambda: None)

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


@pytest.mark.parametrize("hidden_staged", [False, True])
@pytest.mark.parametrize("routing_staged", [False, True])
def test_deep_gemm_mega_moe_stages_only_inputs_not_in_buffer(
    monkeypatch, hidden_staged, routing_staged
):
    """Routing already in the buffer's top-k slices and activations a producer
    wrote are not staged again."""
    staging = []
    module = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
    monkeypatch.setattr(
        f"{module}.prepare_megamoe_inputs", lambda *args, **kwargs: staging.append(args)
    )
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(
            fp8_fp4_mega_moe=lambda y, *args, **kwargs: y.zero_(),
            get_block_m_for_mega_moe=lambda *args: 8,
        ),
    )
    monkeypatch.setattr(f"{module}.get_ep_group", lambda: SimpleNamespace(world_size=1))
    experts = _make_experts(
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
    )
    symm_buffer = SimpleNamespace(
        x=torch.empty(4, 128, dtype=torch.float8_e4m3fn),
        x_sf=torch.empty(4, 1, dtype=torch.int32),
        topk_idx=torch.empty(4, 1, dtype=torch.int64),
        topk_weights=torch.empty(4, 1),
        num_max_tokens_per_rank=4,
    )
    experts.get_symm_buffer = lambda: symm_buffer
    experts._synchronize_first_launch = lambda: None
    inputs = experts.input_buffers(2)
    assert inputs is not None and inputs.x_sf is not None
    routing = (
        (inputs.topk_weights, inputs.topk_idx)
        if routing_staged
        else (torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64))
    )

    experts(
        torch.zeros(2, 128, dtype=torch.bfloat16),
        *routing,
        hidden_states_staged=hidden_staged,
    )

    ((hidden, topk_weights, topk_ids, *_),) = staging
    assert (hidden is None) == hidden_staged
    assert (topk_ids is None) == (topk_weights is None) == routing_staged


@requires_sm100
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


@requires_sm100
@pytest.mark.parametrize("staged", ["hidden", "routing"])
def test_prepare_megamoe_inputs_stages_only_the_missing_half(staged):
    """Inputs a producer already wrote are left alone; the rest is staged
    exactly as the full staging writes it."""
    device = torch.device("cuda")
    num_tokens, hidden_size, top_k = 65, 256, 8
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.randint(0, 256, (num_tokens, top_k), device=device)
    topk_weights = torch.rand(num_tokens, top_k, device=device)

    def buffers():
        return (
            torch.full((num_tokens, hidden_size), 3.0, device=device).to(
                torch.float8_e4m3fn
            ),
            torch.full((num_tokens, hidden_size // 128), 7, device=device).int(),
            torch.full((num_tokens, top_k), 5, device=device, dtype=torch.int64),
            torch.full((num_tokens, top_k), 0.5, device=device),
        )

    full = buffers()
    prepare_megamoe_inputs(hidden_states, topk_weights, topk_ids, *full)
    partial = buffers()
    untouched = buffers()
    if staged == "hidden":
        prepare_megamoe_inputs(None, topk_weights, topk_ids, *partial)
        expected = (*untouched[:2], *full[2:])
    else:
        prepare_megamoe_inputs(hidden_states, None, None, *partial)
        expected = (*full[:2], *untouched[2:])
    for actual, reference in zip(partial, expected, strict=True):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def _shared_sf_tma_layout(sf: torch.Tensor, block_m: int) -> torch.Tensor:
    """Scatter per-token scales into DeepGEMM's padded MN-major UTCCP pages."""
    aligned_block_m = (block_m + 127) // 128 * 128
    num_rows = (sf.shape[0] + block_m - 1) // block_m * aligned_block_m
    m = torch.arange(sf.shape[0], device=sf.device)
    local = m % block_m
    rows = (
        m // block_m * aligned_block_m
        + local // 128 * 128
        + local % 32 * 4
        + local % 128 // 32
    )
    out = sf.new_zeros(num_rows, sf.shape[1])
    out[rows] = sf
    return out


@requires_sm100
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
    ref_shared_x_sf = _shared_sf_tma_layout(ref_x_sf, shared_block_m)
    num_shared_rows = ref_shared_x_sf.shape[0]

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


@requires_sm100
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


@requires_sm100
@pytest.mark.parametrize("shared", [None, "mxfp8", "bf16"])
@pytest.mark.parametrize("num_tokens", [0, 7, 65, 135])
def test_prepare_megamoe_nvfp4_inputs_is_bitwise_exact(num_tokens, shared):
    """Per-token NVFP4 dispatch, shared-expert inputs, unrouted padding and the
    L2 alphas match DeepGEMM's reference quantizers bit for bit."""
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8
    from vllm.third_party.deep_gemm.utils.nvfp4 import per_token_cast_to_nvfp4

    device = torch.device("cuda")
    hidden_size, top_k, shared_block_m = 5120, 8, 32
    generator = torch.Generator(device=device).manual_seed(num_tokens)
    # Rows span four decades, which one tensor-wide scale cannot cover.
    row_scale = torch.logspace(-3, 1, max(num_tokens, 1), device=device)
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, generator=generator)
        * row_scale[:num_tokens, None]
    ).bfloat16()
    if num_tokens:
        hidden_states[0] = 0
    topk_ids = torch.randint(
        0, 256, (num_tokens, top_k), device=device, generator=generator
    ).int()
    topk_weights = torch.rand(num_tokens, top_k, device=device, generator=generator)
    is_padding = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    is_padding[1::3] = True
    w2_global_scale = torch.rand(6, device=device, generator=generator)

    x = torch.empty(num_tokens, hidden_size // 2, dtype=torch.int8, device=device)
    x_sf = torch.empty(num_tokens, hidden_size // 64, dtype=torch.int32, device=device)
    x_scales = torch.empty(num_tokens, device=device)
    topk_idx_out = torch.empty(num_tokens, top_k, dtype=torch.int64, device=device)
    topk_weights_out = torch.empty_like(topk_weights)
    l2_alpha = torch.empty_like(w2_global_scale)
    shared_x = shared_x_sf = None
    if shared is not None:
        dtype = torch.float8_e4m3fn if shared == "mxfp8" else torch.bfloat16
        shared_x = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
    if shared == "mxfp8":
        ref_shared_x, ref_shared_x_sf = per_token_cast_to_fp8(
            hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        ref_shared_x_sf = _shared_sf_tma_layout(ref_shared_x_sf, shared_block_m)
        shared_x_sf = torch.full_like(ref_shared_x_sf.t(), -1).t()

    prepare_megamoe_nvfp4_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        x,
        x_sf,
        x_scales,
        topk_idx_out,
        topk_weights_out,
        is_padding=is_padding,
        shared_x=shared_x,
        shared_x_sf=shared_x_sf,
        shared_block_m=shared_block_m if shared == "mxfp8" else None,
        w2_global_scale=w2_global_scale,
        l2_alpha=l2_alpha,
        l2_activation_scale=0.37,
    )
    torch.accelerator.synchronize()

    # Written even without local tokens: other ranks route to these experts.
    assert torch.equal(l2_alpha, w2_global_scale * 0.37)
    ref_x_scales = hidden_states.float().abs().amax(-1).clamp_min(1e-12) * (1 / 2688)
    ref_x, ref_x_sf = per_token_cast_to_nvfp4(
        hidden_states.float() / ref_x_scales[:, None]
    )
    assert torch.equal(x_scales, ref_x_scales)
    assert torch.equal(x, ref_x)
    assert torch.equal(x_sf, ref_x_sf)
    ref_topk_idx = topk_ids.long().masked_fill(is_padding[:, None], -1)
    assert torch.equal(topk_idx_out, ref_topk_idx)
    ref_weights = topk_weights.masked_fill(is_padding[:, None], 0.0)
    assert torch.equal(topk_weights_out, ref_weights)
    if shared == "bf16":
        assert torch.equal(shared_x, hidden_states)
    elif shared == "mxfp8":
        assert shared_x is not None and shared_x_sf is not None
        assert torch.equal(shared_x.view(torch.uint8), ref_shared_x.view(torch.uint8))
        populated = ref_shared_x_sf != 0
        assert torch.equal(shared_x_sf[populated], ref_shared_x_sf[populated])
        assert torch.all(shared_x_sf[~populated] == -1)


NUM_EXPERTS, TOP_K, HIDDEN, INTER, NUM_TOKENS = 8, 2, 512, 256, 48

ACTIVATIONS = {
    "swiglu": MegaMoEActivation(),
    "swiglu_clamp": MegaMoEActivation(clamp=0.5),
    "swiglu_alpha_beta": MegaMoEActivation(alpha=1.702, beta=1.0),
    "situ": MegaMoEActivation("situ", alpha=2.0, beta=3.0),
}


def _quantize(w: torch.Tensor, spec: MegaMoEWeightSpec):
    """Return checkpoint ``(weight, scale, global_scale)`` and the dequantized
    weight; NVFP4 takes ``global_scale`` as ``weight_scale_2``."""
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )
    from vllm.third_party.deep_gemm.utils.nvfp4 import (
        cast_back_from_nvfp4,
        per_token_cast_to_nvfp4,
    )

    if spec.scale_group_k is None:
        return w, None, None, w.float()
    if spec.has_global_scale:
        global_scale = w.float().abs().amax().item() / 2688
        q, sf = per_token_cast_to_nvfp4(w, global_scale)
        deq = cast_back_from_nvfp4(q, sf, global_scale)
        return q, sf.view(torch.float8_e4m3fn), torch.tensor(global_scale), deq
    if spec.values_per_element == 2:
        q, sf = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
        deq = cast_back_from_fp4(q, sf, gran_k=32)
    else:
        q, sf = per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
        deq = cast_back_from_fp8(q, sf, gran_k=32)
    return q, (sf.view(torch.int32) >> 23).to(torch.uint8), None, deq


def _round_trip(x: torch.Tensor, quant, nvfp4_scale: float | None = None):
    """What the kernel computes on: x after ``quant`` (None keeps BF16)."""
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp8,
        per_token_cast_to_fp8,
    )
    from vllm.third_party.deep_gemm.utils.nvfp4 import (
        cast_back_from_nvfp4,
        per_token_cast_to_nvfp4,
    )

    if quant is None:
        return x.bfloat16().float()
    if quant == kNvfp4DynamicToken:
        if nvfp4_scale is not None:
            q = per_token_cast_to_nvfp4(x, nvfp4_scale)
            return cast_back_from_nvfp4(*q, nvfp4_scale)
        x = x.bfloat16().float()
        scales = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 2688
        return cast_back_from_nvfp4(*per_token_cast_to_nvfp4(x / scales)) * scales
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


def _expert(x, w13, w2, weight, fmt, act, l2_scale):
    """Reference routed expert. MX and BF16 weight the activation before the
    L2 dispatch; NVFP4 quantizes the unweighted activation with the global
    ``l2_scale`` and weights the BF16 output."""
    a = _activate(x @ w13.t(), act)
    if fmt.hidden_quant == kNvfp4DynamicToken:
        a = _round_trip(a, fmt.hidden_quant, l2_scale)
        return (a @ w2.t()).bfloat16().float() * weight[:, None]
    return _round_trip(a * weight[:, None], fmt.hidden_quant) @ w2.t()


class _ReferenceSharedExpert(torch.nn.Module):
    def __init__(self, w13, w2, spec: MegaMoEWeightSpec, act):
        super().__init__()
        self.w13, self.w2, self.act = w13, w2, act
        self.quant = kMxfp8Dynamic if spec.scale_group_k else None

    def forward(self, x):
        a = _activate(_round_trip(x, self.quant) @ self.w13.t(), self.act)
        return (_round_trip(a, self.quant) @ self.w2.t()).bfloat16()


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)
@pytest.mark.parametrize(
    "fmt_name,shared,act_name",
    [
        pytest.param(
            name,
            shared,
            act_name,
            id=f"{act_name}-{name}-{str(shared).removeprefix('torch.')}",
        )
        for act_name, act in ACTIVATIONS.items()
        for name, fmt in MEGA_MOE_WEIGHT_FORMATS.items()
        if act.name in fmt.activations
        for shared in (None, *(spec.dtype for spec in fmt.shared))
    ],
)
@pytest.mark.parametrize("staged", [False, True])
def test_deep_gemm_mega_moe_matches_reference(
    dist_init, fmt_name, shared, act_name, staged
):
    """``shared`` is the fused shared-expert dtype; None runs it unfused.
    ``staged`` writes the inputs into the buffer the way producers do."""
    fmt, act = MEGA_MOE_WEIGHT_FORMATS[fmt_name], ACTIVATIONS[act_name]
    if staged and fmt.hidden_quant != kMxfp8Dynamic:
        pytest.skip("only MXFP8 activations are written by producers")
    shared_spec = next(
        (spec for spec in fmt.shared if spec.dtype == shared), fmt.shared[0]
    )
    fuse_shared = shared is not None
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    def randn(*shape, fan_in):
        return (torch.randn(*shape, device=device) / fan_in**0.5).bfloat16()

    shared_ckpt = [
        _quantize(randn(*shape, fan_in=k), shared_spec)
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

    with torch.device(device):
        experts = DeepGemmMegaMoEExperts(
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
            prefix=f"model.layers.0.{fmt_name}.{act_name}.{shared}",
        )
    assert experts.has_fused_shared_experts == fuse_shared
    assert experts.shared_spec == (shared_spec if fuse_shared else None)

    w13 = torch.empty(NUM_EXPERTS, 2 * INTER, HIDDEN, device=device)
    w2 = torch.empty(NUM_EXPERTS, HIDDEN, INTER, device=device)
    # NVFP4 L2 input scales; the kernel uses their max.
    l2_scales = [2e-3 * (e + 1) for e in range(NUM_EXPERTS)]
    for e in range(NUM_EXPERTS):
        # w1 and w3 are quantized together: NVFP4 needs one global scale.
        w13_ckpt = _quantize(randn(2 * INTER, HIDDEN, fan_in=HIDDEN), fmt.routed)
        w2_ckpt = _quantize(randn(HIDDEN, INTER, fan_in=INTER), fmt.routed)
        w13[e], w2[e] = w13_ckpt[3], w2_ckpt[3]
        for shard_id, param, rows, (weight, scale, global_scale, _) in (
            ("w1", "w13", slice(0, INTER), w13_ckpt),
            ("w3", "w13", slice(INTER, None), w13_ckpt),
            ("w2", "w2", slice(None), w2_ckpt),
        ):
            tensors = {
                "weight": weight[rows],
                "weight_scale": None if scale is None else scale[rows],
                "weight_scale_2": global_scale,
            }
            if shard_id == "w2" and global_scale is not None:
                tensors["input_scale"] = torch.tensor(l2_scales[e])
            for suffix, tensor in tensors.items():
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

    shared_w13 = torch.cat([shared_ckpt[0][3], shared_ckpt[1][3]])
    reference_shared = _ReferenceSharedExpert(
        shared_w13, shared_ckpt[2][3], shared_spec, act
    )
    if fuse_shared:
        for (weight, scale, _, _), shard in zip(shared_ckpt, ("w1", "w3", "w2")):
            experts.load_shared_expert_weight(f"shared_experts.{shard}.weight", weight)
            if scale is not None:
                experts.load_shared_expert_weight(
                    f"shared_experts.{shard}.weight_scale", scale
                )

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN, device=device).bfloat16()
    scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=device)
    topk_weights, topk_ids = torch.topk(scores.softmax(-1), TOP_K, dim=-1)

    unfused_shared = None if fuse_shared else reference_shared
    if staged:
        inputs = experts.input_buffers(NUM_TOKENS)
        assert inputs is not None
        prepare_megamoe_inputs(
            hidden_states,
            None,
            None,
            inputs.x,
            inputs.x_sf,
            inputs.topk_idx,
            inputs.topk_weights,
            shared_x_sf=inputs.shared_x_sf,
            shared_block_m=inputs.shared_block_m,
        )
        inputs.topk_idx.copy_(topk_ids)
        inputs.topk_weights.copy_(topk_weights)
        output = experts(
            hidden_states,
            inputs.topk_weights,
            inputs.topk_idx,
            unfused_shared,
            hidden_states_staged=True,
        )
    else:
        output = experts(hidden_states, topk_weights, topk_ids, unfused_shared)

    x = _round_trip(hidden_states.float(), fmt.hidden_quant)
    l2_scale = experts.l2_activation_scale
    expected = reference_shared(hidden_states).float()
    for e in range(NUM_EXPERTS):
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        expected.index_add_(
            0,
            token,
            _expert(
                x[token], w13[e], w2[e], topk_weights[token, slot], fmt, act, l2_scale
            ),
        )

    assert output.dtype == torch.bfloat16
    assert _calc_diff(output, expected) < 1e-3
    assert F.cosine_similarity(output.float(), expected, dim=-1).min() > 0.99
