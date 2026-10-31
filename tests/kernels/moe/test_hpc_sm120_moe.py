# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.hpc_moe as hpc_moe
import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle
import vllm.utils.hpc as hpc_utils
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    mxfp4_mxfp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.hpc_moe import (
    HPCExperts,
    HPCMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _backend_activation_key,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
    kMxfp8Dynamic,
)


def _fake_platform(monkeypatch, capability: int) -> None:
    monkeypatch.setattr(hpc_moe.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        hpc_moe.current_platform,
        "is_device_capability",
        lambda requested: requested == capability,
    )
    monkeypatch.setattr(hpc_moe, "has_hpc", lambda: True)
    monkeypatch.setattr(hpc_moe, "has_hpc_sm120_moe", lambda: True)


def _moe_config(**overrides):
    values = dict(
        device="cpu",
        num_local_experts=2,
        num_experts=2,
        experts_per_token=2,
        in_dtype=torch.bfloat16,
        intermediate_size_per_partition=256,
        has_bias=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        moe_parallel_config=SimpleNamespace(
            ep_rank=0,
            ep_size=1,
            tp_rank=0,
            tp_size=1,
            use_ep=False,
            enable_eplb=False,
        ),
    )
    return SimpleNamespace(**(values | overrides))


def _apply(experts, output, hidden_states, w1, w2, a1q_scale=None):
    num_tokens = hidden_states.shape[0]
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# FP8 128-block (HPCExperts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", [89, 90, 100, 120])
def test_hpc_fp8_supported_devices(monkeypatch, capability):
    _fake_platform(monkeypatch, capability)
    assert HPCExperts._supports_current_device() == (capability in (90, 120))

    monkeypatch.setattr(hpc_moe, "has_hpc_sm120_moe", lambda: False)
    assert HPCExperts._supports_current_device() == (capability == 90)


@pytest.mark.parametrize("capability", [90, 120])
def test_hpc_fp8_quant_schemes(monkeypatch, capability):
    _fake_platform(monkeypatch, capability)
    assert HPCExperts._supports_quant_scheme(kFp8Static128BlockSym, kFp8Dynamic128Sym)
    assert HPCExperts._supports_quant_scheme(
        kFp8StaticTensorSym, kFp8StaticTensorSym
    ) == (capability == 90)


def test_hpc_fp8_sm120_rejects_unaligned_tp_split(monkeypatch):
    _fake_platform(monkeypatch, 120)
    monkeypatch.setattr(
        hpc_moe.mk.FusedMoEExperts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )

    def supported_with(intermediate_size_per_partition):
        return HPCExperts.is_supported_config(
            HPCExperts,
            _moe_config(
                intermediate_size_per_partition=intermediate_size_per_partition
            ),
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            hpc_moe.mk.FusedMoEActivationFormat.Standard,
        )[0]

    assert supported_with(256)
    assert not supported_with(192)


def test_hpc_fp8_sm120_pads_unaligned_tp_split(monkeypatch):
    # Qwen3.8-Flash-Next at TP4: 640 // 4 = 160 splits a 128-block.
    _fake_platform(monkeypatch, 120)
    config = SimpleNamespace(
        moe_backend="hpc",
        intermediate_size=640,
        intermediate_size_per_partition=160,
        hidden_dim=2048,
        tp_size=4,
        ep_size=1,
        is_lora_enabled=False,
        has_bias=False,
        tp_shard_with_padding=False,
    )
    block_shape, refine = fp8_oracle.resolve_fp8_moe_weight_block_shape(
        config, [128, 128], kFp8Dynamic128Sym, is_checkpoint_fp8_serialized=True
    )
    assert block_shape == [128, 128]
    assert refine is None
    assert config.tp_shard_with_padding
    assert config.intermediate_size_per_partition == 256


@pytest.mark.parametrize(
    "experts,weight_key,activation_key",
    [
        (HPCExperts, kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (HPCMxfp4Experts, kMxfp4Static, kMxfp8Dynamic),
    ],
)
@pytest.mark.parametrize(
    "use_ep,enable_eplb,placement,supported",
    [
        (True, False, "linear", True),
        (True, True, "linear", False),
        (True, False, "round_robin", False),
        (False, False, "round_robin", True),
    ],
)
def test_hpc_rejects_non_contiguous_expert_placement(
    monkeypatch,
    experts,
    weight_key,
    activation_key,
    use_ep,
    enable_eplb,
    placement,
    supported,
):
    """The kernels derive the local experts from ep_rank alone."""
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    _fake_platform(monkeypatch, 120)
    monkeypatch.setattr(
        hpc_moe.mk.FusedMoEExperts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )
    moe_config = _moe_config(
        moe_parallel_config=SimpleNamespace(
            ep_rank=0,
            ep_size=2,
            tp_rank=0,
            tp_size=1,
            use_ep=use_ep,
            enable_eplb=enable_eplb,
        )
    )
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(expert_placement_strategy=placement)
    )
    with set_current_vllm_config(vllm_config):
        ok, reason = experts.is_supported_config(
            experts,
            moe_config,
            weight_key,
            activation_key,
            hpc_moe.mk.FusedMoEActivationFormat.Standard,
        )
    assert ok == supported, reason


def test_hpc_fp8_sm120_prepares_weights_at_load_time(monkeypatch):
    monkeypatch.setattr(
        fp8_oracle.current_platform,
        "is_device_capability",
        lambda requested: requested == 120,
    )
    calls = []

    def fake_prepare(weight, scale):
        calls.append((weight, scale))
        num_experts, rows, cols = weight.shape
        return weight, torch.zeros(num_experts * rows * cols // 128, dtype=torch.int32)

    monkeypatch.setattr(hpc_utils, "hpc_prepare_fp8_moe_weights_sm120", fake_prepare)

    w13 = torch.empty((2, 512, 256), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((2, 256, 256), dtype=torch.float8_e4m3fn)
    w13_scale = torch.ones((2, 4, 2))
    w2_scale = torch.ones((2, 2, 2))
    layer = SimpleNamespace(weight_block_size=[128, 128])

    out = convert_to_fp8_moe_kernel_format(
        Fp8MoeBackend.HPC, layer, w13, w2, w13_scale, w2_scale, None, None
    )

    assert [c[0] for c in calls] == [w13, w2]
    assert [c[1] for c in calls] == [w13_scale, w2_scale]
    assert out[0] is w13 and out[1] is w2
    assert out[2].dtype == torch.int32 and out[2].shape == (2 * 512 * 2,)
    assert out[3].dtype == torch.int32 and out[3].shape == (2 * 256 * 2,)


def test_hpc_fp8_sm120_passes_bf16_activations(monkeypatch):
    # The kernel quantizes BF16 activations to MXFP8 as it gathers them.
    _fake_platform(monkeypatch, 120)
    w1_scale = torch.empty((2 * 512 * 2,), dtype=torch.int32)
    w2_scale = torch.empty((2 * 256 * 2,), dtype=torch.int32)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
        gemm1_clamp_limit=7.0,
    )
    experts = HPCExperts(_moe_config(), quant_config)
    assert experts.expects_unquantized_inputs

    captured = {}
    monkeypatch.setattr(
        hpc_moe,
        "hpc_fuse_moe_blockwise_fp8_sm120",
        lambda **kwargs: captured.update(kwargs),
    )

    hidden_states = torch.empty((3, 256), dtype=torch.bfloat16)
    w1 = torch.empty((2, 512, 256), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((2, 256, 256), dtype=torch.float8_e4m3fn)
    output = torch.empty((3, 256), dtype=torch.bfloat16)
    topk_weights, topk_ids = _apply(experts, output, hidden_states, w1, w2)

    assert captured["x"] is hidden_states
    assert captured["x_scale"] is None
    assert captured["gate_up_weight"] is w1
    assert captured["gate_up_weight_scale"] is w1_scale
    assert captured["down_weight"] is w2
    assert captured["down_weight_scale"] is w2_scale
    assert captured["topk_ids"] is topk_ids
    assert captured["topk_scale"] is topk_weights
    assert captured["num_expert_total"] == 2
    assert captured["output"] is output
    assert captured["swiglu_limit"] == 7.0


@pytest.mark.parametrize(
    "experts,weight_key,activation_key",
    [
        (HPCExperts, kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (HPCMxfp4Experts, kMxfp4Static, kMxfp8Dynamic),
    ],
)
def test_hpc_sm120_validates_kernel_limits(
    monkeypatch, experts, weight_key, activation_key
):
    _fake_platform(monkeypatch, 120)
    monkeypatch.setattr(
        hpc_moe.mk.FusedMoEExperts,
        "is_supported_config",
        staticmethod(lambda *args: (True, None)),
    )

    def supported_with(**overrides):
        return experts.is_supported_config(
            experts,
            _moe_config(**overrides),
            weight_key,
            activation_key,
            hpc_moe.mk.FusedMoEActivationFormat.Standard,
        )

    assert supported_with() == (True, None)
    assert not supported_with(in_dtype=torch.float16)[0]
    assert not supported_with(num_local_experts=513)[0]
    assert not supported_with(experts_per_token=129)[0]
    assert not supported_with(has_bias=True)[0]
    assert supported_with(swiglu_limit=7.0) == (True, None)
    assert not supported_with(swiglu_limit=-1.0)[0]
    assert not supported_with(swiglu_alpha=1.702)[0]
    assert not supported_with(swiglu_beta=1.0)[0]


# ---------------------------------------------------------------------------
# MXFP4 weights x MXFP8 activations (HPCMxfp4Experts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", [90, 100, 120])
def test_hpc_mxfp4_only_supports_sm120(monkeypatch, capability):
    _fake_platform(monkeypatch, capability)
    assert HPCMxfp4Experts._supports_current_device() == (capability == 120)

    monkeypatch.setattr(hpc_moe, "has_hpc_sm120_moe", lambda: False)
    assert not HPCMxfp4Experts._supports_current_device()


def test_hpc_mxfp4_oracle_wiring():
    assert map_mxfp4_backend("hpc") == [Mxfp4MoeBackend.HPC]
    assert _backend_activation_key(Mxfp4MoeBackend.HPC) == kMxfp8Dynamic
    # gpt-oss at TP4: hidden 2880, intermediate 2880 // 4 = 720.
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.HPC, 2880, 720
    ) == (2944, 768)
    assert HPCMxfp4Experts._supports_quant_scheme(kMxfp4Static, kMxfp8Dynamic)
    assert not HPCMxfp4Experts._supports_quant_scheme(kMxfp4Static, kFp8StaticTensorSym)

    w1_scale = torch.empty((2, 128), dtype=torch.uint8)
    w2_scale = torch.empty((2, 128), dtype=torch.uint8)
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.HPC, w1_scale, w2_scale, swiglu_limit=7.0
    )

    assert quant_config is not None
    assert quant_config.quant_dtype == "mxfp8"
    assert quant_config.weight_quant_dtype == "mxfp4"
    assert not quant_config.is_scale_swizzled
    assert quant_config.w1_scale is w1_scale
    assert quant_config.w2_scale is w2_scale
    assert quant_config.gemm1_clamp_limit == 7.0


def test_hpc_mxfp4_packs_checkpoint_weight_scales(monkeypatch):
    pack_calls = []

    def fake_pack(scale, rows, cols):
        pack_calls.append((scale, rows, cols))
        return scale.reshape(scale.shape[0], -1)

    monkeypatch.setattr(hpc_utils, "hpc_prepare_mxfp4_moe_weights_sm120", fake_pack)
    w1 = torch.empty((2, 256, 64), dtype=torch.uint8)
    w2 = torch.empty((2, 128, 64), dtype=torch.uint8)
    w1_scale = torch.empty((2, 256, 4), dtype=torch.uint8)
    w2_scale = torch.empty((2, 128, 4), dtype=torch.uint8)

    converted = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.HPC, SimpleNamespace(), w1, w2, w1_scale, w2_scale
    )

    assert converted[0].data_ptr() == w1.data_ptr()
    assert converted[1].data_ptr() == w2.data_ptr()
    assert converted[2].shape == (2, 1024)
    assert converted[3].shape == (2, 512)
    assert pack_calls[0][0].data_ptr() == w1_scale.data_ptr()
    assert pack_calls[1][0].data_ptr() == w2_scale.data_ptr()
    assert [call[1:] for call in pack_calls] == [(256, 128), (128, 128)]


def test_hpc_mxfp4_dispatches_to_blockwise_kernel(monkeypatch):
    _fake_platform(monkeypatch, 120)
    w1_scale = torch.empty((2, 128), dtype=torch.uint8)
    w2_scale = torch.empty((2, 128), dtype=torch.uint8)
    quant_config = mxfp4_mxfp8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gemm1_clamp_limit=7.0,
        is_scale_swizzled=False,
    )
    experts = HPCMxfp4Experts(_moe_config(), quant_config)
    # The kernel quantizes the BF16 activations itself.
    assert experts.expects_unquantized_inputs

    captured = {}
    monkeypatch.setattr(
        hpc_moe,
        "hpc_fuse_moe_blockwise_mxfp4_sm120",
        lambda **kwargs: captured.update(kwargs),
    )

    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    w1 = torch.empty((2, 256, 64), dtype=torch.uint8)
    w2 = torch.empty((2, 128, 64), dtype=torch.uint8)
    output = torch.empty((3, 128), dtype=torch.bfloat16)
    topk_weights, topk_ids = _apply(experts, output, hidden_states, w1, w2)

    assert captured["x"] is hidden_states
    assert captured["x_scale"] is None
    assert captured["gate_up_weight"] is w1
    assert captured["gate_up_weight_scale"] is w1_scale
    assert captured["down_weight"] is w2
    assert captured["down_weight_scale"] is w2_scale
    assert captured["topk_ids"] is topk_ids
    assert captured["topk_scale"] is topk_weights
    assert captured["rank_ep"] == 0
    assert captured["num_expert_total"] == 2
    assert captured["output"] is output
    assert captured["swiglu_limit"] == 7.0


@pytest.mark.parametrize("prequantized", [False, True])
def test_hpc_mxfp4_wrappers_call_the_sm120_kernel(monkeypatch, prequantized):
    num_tokens = 3
    calls = []

    def entry(*args, **kwargs):
        calls.append(kwargs)
        return args[0]

    def pack(scale, rows, cols):
        calls.append((rows, cols))
        return scale

    monkeypatch.setitem(
        sys.modules,
        "hpc",
        SimpleNamespace(
            fuse_moe_blockwise_mxfp4_sm120=entry,
            prepare_mxfp4_moe_weights_sm120=pack,
        ),
    )
    gate_scale = torch.empty((256, 1024, 128), dtype=torch.uint8, device="meta")
    prepare = hpc_utils.hpc_prepare_mxfp4_moe_weights_sm120
    assert prepare(gate_scale, 1024, 4096) is gate_scale
    assert calls.pop() == (1024, 4096)

    x = torch.empty((num_tokens, 4096), dtype=torch.bfloat16, device="meta")
    x_scale = (
        torch.empty((num_tokens, 128), dtype=torch.uint8, device="meta")
        if prequantized
        else None
    )
    gate_weight = torch.empty((256, 1024, 2048), dtype=torch.uint8, device="meta")
    down_weight = torch.empty((256, 4096, 256), dtype=torch.uint8, device="meta")
    down_scale = torch.empty((256, 4096, 16), dtype=torch.uint8, device="meta")
    topk_ids = torch.empty((num_tokens, 6), dtype=torch.int32, device="meta")
    topk_scale = torch.empty((num_tokens, 6), dtype=torch.float32, device="meta")
    output = torch.empty_like(x)

    actual = hpc_utils.hpc_fuse_moe_blockwise_mxfp4_sm120(
        x,
        x_scale,
        gate_weight,
        gate_scale,
        down_weight,
        down_scale,
        topk_ids,
        topk_scale,
        rank_ep=0,
        num_expert_total=2,
        output=output,
        swiglu_limit=7.0,
    )
    assert actual is x
    assert calls == [
        {"rank_ep": 0, "num_expert_total": 2, "output": output, "swiglu_limit": 7.0}
    ]


def test_hpc_fp8_wrappers_call_the_sm120_kernel(monkeypatch):
    calls = []

    def entry(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0]

    compact = torch.empty((2 * 512 * 2,), dtype=torch.int32, device="meta")
    monkeypatch.setitem(
        sys.modules,
        "hpc",
        SimpleNamespace(
            fuse_moe_blockwise_fp8_sm120=entry,
            prepare_fp8_moe_weights_sm120=lambda weight, scale: compact,
        ),
    )
    weight = torch.empty((2, 512, 256), dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.empty((2, 4, 2), dtype=torch.float32, device="meta")
    prepared_weight, prepared_scale = hpc_utils.hpc_prepare_fp8_moe_weights_sm120(
        weight, scale
    )
    assert prepared_weight is weight and prepared_scale is compact

    x = torch.empty((3, 256), dtype=torch.bfloat16, device="meta")
    output = torch.empty_like(x)
    actual = hpc_utils.hpc_fuse_moe_blockwise_fp8_sm120(
        x,
        None,
        weight,
        compact,
        weight,
        compact,
        torch.empty((3, 2), dtype=torch.int32, device="meta"),
        torch.empty((3, 2), dtype=torch.float32, device="meta"),
        rank_ep=0,
        num_expert_total=2,
        output=output,
        swiglu_limit=7.0,
    )
    assert actual is x
    ((args, kwargs),) = calls
    assert args[1] is None
    assert kwargs["output"] is output
    assert kwargs["swiglu_limit"] == 7.0


# ---------------------------------------------------------------------------
# One vLLM FusedMoE layer on an SM120 GPU
# ---------------------------------------------------------------------------


def _random_checkpoint_tensor(param, name, gen):
    if param.dtype == torch.float8_e4m3fn:
        values = torch.randn(param.shape, generator=gen, device="cuda") * 0.5
        return values.to(param.dtype)
    if param.dtype == torch.uint8 and "scale" in name:
        # UE8M0 scales between 2^-7 and 2^-3.
        return torch.randint(
            120, 125, param.shape, generator=gen, device="cuda", dtype=torch.uint8
        )
    if param.dtype == torch.uint8:
        return torch.randint(
            0, 256, param.shape, generator=gen, device="cuda", dtype=torch.uint8
        )
    if param.dtype == torch.float32 and "scale" in name:
        return torch.rand(param.shape, generator=gen, device="cuda") * 0.01 + 0.005
    return torch.randn(param.shape, generator=gen, device="cuda").to(param.dtype) * 0.01


def _run_vllm_layer(quant, backend, shape, checkpoint, inputs):
    """Build a single-rank FusedMoE layer for `backend`, load `checkpoint`
    (filled with random tensors on first use), and apply it to `inputs`."""
    from vllm.config import (
        KernelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory

    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        scheduler_config=SchedulerConfig.default_factory(max_num_batched_tokens=2048),
        kernel_config=KernelConfig(moe_backend=backend),
    )
    with set_current_vllm_config(config):
        if quant == "fp8_block":
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config

            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
        else:
            from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

            quant_config = Mxfp4Config()
        torch.set_default_device("cuda")
        try:
            layer = FusedMoEFactory(
                num_experts=shape["experts"],
                top_k=shape["topk"],
                hidden_size=shape["hidden"],
                intermediate_size=shape["intermediate"],
                params_dtype=torch.bfloat16,
                renormalize=False,
                quant_config=quant_config,
                tp_size=1,
                dp_size=1,
                pcp_size=1,
                prefix="from_forward_context",
                activation="silu",
                swiglu_limit=shape["swiglu_limit"],
            )
        finally:
            torch.set_default_device(None)
        experts = layer.routed_experts
        gen = torch.Generator(device="cuda").manual_seed(0)
        for name, param in sorted(experts.named_parameters()):
            if name not in checkpoint:
                checkpoint[name] = _random_checkpoint_tensor(param, name, gen)
            param.data.copy_(checkpoint[name])
        quant_method = layer._quant_method
        quant_method.process_weights_after_loading(experts)

        outputs = []
        for x, topk_weights, topk_ids in inputs:
            with set_forward_context(None, config, num_tokens=x.shape[0]):
                out = quant_method.apply(experts, x, topk_weights, topk_ids, None, None)
            assert isinstance(out, torch.Tensor)
            outputs.append(out.float())
        torch.accelerator.synchronize()
    return outputs


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hpc_moe.current_platform.is_device_capability(120)
    or not hpc_utils.has_hpc_sm120_moe(),
    reason="needs an SM120 GPU and hpc-ops with the SM120 fused MoE",
)
@pytest.mark.parametrize(
    "quant,reference,entry,swiglu_limit",
    [
        ("fp8_block", "triton", "hpc_fuse_moe_blockwise_fp8_sm120", None),
        # Small enough to clamp the gate and up projections of this layer.
        ("fp8_block", "triton", "hpc_fuse_moe_blockwise_fp8_sm120", 0.1),
        ("mxfp4", "marlin", "hpc_fuse_moe_blockwise_mxfp4_sm120", 10.0),
    ],
)
def test_hpc_sm120_layer_matches_reference_backend(
    monkeypatch, dist_init, workspace_init, quant, reference, entry, swiglu_limit
):
    shape = dict(
        experts=16, topk=8, hidden=1024, intermediate=512, swiglu_limit=swiglu_limit
    )
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = []
    # One token, small batches routed in one block, and a multi-block batch.
    for num_tokens in (1, 8, 64, 129, 512):
        x = torch.randn(
            num_tokens, shape["hidden"], generator=gen, device="cuda"
        ).bfloat16()
        topk_ids = (
            torch.rand(num_tokens, shape["experts"], generator=gen, device="cuda")
            .topk(shape["topk"], dim=-1)
            .indices.to(torch.int32)
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, shape["topk"], generator=gen, device="cuda"), -1
        )
        inputs.append((x, topk_weights, topk_ids))

    calls = 0
    hpc_entry = getattr(hpc_moe, entry)

    def counted_entry(**kwargs):
        nonlocal calls
        calls += 1
        return hpc_entry(**kwargs)

    monkeypatch.setattr(hpc_moe, entry, counted_entry)
    checkpoint: dict[str, torch.Tensor] = {}
    actual = _run_vllm_layer(quant, "hpc", shape, checkpoint, inputs)
    assert calls == len(inputs)
    expected = _run_vllm_layer(quant, reference, shape, checkpoint, inputs)

    # HPC quantizes the activations and the SwiGLU output to MXFP8, and
    # requantizes FP8 weights against power-of-two scales, where the reference
    # backends keep more precision; that alone puts them 5-7% apart.
    for (x, _, _), a, e in zip(inputs, actual, expected):
        assert torch.isfinite(a).all()
        rel = ((a - e).norm() / e.norm()).item()
        assert rel < 0.08, f"{quant} tokens={x.shape[0]}: relative L2 error {rel:.4f}"
