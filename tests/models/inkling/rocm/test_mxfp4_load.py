# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Quark OCP MXFP4 weights loaded into AITER MoE layouts."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.models.inkling.amd.moe import InklingMoE


def _fake_moe() -> tuple[InklingMoE, SimpleNamespace]:
    moe = InklingMoE.__new__(InklingMoE)
    nn.Module.__init__(moe)
    moe.n_routed_experts = 4
    routed = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_rank=1, tp_size=2)
        ),
        # Logical intermediate per rank is 3; AITER pads it to 4.
        w13_weight=nn.Parameter(torch.zeros(4, 8, 2, dtype=torch.uint8), False),
        w2_weight=nn.Parameter(torch.zeros(4, 5, 3, dtype=torch.uint8), False),
        w13_weight_scale=nn.Parameter(torch.ones(4, 8, 2, dtype=torch.uint8), False),
        w2_weight_scale=nn.Parameter(torch.ones(4, 5, 3, dtype=torch.uint8), False),
    )
    moe.experts = SimpleNamespace(routed_experts=routed)
    moe._local_expert_slots = (  # type: ignore[method-assign]
        lambda: dict(enumerate(range(4)))
    )
    return moe, routed


def test_loads_logical_tp_shards_and_preserves_aiter_padding():
    moe, routed = _fake_moe()

    w13 = torch.arange(4 * 12 * 2, dtype=torch.uint8).view(4, 12, 2)
    w2 = torch.arange(4 * 5 * 4, dtype=torch.uint8).view(4, 5, 4)
    moe.load_expert_weight("experts.w13_weight", w13)
    moe.load_expert_weight("experts.w2_weight", w2)

    # TP rank 1 consumes the second six interleaved gate/up rows. AITER keeps
    # one padding row in each destination half and one padding w2 column.
    torch.testing.assert_close(routed.w13_weight[:, :3], w13[:, 6:12:2])
    torch.testing.assert_close(routed.w13_weight[:, 4:7], w13[:, 7:12:2])
    assert torch.count_nonzero(routed.w13_weight[:, 3]) == 0
    assert torch.count_nonzero(routed.w13_weight[:, 7]) == 0
    torch.testing.assert_close(routed.w2_weight[:, :, :2], w2[:, :, 2:4])
    assert torch.count_nonzero(routed.w2_weight[:, :, 2]) == 0


def test_unflattens_quark_scales_and_preserves_scale_padding():
    moe, routed = _fake_moe()

    flat_w13_scale = torch.arange(4 * 12 * 2, dtype=torch.uint8).view(4 * 12, 2)
    flat_w2_scale = torch.arange(4 * 5 * 4, dtype=torch.uint8).view(4 * 5, 4)
    moe.load_expert_weight("experts.w13_weight_scale", flat_w13_scale)
    moe.load_expert_weight("experts.w2_weight_scale", flat_w2_scale)

    w13_scale = flat_w13_scale.view(4, 12, 2)
    w2_scale = flat_w2_scale.view(4, 5, 4)
    torch.testing.assert_close(routed.w13_weight_scale[:, :3], w13_scale[:, 6:12:2])
    torch.testing.assert_close(routed.w13_weight_scale[:, 4:7], w13_scale[:, 7:12:2])
    assert torch.all(routed.w13_weight_scale[:, 3] == 1)
    assert torch.all(routed.w13_weight_scale[:, 7] == 1)
    torch.testing.assert_close(routed.w2_weight_scale[:, :, :2], w2_scale[:, :, 2:4])
    assert torch.all(routed.w2_weight_scale[:, :, 2] == 1)


def test_direct_reload_restores_aiter_padding(monkeypatch):
    """After a dummy cold start (no loader ran), a direct reload resets the
    AITER padding of a Quark OCP_MX layer (zero weights, 0x01 scales, as
    ``create_weights`` makes them) instead of keeping the dummy bytes."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe import routed_experts
    from vllm.model_executor.layers.quantization.quark import quark_moe
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Static,
    )
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        record_checkpoint_layout,
    )
    from vllm.utils.math_utils import round_up

    monkeypatch.setattr(
        quark_moe,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=None)),
    )
    monkeypatch.setattr(
        quark_moe,
        "select_mxfp4_moe_backend",
        lambda *args, **kwargs: (quark_moe.Mxfp4MoeBackend.NONE, None),
    )
    # AITER rounds the intermediate size up (to 256; 64 here).
    monkeypatch.setattr(
        quark_moe,
        "mxfp4_round_up_hidden_size_and_intermediate_size",
        lambda backend, hidden, intermediate: (hidden, round_up(intermediate, 64)),
    )
    monkeypatch.setattr(
        routed_experts,
        "resolve_quant_method",
        lambda quant_config, layer, prefix: quark_moe.QuarkOCP_MX_MoEMethod(
            layer.moe_config, kMxfp4Static, None
        ),
    )
    num_experts, hidden, intermediate = 4, 64, 32

    def make_moe() -> tuple[InklingMoE, RoutedExperts]:
        config = make_dummy_moe_config(
            num_experts=num_experts,
            hidden_dim=hidden,
            intermediate_size=intermediate,
            experts_per_token=2,
            in_dtype=torch.float32,
        )
        layer = RoutedExperts(
            "model.layers.0.mlp.experts",
            torch.float32,
            config,
            SimpleNamespace(get_name=lambda: "quark"),
            expert_map_manager=SimpleNamespace(
                local_num_experts=num_experts,
                num_fused_shared_experts=0,
                placement_strategy="linear",
                expert_map=None,
                expert_mask=None,
                routing_tables=None,
                map_global_to_local=lambda index: index,
            ),
        )
        assert layer.w13_weight.shape == (num_experts, 128, hidden // 2)
        moe, _ = _fake_moe()
        moe.experts = nn.Module()
        moe.experts.routed_experts = layer
        return moe, layer

    generator = torch.Generator().manual_seed(0)

    def randbytes(*shape):
        return torch.randint(2, 255, shape, generator=generator, dtype=torch.uint8)

    # Quark flattens the block scales over experts.
    checkpoint = {
        "experts.w13_weight": randbytes(num_experts, 2 * intermediate, hidden // 2),
        "experts.w2_weight": randbytes(num_experts, hidden, intermediate // 2),
        "experts.w13_weight_scale": randbytes(num_experts * 2 * intermediate, 2),
        "experts.w2_weight_scale": randbytes(num_experts * hidden, 1),
    }

    def load(moe):
        for name, weight in checkpoint.items():
            moe.load_expert_weight(name, weight)

    expected_moe, expected = make_moe()
    load(expected_moe)
    assert torch.all(expected.w2_weight_scale[:, :, 1] == 1)

    moe, layer = make_moe()
    for p in layer.parameters():
        p.data.fill_(7)  # dummy weights; no loader ran
    record_checkpoint_layout(moe)

    begin_direct_reload(moe)
    load(moe)

    for name, p in layer.named_parameters():
        torch.testing.assert_close(p, getattr(expected, name), rtol=0, atol=0)
