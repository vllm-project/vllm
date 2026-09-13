# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import fp8 as fp8_module
from vllm.model_executor.layers.quantization.fp8 import Fp8Config


def _make_fp8_tp_experts(
    monkeypatch,
    tp_size,
    tp_rank,
    backend="flashinfer_trtllm",
    *,
    mock_backend=True,
    num_experts=2,
    hidden_dim=256,
    intermediate_size=640,
    quant_config=None,
    config_overrides=None,
):
    # Exercise allocation and the public weight loader without selecting a GPU kernel.
    if mock_backend:
        monkeypatch.setattr(
            fp8_module, "select_fp8_moe_backend", lambda **kwargs: (None, None)
        )
    monkeypatch.setattr(
        fp8_module, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    config = make_dummy_moe_config(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        experts_per_token=min(num_experts, 10),
    )
    config.routing_method = RoutingMethodType.RenormalizeNaive
    config.moe_parallel_config.tp_size = tp_size
    config.moe_parallel_config.tp_rank = tp_rank
    config.intermediate_size_per_partition = intermediate_size // tp_size
    config.intermediate_size_per_partition_unpadded = intermediate_size // tp_size
    config.moe_backend = backend
    for field, value in (config_overrides or {}).items():
        setattr(config, field, value)
    return RoutedExperts(
        "model.layers.0.mlp.experts",
        torch.bfloat16,
        config,
        quant_config
        or Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]),
        expert_map_manager=SimpleNamespace(
            local_num_experts=num_experts,
            placement_strategy="linear",
            expert_map=None,
            expert_mask=None,
            routing_tables=None,
            map_global_to_local=lambda index: index,
        ),
    )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("intermediate_size", [640, 896])
def test_fp8_block_aligned_tp_preserves_checkpoint(
    monkeypatch, tp_size, batched, intermediate_size
):
    """All ranks reconstruct the original dequantized projections, including
    padding-only ranks. Reloading must clear stale weights and scales.
    """
    generator = torch.Generator().manual_seed(42)
    weights = {
        "w1": torch.randn(2, intermediate_size, 256, generator=generator).to(
            torch.float8_e4m3fn
        ),
        "w3": torch.randn(2, intermediate_size, 256, generator=generator).to(
            torch.float8_e4m3fn
        ),
        "w2": torch.randn(2, 256, intermediate_size, generator=generator).to(
            torch.float8_e4m3fn
        ),
    }
    scales = {
        name: torch.rand(2, w.shape[1] // 128, w.shape[2] // 128, generator=generator)
        + 0.1
        for name, w in weights.items()
    }
    reconstructed: dict[str, list[torch.Tensor]] = {name: [] for name in weights}
    num_blocks = intermediate_size // 128
    for rank in range(tp_size):
        layer = _make_fp8_tp_experts(
            monkeypatch, tp_size, rank, intermediate_size=intermediate_size
        )
        assert layer.quant_method.weight_scale_refine is None
        assert layer.quant_method.moe_block_shape == [128, 128]
        width = layer.moe_config.intermediate_size_per_partition
        assert width == ((num_blocks + tp_size - 1) // tp_size) * 128
        for name in weights:
            prefix = "w2" if name == "w2" else "w13"
            for suffix, checkpoint in (
                ("weight", weights[name]),
                ("weight_scale_inv", scales[name]),
            ):
                param_name = f"{prefix}_{suffix}"
                param = getattr(layer, param_name)
                # Also reload after dirtying the destination, to catch padding leaks.
                for _ in range(2):
                    if name != "w3":
                        param.data.fill_(7)
                    if batched:
                        assert param.weight_loader(
                            param, checkpoint, param_name, name, 0, return_success=True
                        )
                    else:
                        for expert in range(2):
                            assert param.weight_loader(
                                param,
                                checkpoint[expert],
                                param_name,
                                name,
                                expert,
                                return_success=True,
                            )
            w = getattr(layer, f"{prefix}_weight").float()
            s = getattr(layer, f"{prefix}_weight_scale_inv")
            if name in ("w1", "w3"):
                half = 0 if name == "w1" else 1
                w = w.chunk(2, dim=1)[half]
                s = s.chunk(2, dim=1)[half]
            dim = 2 if name == "w2" else 1
            valid = max(0, min(width, intermediate_size - rank * width))
            assert torch.count_nonzero(w.narrow(dim, valid, width - valid)) == 0
            assert torch.all(s.narrow(dim, valid // 128, (width - valid) // 128) == 1)
            dequant = w * s.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
            reconstructed[name].append(dequant.narrow(dim, 0, valid))
    for name, parts in reconstructed.items():
        expected = weights[name].float() * scales[name].repeat_interleave(
            128, dim=1
        ).repeat_interleave(128, dim=2)
        assert torch.equal(torch.cat(parts, dim=2 if name == "w2" else 1), expected)


@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("tp_size,block", [(2, 64), (4, 32)])
def test_fp8_tp_default_keeps_refined_layout(monkeypatch, backend, tp_size, block):
    layer = _make_fp8_tp_experts(monkeypatch, tp_size, 0, backend)
    assert layer.moe_config.intermediate_size_per_partition == 640 // tp_size
    assert layer.quant_method.moe_block_shape == [block, block]


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_dim", 192),
        ("intermediate_size", 672),
        ("is_lora_enabled", True),
        ("has_bias", True),
    ],
)
def test_fp8_block_aligned_tp_rejects_unsupported_layout(monkeypatch, field, value):
    with pytest.raises(ValueError, match="Block-aligned FP8 TP sharding requires"):
        _make_fp8_tp_experts(monkeypatch, 4, 0, config_overrides={field: value})


@pytest.mark.parametrize("tp_size", [1, 5])
def test_fp8_aligned_tp_keeps_original_layout(monkeypatch, tp_size):
    layer = _make_fp8_tp_experts(monkeypatch, tp_size, 0)
    assert layer.moe_config.intermediate_size_per_partition == 640 // tp_size
    assert not layer.moe_config.tp_shard_with_padding
    assert layer.quant_method.weight_scale_refine is None


def test_fp8_skipped_layer_keeps_original_tp_layout(monkeypatch, default_vllm_config):
    layer = _make_fp8_tp_experts(
        monkeypatch,
        4,
        0,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            ignored_layers=["model.layers.0.mlp.experts"],
        ),
    )
    assert not layer.moe_config.tp_shard_with_padding
    assert layer.moe_config.intermediate_size_per_partition == 160


def test_fp8_block_aligned_tp_rejects_presharded_weights(monkeypatch):
    layer = _make_fp8_tp_experts(monkeypatch, 4, 0)
    with pytest.raises(ValueError, match="unsharded checkpoint"):
        layer.w13_weight.weight_loader(
            layer.w13_weight,
            torch.zeros(160, 256, dtype=torch.float8_e4m3fn),
            "w13_weight",
            "w1",
            0,
        )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 17])
@torch.inference_mode()
def test_fp8_block_aligned_tp_flashinfer_matches_unsharded(
    monkeypatch,
    tp_size,
    num_tokens,
    dist_init,
    workspace_init,
):
    """Compare sharded FI with unsharded FI, reporting the cross-backend
    difference separately because activation quantization can differ.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        pytest.skip("Requires datacenter Blackwell and FlashInfer TRTLLM")
    torch.manual_seed(42)
    e, h, n, topk = 16, 2560, 640, 10
    with torch.device("cuda"), set_current_vllm_config(VllmConfig()):
        x = torch.randn(num_tokens, h, dtype=torch.bfloat16) / 10
        logits = torch.randn(num_tokens, e, dtype=torch.float32)
        probabilities = logits.softmax(dim=-1)
        topk_weights, topk_ids = probabilities.topk(topk, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        weights = {
            "w1": torch.randn(e, n, h).to(torch.float8_e4m3fn),
            "w3": torch.randn(e, n, h).to(torch.float8_e4m3fn),
            "w2": torch.randn(e, h, n).to(torch.float8_e4m3fn),
        }
        scales = {
            name: (torch.rand(e, w.shape[1] // 128, w.shape[2] // 128) + 0.5) / 32
            for name, w in weights.items()
        }
        ref = fused_experts(
            x,
            torch.cat([weights["w1"], weights["w3"]], dim=1),
            weights["w2"],
            topk_weights,
            topk_ids,
            quant_config=fp8_w8a8_moe_quant_config(
                w1_scale=torch.cat([scales["w1"], scales["w3"]], dim=1),
                w2_scale=scales["w2"],
                block_shape=[128, 128],
            ),
        )
        total = torch.zeros_like(ref, dtype=torch.float32)
        fi_reference = None
        for rank in range(-1, tp_size):
            layer = _make_fp8_tp_experts(
                monkeypatch,
                1 if rank == -1 else tp_size,
                max(rank, 0),
                mock_backend=False,
                num_experts=e,
                hidden_dim=h,
            )
            for name in weights:
                prefix = "w2" if name == "w2" else "w13"
                for suffix, checkpoint in (
                    ("weight", weights[name]),
                    ("weight_scale_inv", scales[name]),
                ):
                    param_name = f"{prefix}_{suffix}"
                    param = getattr(layer, param_name)
                    param.weight_loader(param, checkpoint, param_name, name, 0)
            method = layer.quant_method
            method.process_weights_after_loading(layer)
            assert method.is_monolithic
            out = method.apply_monolithic(layer, x, logits)
            assert torch.isfinite(out).all()
            if rank == -1:
                fi_reference = out.float().clone()
            else:
                total += out.float()
        # Different FP8 GEMM reduction orders need not be bitwise identical.
        assert fi_reference is not None
        relative_l2 = (total - fi_reference).norm() / fi_reference.norm().clamp_min(
            1e-8
        )
        cross_backend_l2 = (
            fi_reference - ref.float()
        ).norm() / ref.float().norm().clamp_min(1e-8)
        print(
            f"TP{tp_size} M={num_tokens}: shard_l2={relative_l2.item():.6f}, "
            f"unsharded_fi_vs_triton_l2={cross_backend_l2.item():.6f}"
        )
        assert relative_l2 < 0.01, relative_l2.item()
