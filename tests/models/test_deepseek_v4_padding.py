# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.models import deepseek_v4, deepseek_v4_mtp
from vllm.model_executor.models.deepseek_v4 import (
    DeepseekV4FP8Config,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4MoE,
    _balanced_tp_block_indices,
    _pad_deepseek_v4_tensor,
    _pad_deepseek_v4_tensor_by_tp_blocks,
    _padded_moe_intermediate_size,
)
from vllm.model_executor.models.deepseek_v4_mtp import DeepSeekV4MTP

pytestmark = pytest.mark.skip_global_cleanup


def _fp8_block_config() -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )


def _empty_deepseek_v4_model() -> DeepseekV4Model:
    model = object.__new__(DeepseekV4Model)
    model.config = SimpleNamespace(moe_intermediate_size=3072, n_shared_experts=1)
    model.quant_config = _fp8_block_config()
    return model


def _empty_deepseek_v4_mtp_model() -> DeepSeekV4MTP:
    model = object.__new__(DeepSeekV4MTP)
    model.config = SimpleNamespace(moe_intermediate_size=3072, n_shared_experts=1)
    model.quant_config = _fp8_block_config()
    return model


def _deepseek_v4_moe_config(*, expert_dtype: str = "fp4") -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=7168,
        n_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        n_shared_experts=1,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        num_hash_layers=0,
        hidden_act="silu",
        topk_method="noaux_tc",
        expert_dtype=expert_dtype,
    )


def _assert_tp_balanced_padding(
    padded: torch.Tensor,
    original: torch.Tensor,
    *,
    dim: int,
    block_size: int,
    fill_value: int | float,
    tp_size: int = 16,
) -> None:
    original_blocks = original.shape[dim] // block_size
    padded_blocks = padded.shape[dim] // block_size
    block_indices = _balanced_tp_block_indices(
        original_blocks,
        padded_blocks,
        tp_size,
    )

    src_slices = [slice(None)] * original.ndim
    dst_slices = [slice(None)] * padded.ndim
    for src_block, dst_block in enumerate(block_indices):
        src_slices[dim] = slice(
            src_block * block_size,
            (src_block + 1) * block_size,
        )
        dst_slices[dim] = slice(
            dst_block * block_size,
            (dst_block + 1) * block_size,
        )
        assert torch.equal(padded[tuple(dst_slices)], original[tuple(src_slices)])

    padded_block_set = set(range(padded_blocks))
    original_block_set = set(block_indices)
    for dst_block in padded_block_set - original_block_set:
        dst_slices[dim] = slice(
            dst_block * block_size,
            (dst_block + 1) * block_size,
        )
        assert torch.all(padded[tuple(dst_slices)] == fill_value)


def test_padded_moe_intermediate_size_only_pads_misaligned_fp8_tp():
    quant_config = _fp8_block_config()

    assert _padded_moe_intermediate_size(3072, quant_config, 1) == 3072
    assert _padded_moe_intermediate_size(3072, quant_config, 8) == 3072
    assert _padded_moe_intermediate_size(3072, quant_config, 16) == 4096
    assert _padded_moe_intermediate_size(4096, quant_config, 16) == 4096
    assert _padded_moe_intermediate_size(3072, Mxfp4Config(), 16) == 3072
    assert _padded_moe_intermediate_size(3072, None, 16) == 3072


def test_pad_deepseek_v4_tensor_preserves_values_and_fills_padding():
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    padded = _pad_deepseek_v4_tensor(weight, dim=1, target_size=5, fill_value=1.5)

    assert padded.shape == (2, 5)
    torch.testing.assert_close(padded[:, :3], weight)
    torch.testing.assert_close(padded[:, 3:], torch.full((2, 2), 1.5))


def test_pad_deepseek_v4_tensor_rejects_truncation():
    weight = torch.empty((2, 3))

    with pytest.raises(ValueError, match="Cannot pad DeepSeek V4 tensor"):
        _pad_deepseek_v4_tensor(weight, dim=1, target_size=2)


def test_balanced_tp_block_padding_spreads_zero_blocks_across_ranks():
    assert _balanced_tp_block_indices(24, 32, 16) == [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
        24,
        25,
        26,
        28,
        29,
        30,
    ]

    weight = torch.arange(24, dtype=torch.float32).reshape(24, 1)
    padded = _pad_deepseek_v4_tensor_by_tp_blocks(
        weight,
        dim=0,
        target_size=32,
        block_size=1,
        tp_size=16,
        fill_value=-1.0,
    )

    _assert_tp_balanced_padding(
        padded,
        weight,
        dim=0,
        block_size=1,
        fill_value=-1.0,
    )


def test_balanced_tp_block_padding_preserves_linear_round_trip():
    hidden_size = 5
    original_intermediate_size = 3
    padded_intermediate_size = 4

    hidden_states = torch.randn(2, hidden_size)
    gate_up_weight = torch.randn(original_intermediate_size, hidden_size)
    down_weight = torch.randn(hidden_size, original_intermediate_size)

    original_output = hidden_states.matmul(gate_up_weight.t()).matmul(down_weight.t())

    padded_gate_up_weight = _pad_deepseek_v4_tensor_by_tp_blocks(
        gate_up_weight,
        dim=0,
        target_size=padded_intermediate_size,
        block_size=1,
        tp_size=2,
    )
    padded_down_weight = _pad_deepseek_v4_tensor_by_tp_blocks(
        down_weight,
        dim=1,
        target_size=padded_intermediate_size,
        block_size=1,
        tp_size=2,
    )
    padded_output = hidden_states.matmul(padded_gate_up_weight.t()).matmul(
        padded_down_weight.t()
    )

    torch.testing.assert_close(padded_output, original_output)


def test_shared_experts_loader_padding_for_gate_up_and_down(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose float8_e4m3fn")

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()

    gate_up = torch.ones((3072, 7168), dtype=torch.float8_e4m3fn)
    padded_gate_up = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.gate_up_proj.weight",
        gate_up,
    )
    assert padded_gate_up.shape == (4096, 7168)
    _assert_tp_balanced_padding(
        padded_gate_up.view(torch.uint8),
        gate_up.view(torch.uint8),
        dim=0,
        block_size=128,
        fill_value=0,
    )

    down = torch.ones((7168, 3072), dtype=torch.float8_e4m3fn)
    padded_down = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.down_proj.weight",
        down,
    )
    assert padded_down.shape == (7168, 4096)
    _assert_tp_balanced_padding(
        padded_down.view(torch.uint8),
        down.view(torch.uint8),
        dim=1,
        block_size=128,
        fill_value=0,
    )


def test_shared_experts_scale_loader_padding_uses_e8m0_identity(monkeypatch):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build does not expose float8_e8m0fnu")

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()

    gate_up_scale = torch.ones((24, 56), dtype=torch.float8_e8m0fnu)
    padded_gate_up_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.gate_up_proj.weight_scale_inv",
        gate_up_scale,
    )
    assert padded_gate_up_scale.shape == (32, 56)
    _assert_tp_balanced_padding(
        padded_gate_up_scale.view(torch.uint8),
        gate_up_scale.view(torch.uint8),
        dim=0,
        block_size=1,
        fill_value=127,
    )

    down_scale = torch.ones((56, 24), dtype=torch.float8_e8m0fnu)
    padded_down_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv",
        down_scale,
    )
    assert padded_down_scale.shape == (56, 32)
    _assert_tp_balanced_padding(
        padded_down_scale.view(torch.uint8),
        down_scale.view(torch.uint8),
        dim=1,
        block_size=1,
        fill_value=127,
    )


def test_routed_fp8_experts_loader_padding_for_gate_up_and_down(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose float8_e4m3fn")

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()
    model.config.expert_dtype = "fp8"

    w1 = torch.ones((3072, 7168), dtype=torch.float8_e4m3fn)
    padded_w1 = model._maybe_pad_routed_experts_weight(
        "model.layers.0.ffn.experts.0.w1.weight",
        w1,
    )
    assert padded_w1.shape == (4096, 7168)
    _assert_tp_balanced_padding(
        padded_w1.view(torch.uint8),
        w1.view(torch.uint8),
        dim=0,
        block_size=128,
        fill_value=0,
    )

    w2 = torch.ones((7168, 3072), dtype=torch.float8_e4m3fn)
    padded_w2 = model._maybe_pad_routed_experts_weight(
        "model.layers.0.ffn.experts.0.w2.weight",
        w2,
    )
    assert padded_w2.shape == (7168, 4096)
    _assert_tp_balanced_padding(
        padded_w2.view(torch.uint8),
        w2.view(torch.uint8),
        dim=1,
        block_size=128,
        fill_value=0,
    )


def test_routed_fp8_experts_scale_padding_uses_float_identity(monkeypatch):
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()
    model.config.expert_dtype = "fp8"

    w1_scale = torch.ones((24, 56), dtype=torch.float32)
    padded_w1_scale = model._maybe_pad_routed_experts_weight(
        "model.layers.0.ffn.experts.0.w1.weight_scale_inv",
        w1_scale,
    )
    assert padded_w1_scale.shape == (32, 56)
    _assert_tp_balanced_padding(
        padded_w1_scale,
        w1_scale,
        dim=0,
        block_size=1,
        fill_value=1.0,
    )

    w2_scale = torch.ones((56, 24), dtype=torch.float32)
    padded_w2_scale = model._maybe_pad_routed_experts_weight(
        "model.layers.0.ffn.experts.0.w2.weight_scale_inv",
        w2_scale,
    )
    assert padded_w2_scale.shape == (56, 32)
    _assert_tp_balanced_padding(
        padded_w2_scale,
        w2_scale,
        dim=1,
        block_size=1,
        fill_value=1.0,
    )


def test_mtp_shared_experts_loader_reuses_tp16_padding(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose float8_e4m3fn")

    monkeypatch.setattr(
        deepseek_v4_mtp, "get_tensor_model_parallel_world_size", lambda: 16
    )
    model = _empty_deepseek_v4_mtp_model()

    gate_up = torch.ones((3072, 7168), dtype=torch.float8_e4m3fn)
    padded_gate_up = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.gate_up_proj.weight",
        gate_up,
    )
    assert padded_gate_up.shape == (4096, 7168)
    _assert_tp_balanced_padding(
        padded_gate_up.view(torch.uint8),
        gate_up.view(torch.uint8),
        dim=0,
        block_size=128,
        fill_value=0,
    )

    down = torch.ones((7168, 3072), dtype=torch.float8_e4m3fn)
    padded_down = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.down_proj.weight",
        down,
    )
    assert padded_down.shape == (7168, 4096)
    _assert_tp_balanced_padding(
        padded_down.view(torch.uint8),
        down.view(torch.uint8),
        dim=1,
        block_size=128,
        fill_value=0,
    )


def test_mtp_shared_experts_scale_padding_uses_e8m0_identity(monkeypatch):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build does not expose float8_e8m0fnu")

    monkeypatch.setattr(
        deepseek_v4_mtp, "get_tensor_model_parallel_world_size", lambda: 16
    )
    model = _empty_deepseek_v4_mtp_model()

    down_scale = torch.ones((56, 24), dtype=torch.float8_e8m0fnu)
    padded_down_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.down_proj.weight_scale_inv",
        down_scale,
    )

    assert padded_down_scale.shape == (56, 32)
    _assert_tp_balanced_padding(
        padded_down_scale.view(torch.uint8),
        down_scale.view(torch.uint8),
        dim=1,
        block_size=1,
        fill_value=127,
    )


def test_tp16_fp8_shared_experts_mlp_requires_padded_intermediate(monkeypatch):
    import vllm.distributed as distributed
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.layers.quantization.fp8 as fp8
    import vllm.model_executor.parameter as parameter

    for module in (distributed, linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 16)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    monkeypatch.setattr(fp8, "init_fp8_linear_kernel", lambda **kwargs: object())

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    quant_config = _fp8_block_config()

    with set_current_vllm_config(vllm_config):
        with pytest.raises(ValueError, match="input_size_per_partition = 192"):
            DeepseekV4MLP(
                hidden_size=7168,
                intermediate_size=3072,
                hidden_act="silu",
                quant_config=quant_config,
                prefix="model.layers.0.ffn.shared_experts",
            )

        padded_size = _padded_moe_intermediate_size(3072, quant_config, 16)
        mlp = DeepseekV4MLP(
            hidden_size=7168,
            intermediate_size=padded_size,
            hidden_act="silu",
            quant_config=quant_config,
            prefix="model.layers.0.ffn.shared_experts",
        )

    assert padded_size == 4096
    assert mlp.down_proj.input_size_per_partition == 256
    assert mlp.gate_up_proj.output_partition_sizes == [256, 256]


@pytest.mark.parametrize(
    ("expert_dtype", "expected_routed_size", "expected_shared_size"),
    [
        ("fp4", 3072, 4096),
        ("fp8", 4096, 4096),
    ],
)
def test_tp16_moe_construction_preserves_expert_dtype_padding_contract(
    expert_dtype: str,
    expected_routed_size: int,
    expected_shared_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeGate(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object):
            super().__init__()
            self.e_score_correction_bias = None
            self.tid2eid = None

    class FakeMLP(torch.nn.Module):
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs: object):
            super().__init__()
            self.kwargs = kwargs
            FakeMLP.calls.append(kwargs)

    class FakeFusedMoE(torch.nn.Module):
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs: object):
            super().__init__()
            self.kwargs = kwargs
            FakeFusedMoE.calls.append(kwargs)

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(deepseek_v4, "GateLinear", FakeGate)
    monkeypatch.setattr(deepseek_v4, "DeepseekV4MLP", FakeMLP)
    monkeypatch.setattr(deepseek_v4, "FusedMoE", FakeFusedMoE)

    config = _deepseek_v4_moe_config(expert_dtype=expert_dtype)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=_fp8_block_config(),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        kernel_config=SimpleNamespace(moe_backend=None),
    )

    moe = DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")

    assert moe.shared_experts_intermediate_size == expected_shared_size
    assert FakeMLP.calls[0]["intermediate_size"] == expected_shared_size
    assert FakeFusedMoE.calls[0]["intermediate_size"] == expected_routed_size


def test_deepseek_v4_fp8_config_preserves_expert_dtype_dispatch(monkeypatch):
    import vllm.config as vllm_config_module
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    fake_layer = object.__new__(FusedMoE)
    fake_layer.moe_config = SimpleNamespace()
    fake_layer.local_num_experts = 1

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.mxfp4.select_mxfp4_moe_backend",
        lambda moe: (None, None),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.select_fp8_moe_backend",
        lambda **kwargs: (None, None),
    )
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    vllm_config.model_config.hf_config = SimpleNamespace(expert_dtype="fp4")
    monkeypatch.setattr(
        vllm_config_module, "get_current_vllm_config", lambda: vllm_config
    )
    with set_current_vllm_config(vllm_config):
        assert isinstance(
            quant_config.get_quant_method(fake_layer, "model.layers.0.ffn.experts"),
            Mxfp4MoEMethod,
        )
    assert quant_config.is_mxfp4_quant("model.layers.0.ffn.experts", fake_layer)
    assert quant_config.is_scale_e8m0

    quant_config._resolved_expert_dtype = None
    vllm_config.model_config.hf_config = SimpleNamespace(expert_dtype="fp8")
    monkeypatch.setattr(
        vllm_config_module, "get_current_vllm_config", lambda: vllm_config
    )
    with set_current_vllm_config(vllm_config):
        method = quant_config.get_quant_method(fake_layer, "model.layers.0.ffn.experts")
    assert isinstance(method, Fp8MoEMethod)
    assert not isinstance(method, Mxfp4MoEMethod | UnquantizedFusedMoEMethod)
    assert not quant_config.is_mxfp4_quant("model.layers.0.ffn.experts", fake_layer)
    assert not quant_config.is_scale_e8m0


def test_routed_mxfp4_experts_keep_checkpoint_intermediate_size():
    # The default DeepSeek V4 Flash route keeps routed experts on MXFP4, while
    # only FP8 linear/shared expert paths require TP16 block-shape padding.
    assert _padded_moe_intermediate_size(3072, Mxfp4Config(), 16) == 3072
