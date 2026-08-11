# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU test for W4A16 MoE through the humming indexed path.

Runs vLLM's ``HummingIndexedExperts.apply`` against any provider of the
``humming`` package (upstream or the chord shim) with uint4/group-32/BF16-scale
weights, checking the output against a dequantized PyTorch reference.

Case matrix:
- small generic shapes (default profile, WGMMA prefill layout on H200);
- Kimi-K2.5 EP8 projection shapes at a routed-M where the tuned w13/w2
  block-M values diverge in the provider's own tables (exercises whatever
  routing alignment the provider ships);
- the same EP8 shapes under the SM90 decode role (MMA + swap-AB).
"""

import pytest
import torch
from torch.nn import Parameter

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.platforms import current_platform

humming = pytest.importorskip("humming")
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires a CUDA GPU"
)


def quantize_w4a16(w_ref: torch.Tensor, group_size: int = 32):
    """Symmetric uint4 quantize bf16 [E, N, K] weights, humming checkpoint layout."""
    E, N, K = w_ref.shape
    wg = w_ref.float().view(E, N, K // group_size, group_size)
    amax = wg.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp_min(1e-6)
    codes = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).add(8).to(torch.int32)
    packed = codes.view(E, N, K // 8, 8)
    shifts = torch.arange(8, dtype=torch.int32, device=w_ref.device) * 4
    packed = (packed << shifts).sum(-1)
    return packed.to(torch.int32), scale.to(torch.bfloat16)


def dequant_w4a16(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    E, N, words = packed.shape
    shifts = torch.arange(8, dtype=torch.int64, device=packed.device) * 4
    codes = (
        (packed.to(torch.int64).unsqueeze(-1) >> shifts)
        .bitwise_and(0xF)
        .reshape(E, N, words * 8)
        .float()
        - 8.0
    )
    return (codes * scale.float().repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def make_w4a16_humming_experts(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    seed: int = 0,
    checkpoint_format: str = "humming",
):
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.quantization.utils import humming_utils
    from vllm.utils import humming

    activation = MoEActivation.SILU
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16

    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate_up_size = intermediate_size * 2
    refs = {}
    packed_by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for sublayer_name, shape_n, shape_k in (
        ("w13", gate_up_size, hidden_size),
        ("w2", hidden_size, intermediate_size),
    ):
        w_ref = torch.randn(
            (num_experts, shape_n, shape_k),
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        packed, scale = quantize_w4a16(w_ref)
        refs[sublayer_name] = dequant_w4a16(packed, scale).float()
        packed_by_name[sublayer_name] = (packed, scale)

    if checkpoint_format == "humming":
        # humming checkpoint layout: weight [E, N, K/8], scale [E, N, K/32]
        for sublayer_name, (packed, scale) in packed_by_name.items():
            layer.register_parameter(
                f"{sublayer_name}_weight", Parameter(packed, requires_grad=False)
            )
            layer.register_parameter(
                f"{sublayer_name}_weight_scale",
                Parameter(scale, requires_grad=False),
            )
        humming_utils.convert_to_humming_moe_kernel_format(
            layer,
            weight_schema=humming.HummingWeightSchema(
                b_dtype=humming.dtypes.uint4,
                weight_scale_group_size=32,
            ),
            input_schema=humming.HummingInputSchema(a_dtype=humming.dtypes.bfloat16),
        )
    elif checkpoint_format == "compressed_tensors":
        # compressed-tensors WNA16 layout: weight [E, K/8, N], scale
        # [E, K/32, N] (K-major in both, unlike the humming checkpoint).
        from compressed_tensors.quantization import QuantizationArgs
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
            convert_to_wna16_moe_kernel_format,
        )

        quant_config = QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=32
        )
        convert_to_wna16_moe_kernel_format(
            backend=WNA16MoEBackend.HUMMING,
            layer=layer,
            quant_config=quant_config,
            input_dtype=torch.bfloat16,
            **{
                name: tensor
                for sublayer_name, (packed, scale) in packed_by_name.items()
                for name, tensor in _ct_register(layer, sublayer_name, packed, scale)
            },
        )
    else:
        raise ValueError(checkpoint_format)

    layer.local_num_experts = layer.global_num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    quant_config = humming_utils.get_humming_moe_quant_config(layer)
    experts = HummingIndexedExperts(layer, moe_config, quant_config)
    return experts, refs


def _ct_register(
    layer: torch.nn.Module, sublayer_name: str, packed: torch.Tensor, scale: torch.Tensor
):
    """Register compressed-tensors K-major params and return the convert args."""
    packed_ct = packed.transpose(1, 2).contiguous()  # [E, K/8, N]
    scale_ct = scale.transpose(1, 2).contiguous()  # [E, K/32, N]
    layer.register_parameter(
        f"{sublayer_name}_weight_packed", Parameter(packed_ct, requires_grad=False)
    )
    layer.register_parameter(
        f"{sublayer_name}_weight_scale", Parameter(scale_ct, requires_grad=False)
    )
    return [
        (sublayer_name, packed_ct),
        (f"{sublayer_name}_scale", scale_ct),
    ]


def reference_moe(
    hidden_states: torch.Tensor,
    w13_ref: torch.Tensor,
    w2_ref: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gated: bool = True,
) -> torch.Tensor:
    out = torch.zeros(
        hidden_states.shape[0],
        w2_ref.shape[1],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    x = hidden_states.float()
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            gu = x[t] @ w13_ref[e].T
            if gated:
                n = gu.shape[0] // 2
                act = torch.nn.functional.silu(gu[:n]) * gu[n:]
            else:
                act = torch.nn.functional.silu(gu)
            out[t] += topk_weights[t, k].float() * (act @ w2_ref[e].T)
    return out


def run_humming_case(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    seed: int = 0,
    checkpoint_format: str = "humming",
) -> None:
    experts, refs = make_w4a16_humming_experts(
        num_experts, hidden_size, intermediate_size, top_k, seed, checkpoint_format
    )
    _apply_and_check(
        experts, refs, num_experts, hidden_size, intermediate_size, top_k,
        num_tokens, MoEActivation.SILU,
    )


def _apply_and_check(
    experts,
    refs,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    activation,
    seed: int = 0,
) -> None:
    moe_config = experts.moe_config

    workspace13_shape, workspace2_shape, _ = experts.workspace_shapes(
        M=num_tokens,
        N=intermediate_size,
        K=hidden_size,
        topk=top_k,
        global_num_experts=num_experts,
        local_num_experts=num_experts,
        expert_tokens_meta=None,
        activation=activation,
    )
    device = torch.device("cuda")
    workspace13 = torch.empty(workspace13_shape, dtype=torch.bfloat16, device=device)
    workspace2 = torch.empty(workspace2_shape, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), generator=gen, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    output = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1 / top_k, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=gen, device=device
    )
    unused = torch.empty((num_experts, 0), device=device)

    with set_forward_context(None, VllmConfig(), num_tokens=num_tokens):
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=unused,
            w2=unused,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    ref = reference_moe(
        hidden_states, refs["w13"], refs["w2"], topk_weights, topk_ids
    )
    torch.testing.assert_close(
        output.float(), ref, rtol=1e-1, atol=1e-1 * max(1.0, ref.abs().max().item())
    )


def test_w4a16_humming_indexed_small_shapes():
    # Non-published shapes: conservative default tiles on both projections.
    run_humming_case(
        num_experts=8, hidden_size=256, intermediate_size=128, top_k=2, num_tokens=4
    )


def test_w4a16_humming_indexed_kimi_ep8_divergent_block_m():
    # Kimi-K2.5 EP8 shapes. routed_m = 2000*8 = 16000 lands where the H200
    # prefill tables pick different native block_m for w13 (176) vs w2 (128);
    # the shared routing works only if the provider aligns w2's block_m to
    # w13's (or receives it explicitly).
    run_humming_case(
        num_experts=48, hidden_size=7168, intermediate_size=2048, top_k=8,
        num_tokens=2000,
    )


def test_w4a16_humming_indexed_kimi_ep8_decode(monkeypatch):
    # SM90 decode role: MMA layout with swap-AB small tiles.
    monkeypatch.setenv("CHORD_SM90_DECODE", "1")
    run_humming_case(
        num_experts=48, hidden_size=7168, intermediate_size=2048, top_k=8,
        num_tokens=40,
    )


def test_w4a16_humming_indexed_compressed_tensors_small():
    # compressed-tensors K-major checkpoint layout routed through
    # convert_to_wna16_moe_kernel_format + _CompressedTensorsHummingWeightSchema.
    run_humming_case(
        num_experts=8, hidden_size=256, intermediate_size=128, top_k=2,
        num_tokens=4, checkpoint_format="compressed_tensors",
    )


def test_w4a16_humming_indexed_compressed_tensors_kimi_ep8():
    # The production combination: compressed-tensors checkpoint, Kimi EP8
    # shapes, divergent native w13/w2 block_m brackets.
    run_humming_case(
        num_experts=48, hidden_size=7168, intermediate_size=2048, top_k=8,
        num_tokens=2000, checkpoint_format="compressed_tensors",
    )


def test_w4a16_humming_native_quant_route_kimi_ep8():
    """The `--quantization humming` route end to end.

    Exercises exactly what ``HummingMoEMethod`` does: build per-expert
    parameters from the checkpoint schema's ``get_padded_tensors_attrs``,
    then convert + transform + apply through the humming layer contract.
    """
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.quantization.utils import humming_utils
    from vllm.utils import humming

    num_experts, hidden_size, intermediate_size, top_k, num_tokens = (
        48, 7168, 2048, 8, 2000,
    )
    activation = MoEActivation.SILU
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )

    ct_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 32,
        "num_bits": 4,
        "observer": "minmax",
        "strategy": "group",
        "symmetric": True,
        "type": "int",
    }
    ct_schema = humming.BaseWeightSchema.from_config(ct_config)

    layer = torch.nn.Module()
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16

    gen = torch.Generator(device="cuda").manual_seed(0)
    refs: dict[str, torch.Tensor] = {}
    layer.sublayer_configs = {}
    for sublayer_name, shape_n, shape_k in (
        ("w13", intermediate_size * 2, hidden_size),
        ("w2", hidden_size, intermediate_size),
    ):
        w_ref = torch.randn(
            (num_experts, shape_n, shape_k),
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        packed, scale = quantize_w4a16(w_ref)
        refs[sublayer_name] = dequant_w4a16(packed, scale).float()

        tensors_attrs = ct_schema.get_padded_tensors_attrs(
            shape_n=shape_n,
            shape_k=shape_k,
            param_dtype=torch.bfloat16,
            num_experts=num_experts,
        )
        fill = {
            "weight_packed": packed,
            "weight_scale": scale,
            "weight_shape": torch.tensor([shape_n, shape_k], dtype=torch.int64),
        }
        for name, attrs in tensors_attrs.items():
            layer.register_parameter(
                f"{sublayer_name}_{name}",
                Parameter(fill[name].contiguous(), requires_grad=False),
            )
        layer.sublayer_configs[sublayer_name] = {
            "shape_n": shape_n,
            "shape_k": shape_k,
        }

    humming_utils.convert_to_humming_moe_kernel_format(
        layer=layer,
        sublayer_configs=layer.sublayer_configs,
        weight_schema=ct_schema,
        input_schema=humming.HummingInputSchema(a_dtype=humming.dtypes.bfloat16),
    )

    layer.local_num_experts = layer.global_num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    quant_config = humming_utils.get_humming_moe_quant_config(layer)
    experts = HummingIndexedExperts(layer, moe_config, quant_config)
    _apply_and_check(experts, refs, num_experts, hidden_size, intermediate_size,
                     top_k, num_tokens, activation)
