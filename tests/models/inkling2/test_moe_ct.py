# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 compressed-tensors experts (`moe_ct`): scheme selection, loading
of llm-compressor's per-expert tensors, and the MoE layer vs. a dequantized
torch reference."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

import vllm.models.inkling2.nvidia.moe_ct as moe_ct
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.models.inkling2.nvidia.moe_ct import (
    Inkling2CTExperts,
    Inkling2CTScheme,
    _pack_cols,
    _pack_rows,
    _unpack_rows,
)
from vllm.models.inkling2.quant_config import Inkling2QuantConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

PREFIX = "model.layers.0.mlp"
EXPERT_TARGETS = [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"]


def _int_weights(num_bits=4, strategy="group", symmetric=True, **kwargs):
    return {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": symmetric,
        "strategy": strategy,
        "group_size": 128 if strategy == "group" else None,
        "dynamic": False,
        **kwargs,
    }


def _fp8_weights(strategy="channel"):
    return {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": strategy,
        "block_structure": [128, 128] if strategy == "block" else None,
        "dynamic": False,
    }


def _dynamic_activations(type_="int", strategy="token"):
    return {
        "num_bits": 8,
        "type": type_,
        "symmetric": True,
        "strategy": strategy,
        "group_size": 128 if strategy == "group" else None,
        "dynamic": True,
    }


# llm-compressor presets of the expert schemes (name -> format, weights, acts).
SCHEMES = {
    "W4A16": ("pack-quantized", _int_weights(), None),
    "W4A16_ASYM": ("pack-quantized", _int_weights(symmetric=False), None),
    "W8A16": ("pack-quantized", _int_weights(8), None),
    # int4 values, one per int8.
    "W4A8": ("int-quantized", _int_weights(), _dynamic_activations()),
    "W8A8_INT8": ("int-quantized", _int_weights(8, "channel"), _dynamic_activations()),
    "W8A8_FP8": ("float-quantized", _fp8_weights(), _dynamic_activations("float")),
    "FP8_BLOCK": (
        "float-quantized",
        _fp8_weights("block"),
        _dynamic_activations("float", "group"),
    ),
}


def _ct_config(fmt, weights, input_activations=None, ignore=()):
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {
            "group_0": {
                "targets": EXPERT_TARGETS,
                "weights": weights,
                "input_activations": input_activations,
                "output_activations": None,
                "format": fmt,
            }
        },
        "ignore": list(ignore),
    }


def _scheme(name: str) -> Inkling2CTScheme:
    fmt, weights, acts = SCHEMES[name]
    config = CompressedTensorsConfig.from_config(_ct_config(fmt, weights, acts))
    return Inkling2CTScheme.from_config(config, PREFIX)


@pytest.mark.parametrize("bits", [4, 8])
def test_zero_point_rows_match_compressed_tensors_packing(bits: int):
    """`_unpack_rows` / `_pack_rows` are compressed-tensors' dim-0 packing."""
    torch.manual_seed(0)
    low = -(1 << (bits - 1))
    values = torch.randint(low, -low, (64, 5), dtype=torch.int8)
    packed = pack_to_int32(values, bits, packed_dim=0)
    rows = _unpack_rows(packed, bits)
    # pack_to_int32 stores the values offset to unsigned.
    torch.testing.assert_close(rows, values.int() - low, atol=0, rtol=0)
    torch.testing.assert_close(_pack_rows(rows, bits), packed, atol=0, rtol=0)
    # W4A8's unpacked int4 weights are packed as `pack-quantized` packs them.
    expected = pack_to_int32(values.T.contiguous(), bits)
    torch.testing.assert_close(_pack_cols(values.T, bits), expected, atol=0, rtol=0)


def test_compressed_tensors_checkpoints_keep_their_method():
    hf_config = SimpleNamespace(architectures=["Inkling2ForConditionalGeneration"])
    override = Inkling2QuantConfig.override_quantization_method
    ct_config = _ct_config(*SCHEMES["W4A16_ASYM"])
    assert override(ct_config, None, hf_config) is None
    # Released checkpoints (a ModelOpt config.json) still use hf_quant_config.
    assert override({"quant_method": "modelopt"}, None, hf_config) == "inkling2"


requires_humming_package = pytest.mark.skipif(
    not has_humming(), reason="humming is not installed"
)


@requires_humming_package
@pytest.mark.parametrize("name", list(SCHEMES))
def test_scheme_maps_to_humming_schemas(name: str):
    fmt, weights, acts = SCHEMES[name]
    scheme = _scheme(name)
    weight_schema = scheme.humming_weight_schema()
    input_schema = scheme.humming_input_schema(fallback=False)
    assert weight_schema.format == ("pack-quantized" if name == "W4A8" else fmt)
    assert weight_schema.num_bits == weights["num_bits"]
    assert weight_schema.type == weights["type"]
    assert weight_schema.symmetric == weights["symmetric"]
    if weights["strategy"] == "block":
        # Block scales are repeated over their rows at load time.
        assert scheme.block_rows == 128
        assert (weight_schema.strategy, weight_schema.group_size) == ("group", 128)
    else:
        assert scheme.block_rows == 1
        assert weight_schema.strategy == weights["strategy"]
        assert weight_schema.group_size == weights.get("group_size")
    if acts is None:
        assert input_schema is None
    else:
        assert input_schema is not None
        a_dtype = input_schema.a_dtype
        assert a_dtype.num_bits == 8
        assert a_dtype.is_floating_point_type == (acts["type"] == "float")
        assert input_schema.input_scale_group_size == (acts["group_size"] or 0)
        # Dynamic: per token, or per group of each token.
        assert input_schema.static_tensor_scale_name is None


@pytest.mark.parametrize(
    "config, match",
    [
        (
            _ct_config("float-quantized", _fp8_weights("tensor"), None),
            "'tensor' weight strategy",
        ),
        (
            _ct_config(
                "float-quantized",
                _fp8_weights(),
                {
                    **_dynamic_activations("float"),
                    "dynamic": False,
                    "strategy": "tensor",
                },
            ),
            "static activation scales",
        ),
        (
            _ct_config(
                "int-quantized",
                _int_weights(8, "channel", False),
                _dynamic_activations(),
            ),
            "zero points",
        ),
        (
            _ct_config(
                "pack-quantized",
                _int_weights(),
                None,
                ignore=[f"{PREFIX}.experts.0.up_proj"],
            ),
            "unquantized Inkling2 experts",
        ),
    ],
)
def test_unsupported_schemes_are_rejected(config, match: str):
    """Rejected at model construction, before any weight is loaded."""
    with pytest.raises(ValueError, match=match):
        Inkling2CTScheme.from_config(
            CompressedTensorsConfig.from_config(config), PREFIX
        )


MOE_CONFIG = SimpleNamespace(
    hidden_size=512,
    intermediate_size=256,
    n_routed_experts=16,
    n_shared_experts=0,
    num_experts_per_tok=8,
    route_scale=8.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    use_gate_bias=True,
)


def _quantize(
    w: torch.Tensor, name: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """compressed-tensors tensors of one bf16 projection `[N, K]` in scheme
    `name`, and their dequantized weights."""
    fmt, args, _ = SCHEMES[name]
    w = w.float()
    n, k = w.shape
    if args["type"] == "float":
        if args["strategy"] == "block":
            blocks = w.reshape(n // 128, 128, k // 128, 128)
            scale = (blocks.abs().amax((1, 3)) / 448).bfloat16()
            full = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        else:
            scale = (w.abs().amax(1, keepdim=True) / 448).bfloat16()
            full = scale.float()
        q = (w / full).to(torch.float8_e4m3fn)
        return {"weight": q, "weight_scale": scale}, q.float() * full

    bits = args["num_bits"]
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    group = args["group_size"] or k
    g = w.reshape(n, k // group, group)
    if args["symmetric"]:
        scale = (g.abs().amax(-1) / qmax).bfloat16()
        zero = torch.zeros_like(scale, dtype=torch.float32)
    else:
        low, high = g.amin(-1).clamp(max=0), g.amax(-1).clamp(min=0)
        scale = ((high - low) / (qmax - qmin)).bfloat16()
        zero = (qmin - low / scale.float()).round().clamp(qmin, qmax)
    q = (g / scale.float()[..., None] + zero[..., None]).round().clamp(qmin, qmax)
    dequant = ((q - zero[..., None]) * scale.float()[..., None]).reshape(n, k)
    q = q.reshape(n, k).to(torch.int8)
    if fmt == "pack-quantized":
        tensors = {"weight_packed": pack_to_int32(q, bits)}
    else:
        tensors = {"weight": q}
    tensors["weight_scale"] = scale
    if not args["symmetric"]:
        tensors["weight_zero_point"] = pack_to_int32(
            zero.to(torch.int8), bits, packed_dim=0
        )
    return tensors, dequant


def _random_experts(name: str, device: str):
    """Per-expert checkpoint tensors {(expert, projection, key): tensor} and
    the dequantized `[E, 2I, H]` (gate rows, then up) and `[E, H, I]`."""
    torch.manual_seed(0)
    cfg = MOE_CONFIG
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    tensors = {}
    w13, w2 = [], []
    for e in range(cfg.n_routed_experts):
        dequant = {}
        for proj, (n, k) in (
            ("gate_proj", (inter, hidden)),
            ("up_proj", (inter, hidden)),
            ("down_proj", (hidden, inter)),
        ):
            w = torch.randn(n, k, device=device) * (
                0.05 if proj == "gate_proj" else 0.1
            )
            w[:, :5] += 0.3  # Uneven groups (zero points away from the middle).
            proj_tensors, dequant[proj] = _quantize(w, name)
            for key, t in proj_tensors.items():
                tensors[(e, proj, key)] = t
        w13.append(torch.cat([dequant["gate_proj"], dequant["up_proj"]]))
        w2.append(dequant["down_proj"])
    return tensors, torch.stack(w13), torch.stack(w2)


def _load(experts: Inkling2CTExperts, tensors) -> None:
    params = dict(experts.named_parameters())
    for (e, proj, key), t in tensors.items():
        name = f"{moe_ct.PROJECTIONS[proj][0]}_{key}"
        experts.load_expert(params[name], t, e, proj, key)


@requires_humming_package
@pytest.mark.parametrize("name", ["W4A16_ASYM", "W4A8", "W8A8_FP8", "FP8_BLOCK"])
def test_experts_load_their_shard_with_gate_up_rows_interleaved(monkeypatch, name: str):
    """Rank 1 of 2 keeps experts 8-15, gate / up rows at even / odd rows."""
    monkeypatch.setattr(moe_ct, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(moe_ct, "get_tensor_model_parallel_world_size", lambda: 2)
    tensors, _, _ = _random_experts(name, "cpu")
    experts = Inkling2CTExperts(MOE_CONFIG, _scheme(name))
    assert experts.num_experts == 8
    _load(experts, tensors)
    block = experts.scheme.block_rows
    for _, key in experts.tensor_names:
        w13 = getattr(experts, f"w13_{key}").data
        w2 = getattr(experts, f"w2_{key}").data
        for local in range(8):
            gate, up, down = (tensors[(8 + local, p, key)] for p in moe_ct.PROJECTIONS)
            if key == "weight_scale" and block > 1:
                gate, up, down = (
                    t.repeat_interleave(block, 0) for t in (gate, up, down)
                )
            if key == "weight" and name == "W4A8":
                gate, up, down = (_pack_cols(t, 4) for t in (gate, up, down))
            rows = w13[local]
            if key == "weight_zero_point":
                rows, gate, up = (_unpack_rows(t, 4) for t in (rows, gate, up))
            assert torch.equal(rows[0::2], gate) and torch.equal(rows[1::2], up)
            assert torch.equal(w2[local], down)


def _quantize_activations(x: torch.Tensor, a_dtype, group: int) -> torch.Tensor:
    """Dynamic symmetric per-token (or per-group) fake quantization."""
    x = x.float()
    shape = x.shape
    if group:
        x = x.reshape(*shape[:-1], -1, group)
    if a_dtype == torch.int8:
        scale = x.abs().amax(-1, keepdim=True) / 127
        q = (x / scale).round().clamp(-128, 127)
    else:
        scale = x.abs().amax(-1, keepdim=True) / 448
        q = (x / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape)


requires_humming = pytest.mark.skipif(
    not (
        (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
        and has_humming()
    ),
    reason="Inkling2 MoE requires Hopper or Blackwell, and Humming",
)


@requires_humming
@pytest.mark.parametrize("name", list(SCHEMES))
@pytest.mark.parametrize("tokens", [5, 33])
def test_ct_moe_layer_matches_reference(dist_init, name: str, tokens: int):
    """Full layer (TP=1) vs. a per-token loop over the dequantized experts,
    with activations quantized as the executed Humming config does."""
    from vllm.models.inkling2.nvidia.moe import Inkling2MoE
    from vllm.models.inkling2.nvidia.ops.linear import linear_k_groups

    cfg = MOE_CONFIG
    tensors, w13, w2 = _random_experts(name, "cuda")
    with torch.device("cuda"):
        moe = Inkling2MoE(cfg, _scheme(name))
    torch.manual_seed(1)
    moe.gate.weight.data.copy_(torch.randn(16, 512, device="cuda") * 0.05)
    moe.gate.bias.data.copy_(torch.randn(16, device="cuda") * 0.01)
    _load(moe.experts, tensors)
    moe.experts.process_weights_after_loading()
    assert moe.experts.humming is not None

    # Humming may run FP8 activations as bf16 where it has no FP8 MMA.
    _, acts = SCHEMES[name][1:]
    config = moe.experts.humming.configs["w13"]
    a_dtype = None
    if acts is not None and config.a_dtype.num_bits == 8:
        a_dtype = torch.int8 if acts["type"] == "int" else torch.float8_e4m3fn
    group = (acts["group_size"] or 0) if acts is not None else 0

    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    out = moe(x)

    logits = linear_k_groups(x, moe.gate.weight)
    _, ids = torch.topk(logits.sigmoid() + moe.gate.bias, 8, sorted=True)
    gammas = torch.nn.functional.logsigmoid(logits.gather(1, ids))
    gammas = gammas.softmax(-1) * cfg.route_scale

    def linear(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if a_dtype is not None:
            a = _quantize_activations(a, a_dtype, group)
        return (a.double() @ w.double().T).bfloat16().double()

    ref = torch.zeros(x.shape, dtype=torch.float64, device="cuda")
    inter = cfg.intermediate_size
    for t in range(tokens):
        for slot in range(8):
            e = ids[t, slot].item()
            gate_up = linear(x[t : t + 1], w13[e])
            gate, up = gate_up[:, :inter], gate_up[:, inter:]
            act = (
                cfg.mlp_gate_softcap
                * torch.tanh(gate / cfg.mlp_gate_softcap)
                * torch.sigmoid(gate)
                * cfg.mlp_up_softcap
                * torch.tanh(up / cfg.mlp_up_softcap)
                * gammas[t, slot]
            )
            ref[t] += linear(act.bfloat16(), w2[e])[0]
    ref = ref.bfloat16()
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)
