# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness and tensor-parallel parity for online quantization."""

from contextlib import contextmanager, nullcontext
from itertools import product

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from tests.quantization.test_online import (
    DEVICE,
    _enable_aiter_mxfp4_moe,
    _fully_quantized_modelopt_mxfp8_config,
    _make_nvfp4_moe_source,
    _materialize_requantization_source,
    _naive_block_fp8_dequant,
    _nvfp4_moe_checkpoint_tensors,
    _serialize_nvfp4_experts,
    _write_minimal_llama_config,
    on_gfx950,
)
from tests.quantization.utils import quant_config_args
from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import ModelConfig
from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.online.int8 import Int8OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
    Nvfp4WeightOnlyOnlineLinearMethod,
    _quantize_nvfp4,
)
from vllm.model_executor.layers.quantization.online.requantization import (
    skip_checkpoint_backend_selection,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
    QuarkW8A8Fp8MoEMethod,
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
    QuarkW8A8Fp8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_tp_weight_quant,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kNvfp4DynamicToken,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype


class _EmulatedRankGroup(SingleRankGroup):
    """Logical TP/EP membership for a rank running on device 0."""

    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1

    def rank(self) -> int:
        return self.rank_in_group

    def size(self) -> int:
        return self.world_size


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def run_numerics(request, tmp_path_factory):
    """Run each logical TP rank sequentially on the same accelerator."""
    if torch.accelerator.device_count() == 0:
        pytest.skip("Requires one accelerator.")
    tp_size = request.param
    path = tmp_path_factory.mktemp("online_numerics")
    _write_minimal_llama_config(path)
    model_config = ModelConfig(
        model=str(path), dtype="bfloat16", skip_tokenizer_init=True
    )

    def run(test, source, target, *options):
        torch.accelerator.set_device_index(0)
        for rank in range(tp_size):
            with (
                override_parallel_groups(SingleRankGroup()),
                pytest.MonkeyPatch.context() as patch,
                set_current_vllm_config(VllmConfig(model_config=model_config)),
                set_default_torch_dtype(torch.bfloat16),
                torch.device(f"{DEVICE}:0"),
            ):
                patch.setattr(parallel_state, "_TP", _EmulatedRankGroup(rank, tp_size))
                # MoE weight MAX uses the EP communicator, even under TP.
                patch.setattr(parallel_state, "_EP", _EmulatedRankGroup(rank, tp_size))
                torch.manual_seed(11)
                test(source, target, rank, tp_size, *options)

    return run


@contextmanager
def _emulate_max_reductions(expected_group, maxima):
    """Check each local contribution and add the peer's shard maximum."""
    pending = iter(maxima)

    def all_reduce(tensor, op, group):
        # Weight scales need a MAX on the intended TP/EP group, not a sum.
        assert op == torch.distributed.ReduceOp.MAX
        assert group is expected_group
        pair = next(pending, None)
        assert pair is not None, "Unexpected weight MAX reduction"
        local, peer = pair
        # Validate the actual local contribution before simulating the merge.
        torch.testing.assert_close(tensor, local, rtol=0, atol=0)
        tensor.copy_(torch.maximum(tensor, peer))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.distributed, "all_reduce", all_reduce)
        yield
    # A bypassed collective must fail even if both ranks happen to pick the same max.
    assert next(pending, None) is None, "Missing weight MAX reduction"


def _offline_quantize(weight, target):
    if target == "mxfp4":
        return mxfp4_quantize(weight)
    if target == "fp8_per_block":
        n, k = weight.shape
        tiles = weight.reshape(n // 128, 128, k // 128, 128)
        fp8_dtype = current_platform.fp8_dtype()
        scale = tiles.abs().amax((1, 3), keepdim=True).clamp_min(1e-4)
        scale = scale / torch.finfo(fp8_dtype).max
        return (tiles * scale.reciprocal()).to(fp8_dtype).reshape(n, k), scale[
            :, 0, :, 0
        ]
    # In contrast to the online path, the dynamic quantizer sees all TP shards.
    per_channel = target == "fp8_per_channel"
    if weight.ndim == 2:
        return ops.scaled_fp8_quant(weight, use_per_token_if_dynamic=per_channel)
    values, scales = zip(
        *(
            ops.scaled_fp8_quant(w, use_per_token_if_dynamic=per_channel)
            for w in weight
        ),
        strict=True,
    )
    scale_tensor = torch.stack(scales) if per_channel else torch.cat(scales)
    return torch.stack(values), scale_tensor


def _assert_requantized_weights(offline, online, target, *stems):
    from tests.quantization.reference_mxfp4 import fix_negative_zeros

    for stem in stems:
        expected = getattr(offline, f"{stem}weight")
        actual = getattr(online, f"{stem}weight")
        if target in ("mxfp4", "nvfp4_per_token", "nvfp4_weight_only"):
            actual = fix_negative_zeros(actual.view(torch.uint8))
            expected = fix_negative_zeros(expected.view(torch.uint8))
        elif expected.is_floating_point():
            actual, expected = actual.float(), expected.float()
        # Compare after kernel layout conversion, not just canonical packing.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        scale_name = "weight_scale_inv" if target == "fp8_per_block" else "weight_scale"
        online_scale_name = "scale" if target == "int8" else scale_name
        actual_scale = getattr(online, f"{stem}{online_scale_name}")
        expected_scale = getattr(offline, f"{stem}{scale_name}")
        if target == "mxfp4":
            # Preserve exact E8M0 scale encodings as well as the packed FP4 values.
            actual_scale = actual_scale.view(torch.uint8)
            expected_scale = expected_scale.view(torch.uint8)
        else:
            actual_scale, expected_scale = actual_scale.float(), expected_scale.float()
        if target == "int8":
            expected_scale = expected_scale.squeeze(-1)
        if target == "fp8_per_tensor" and not stem:
            # The checkpoint loader stores its single scale as a scalar.
            actual_scale = actual_scale.reshape(1)
            expected_scale = expected_scale.reshape(1)
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
        if target == "nvfp4_per_token":
            torch.testing.assert_close(
                getattr(online, f"{stem}weight_scale_2"),
                getattr(offline, f"{stem}weight_scale_2"),
                rtol=0,
                atol=0,
            )


def _linear_requantization_numerics(source, target, rank, tp_size):
    for projection, dim in (
        (RowParallelLinear, 1),
        (ColumnParallelLinear, 0),
        (ReplicatedLinear, None),
    ):
        _check_linear_requantization(source, target, rank, tp_size, projection, dim)


def _check_linear_requantization(source, target, rank, tp_size, projection, dim):
    n = k = 256
    dtype = torch.bfloat16
    source_config = (
        _fully_quantized_modelopt_mxfp8_config()
        if source == "mxfp8"
        else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    )
    # The portable packer can be tested even when no W4A16 GEMM is available.
    pack_only = target == "nvfp4_weight_only" and not is_fp4_marlin_supported()
    source_config.online_quantization_config = OnlineQuantizationConfig(
        quant_config_args(linear="fp8_per_tensor" if pack_only else target)
    )

    online = projection(k, n, bias=False, quant_config=source_config)
    # Global scales span either shard axis; channel scales span the input axis only.
    assert _is_tp_sharded(online) == (dim is not None and tp_size > 1)
    assert _is_tp_sharded(online, reduces_output_dim=False) == (
        dim == 1 and tp_size > 1
    )
    _materialize_requantization_source(online)
    if source == "mxfp8":
        weight = torch.randn(n, k).to(MXFP8_VALUE_DTYPE)
        scale = torch.randint(124, 131, (n, k // 32), dtype=MXFP8_SCALE_DTYPE)
        # Both row and column partitions must have distinct maxima.
        scale[:, k // 64 :] += 3
        scale[n // 2 :] += 3
        reference = weight.float() * (2.0 ** (scale.float() - 127)).repeat_interleave(
            32, dim=-1
        )
    else:
        global_scale = torch.tensor([97.0], dtype=torch.float32)
        weights = torch.randn(1, n, k)
        weights[..., k // 2 :] *= 2
        weights[:, n // 2 :] *= 2
        weight, scale, reference = (
            tensor[0] for tensor in _serialize_nvfp4_experts(weights, global_scale)
        )
        online.weight_scale_2.data.copy_(global_scale.reciprocal())

    def shard(tensor, shard_rank=rank):
        if dim is None:
            return tensor
        return tensor.chunk(tp_size, dim=dim)[shard_rank].contiguous()

    online.weight.data.copy_(shard(weight))
    online.weight_scale.data.copy_(shard(scale))
    method = online.quant_method
    assert isinstance(
        method,
        (
            Mxfp4OnlineLinearMethod,
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerTensorOnlineLinearMethod,
            Fp8PtpcOnlineLinearMethod,
            Nvfp4WeightOnlyOnlineLinearMethod,
        ),
    )
    actual = method.get_weight_for_quantization(online)
    # Check the FP32 intermediate before target rounding can hide a loss of precision.
    torch.testing.assert_close(actual, shard(reference), rtol=0, atol=0)

    maxima = []
    if tp_size == 2 and dim is not None:
        local, peer = (shard(reference, r).abs() for r in (rank, 1 - rank))
        if target in ("fp8_per_tensor", "nvfp4_weight_only"):
            maxima.append((local.amax().reshape(1), peer.amax().reshape(1)))
        elif target == "fp8_per_channel" and dim == 1:
            maxima.append(
                (local.amax(dim=-1, keepdim=True), peer.amax(dim=-1, keepdim=True))
            )
    with _emulate_max_reductions(parallel_state.get_tp_group().device_group, maxima):
        if pack_only:
            amax = amax_for_tp_weight_quant(
                actual.abs().amax().reshape(1), _is_tp_sharded(online)
            )
            packed, scales, global_scale = _quantize_nvfp4(actual, amax)
        else:
            method.process_weights_after_loading(online)

    if target == "fp8_per_block":
        # Requantization must replace the serialized source scales, not keep both.
        assert not hasattr(online, "weight_scale")
        assert online.weight_scale_inv.numel() > 0

    if target == "nvfp4_weight_only":
        g = (reference.abs().amax() / (6 * 448)).clamp_min(1e-12)
        q, s, _ = _serialize_nvfp4_experts(reference[None], g.reciprocal()[None])
        if pack_only:
            torch.testing.assert_close(packed, shard(q[0]), rtol=0, atol=0)
            torch.testing.assert_close(
                scales.float(), shard(s[0]).float(), rtol=0, atol=0
            )
            torch.testing.assert_close(global_scale, g, rtol=0, atol=0)
            return
        offline = projection(
            k,
            n,
            bias=False,
            quant_config=ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True, quant_method="W4A16_NVFP4"
            ),
        )
        offline.weight.data.copy_(shard(q[0]))
        offline.weight_scale.data.copy_(shard(s[0]))
        offline.weight_scale_2.data.fill_(g)
        offline.quant_method.process_weights_after_loading(offline)
        torch.testing.assert_close(
            online.weight_global_scale, offline.weight_global_scale, rtol=0, atol=0
        )
        _assert_requantized_weights(offline, online, target, "")
        return

    q, s = _offline_quantize(reference, target)
    offline = projection(k, n, bias=False)
    if target == "mxfp4":
        checkpoint = QuarkOCP_MX(
            weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp4Dynamic
        )
        s = shard(s)
    elif target == "fp8_per_channel":
        checkpoint = QuarkW8A8Fp8(kFp8StaticChannelSym, kFp8DynamicTokenSym)
        if dim == 0:
            s = shard(s)
        s = s.squeeze(-1)
    else:
        block = [128, 128] if target == "fp8_per_block" else None
        checkpoint = Fp8LinearMethod(Fp8Config(weight_block_size=block))
        if block:
            s = shard(s)
    local_q = shard(q)
    checkpoint.create_weights(
        layer=offline,
        output_partition_sizes=[local_q.shape[0]],
        input_size_per_partition=k // tp_size if dim == 1 else k,
        params_dtype=dtype,
        weight_loader=default_weight_loader,
        **({"input_size": k, "output_size": n} if target != "mxfp4" else {}),
    )
    offline.weight.data.copy_(local_q)
    scale_name = "weight_scale_inv" if target == "fp8_per_block" else "weight_scale"
    getattr(offline, scale_name).data.copy_(s)
    checkpoint.process_weights_after_loading(offline)
    _assert_requantized_weights(offline, online, target, "")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or current_platform.is_fp8_fnuz(),
    reason="Offline FP8 comparison uses the E4M3FN checkpoint format.",
)
@pytest.mark.parametrize("source", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize(
    "target",
    [
        "mxfp4",
        "fp8_per_tensor",
        "fp8_per_channel",
        "fp8_per_block",
        "nvfp4_weight_only",
    ],
)
def test_online_linear_requantization_numerics(run_numerics, source, target):
    run_numerics(_linear_requantization_numerics, source, target)


def _moe_source_method(source_format, scheme, layer):
    moe = layer.moe_config
    if scheme == "nvfp4":
        return _make_nvfp4_moe_source(source_format, moe)
    with skip_checkpoint_backend_selection():
        block = [128, 128] if scheme == "block" else None
        if source_format == "fp8":
            return Fp8MoEMethod(Fp8Config(weight_block_size=block), layer)
        if source_format == "quark":
            w_key, a_key = {
                "block": (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                "tensor": (kFp8StaticTensorSym, kFp8DynamicTensorSym),
                "channel": (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            }[scheme]
            return QuarkW8A8Fp8MoEMethod(moe, w_key, a_key)
        return CompressedTensorsW8A8Fp8MoEMethod(
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy(scheme),
                block_structure=block,
                dynamic=False,
            ),
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.FLOAT,
                strategy=(
                    QuantizationStrategy.TOKEN
                    if scheme == "channel"
                    else QuantizationStrategy.TENSOR
                ),
                dynamic=True,
            ),
            moe,
        )


def _fp8_moe_checkpoint_tensors(source_format, scheme, e, h, i, block, num_shards):
    tensors, reference = {}, []
    for stem, n, k in (("w13", num_shards * i, h), ("w2", h, i)):
        w = torch.randn(e, n, k).to(torch.float8_e4m3fn)
        if scheme == "block":
            bn, bk = block
            shards = num_shards if stem == "w13" else 1
            rows = cdiv(n // shards, bn) * shards
            s = torch.rand(e, rows, cdiv(k, bk), dtype=torch.float32) + 0.5
            dequantized = torch.cat(
                [
                    _naive_block_fp8_dequant(w_shard, s_shard, block)
                    for w_shard, s_shard in zip(
                        w.chunk(shards, dim=1), s.chunk(shards, dim=1), strict=True
                    )
                ],
                dim=1,
            )
        elif scheme == "channel":
            s = torch.rand(e, n, 1, dtype=torch.float32) + 0.5
            dequantized = w.float() * s
        elif stem == "w13":
            s = torch.rand(e, num_shards, dtype=torch.float32) + 0.5
            dequantized = w.float().unflatten(1, (num_shards, i)) * s[:, :, None, None]
            dequantized = dequantized.flatten(1, 2)
        else:
            s = torch.rand(e, dtype=torch.float32) + 0.5
            dequantized = w.float() * s[:, None, None]
        scale_name = (
            "weight_scale_inv"
            if source_format == "fp8" and scheme == "block"
            else "weight_scale"
        )
        tensors.update({f"{stem}_weight": w, f"{stem}_{scale_name}": s})
        reference.append(dequantized)
    return tensors, reference


def _nvfp4_target_checkpoint(weight):
    # Encode scales from the full experts, before any TP slicing.
    amax = weight.abs().flatten(1).amax(dim=1).clamp_min(1e-8)
    encode_scale = (6 * torch.finfo(torch.float8_e4m3fn).max) / amax
    q, s, _ = _serialize_nvfp4_experts(weight, encode_scale)
    return q, s, 1.0 / encode_scale


def _expert_output(inputs, w13, w2):
    gate, up = (inputs @ w13.transpose(-2, -1)).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w2.transpose(-2, -1)


_MOE_TARGETS = {
    "mxfp4": Mxfp4OnlineMoEMethod,
    "fp8_per_tensor": Fp8PerTensorOnlineMoEMethod,
    "fp8_per_channel": Fp8PtpcOnlineMoEMethod,
    "nvfp4_per_token": Nvfp4OnlineMoEMethod,
    "int8": Int8OnlineMoEMethod,
}


def _moe_offline_method(target, layer):
    moe = layer.moe_config
    if target == "mxfp4":
        return QuarkOCP_MX_MoEMethod(moe, kMxfp4Static, kMxfp4Dynamic)
    if target == "fp8_per_channel":
        return QuarkW8A8Fp8MoEMethod(moe, kFp8StaticChannelSym, kFp8DynamicTokenSym)
    if target == "int8":
        return QuarkW8A8Int8MoEMethod(moe, kInt8StaticChannelSym, kInt8DynamicTokenSym)
    if target == "nvfp4_per_token":
        config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        config.moe_activation_override = kNvfp4DynamicToken
        return ModelOptNvFp4FusedMoE(config, moe)
    return Fp8MoEMethod(Fp8Config(), layer)


def _make_moe_layer(prefix, e, h, i, backend, num_shards=2):
    layer = FusedMoEFactory(
        num_experts=e,
        top_k=2,
        hidden_size=h,
        intermediate_size=i,
        activation="silu" if num_shards == 2 else "silu_no_mul",
        ckpt_names=("gate_proj", "down_proj", "up_proj" if num_shards == 2 else None),
        prefix=f"{prefix}_{i}",
    ).routed_experts
    layer.moe_config.moe_backend = backend
    return layer


def _pad_expert_reference(w13, w2, i, dh, di, *, value=0.0):
    # Pad gate/up separately so the up projection lands after the new gate rows.
    return (
        torch.nn.functional.pad(
            w13.unflatten(1, (-1, i)), (0, dh, 0, di), value=value
        ).flatten(1, 2),
        torch.nn.functional.pad(w2, (0, di, 0, dh), value=value),
    )


def _check_moe_padding(method, layer, expected):
    expected_w13, expected_w2 = expected
    source = method.requantization_source
    dequantize = source.dequantize_weight
    source_chunks = []

    def record_source_chunks(layer):
        for chunk in dequantize(layer):
            source_chunks.append(chunk)
            yield chunk

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source, "dequantize_weight", record_source_chunks)
        w13, w2 = method.get_weights_for_quantization(layer)
    torch.testing.assert_close((w13, w2), expected, rtol=0, atol=0)
    ((_, source_w13, source_w2),) = source_chunks
    # For one unpadded chunk, reuse the dequantized tensors without a copy.
    assert w13 is source_w13 and w2 is source_w2

    config = layer.moe_config
    h, i = config.hidden_dim, config.intermediate_size_per_partition
    for dh, di, bias, error, match in (
        (32, 32, None, None, ""),
        (32, 32, "w13_bias", NotImplementedError, "biases is unsupported"),
        (32, 32, "w2_bias", NotImplementedError, "biases is unsupported"),
        (-32, 0, None, ValueError, "cannot shrink weights"),
        (0, -32, None, ValueError, "cannot shrink weights"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(config, "hidden_dim", h + dh)
            patch.setattr(config, "intermediate_size_per_partition", i + di)
            if bias is not None:
                patch.setattr(config, "has_bias", True)
                weight = w13 if bias == "w13_bias" else w2
                patch.setattr(
                    layer,
                    bias,
                    torch.nn.Parameter(weight.new_zeros(weight.shape[:2])),
                    raising=False,
                )
            if error is not None:
                # Reject invalid layouts instead of silently dropping weights or biases.
                with pytest.raises(error, match=match):
                    method.get_weights_for_quantization(layer)
                continue
            torch.testing.assert_close(
                method.get_weights_for_quantization(layer),
                _pad_expert_reference(expected_w13, expected_w2, i, dh, di),
                rtol=0,
                atol=0,
            )


def _moe_requantization_numerics(source, target, rank, tp_size, case, i=256):
    padding = case != "packing"
    e, h = (2, 128) if padding else (4, 256)
    num_shards = 1 if case == "padding-ungated" else 2
    dtype = torch.bfloat16
    source_format, scheme = source.split("/")
    partial_block = source == "fp8/block" and i % 128 != 0
    backend = {
        "mxfp4": "aiter" if on_gfx950() else "emulation",
        "nvfp4_per_token": "auto",
    }.get(target, "triton")

    def layer(prefix):
        return _make_moe_layer(prefix, e, h, i, backend, num_shards)

    def shard(tensor, stem, *, channel_scale=False, shard_rank=rank):
        if stem == "w13" and tensor.ndim == 3:
            shards = tensor.unflatten(1, (num_shards, -1)).chunk(tp_size, dim=2)
            return shards[shard_rank].flatten(1, 2)
        if stem == "w2" and tensor.ndim == 3 and not channel_scale:
            return tensor.chunk(tp_size, dim=-1)[shard_rank].contiguous()
        return tensor

    online = layer("online")
    # A passing comparison must actually use the requested number of logical shards.
    assert online.moe_config.tp_size == tp_size
    checkpoint = _moe_source_method(source_format, scheme, online)
    # Padding runs before backend packing, including for ungated layouts that some
    # backends cannot execute. Keep the real methods without selecting a GEMM.
    with skip_checkpoint_backend_selection() if padding else nullcontext():
        method = _MOE_TARGETS[target](moe=online.moe_config)
    method.set_requantization_source(checkpoint)
    sizes = dict(
        num_experts=e,
        hidden_size=h,
        intermediate_size_per_partition=i // tp_size,
        params_dtype=dtype,
        global_num_experts=e,
    )
    method.create_weights(layer=online, **sizes)
    block = [128, 128]
    if source == "fp8/block":
        assert isinstance(checkpoint, Fp8MoEMethod)
        assert checkpoint.scale_block_size is not None
        block = checkpoint.scale_block_size
    if partial_block:
        # The source uses a refined grid, but its layout must not configure the target.
        assert method.uses_meta_device
        assert online.weight_block_size != [128, 128]
        assert block != [128, 128], "The source should refine its block grid"
    _materialize_requantization_source(online)
    if partial_block:
        assert online.w13_weight.dtype == torch.float8_e4m3fn
    if scheme == "nvfp4":
        w1 = torch.randn(e, i, h)
        w3 = torch.randn_like(w1) if num_shards == 2 else None
        w2 = torch.randn(e, h, i)
        globals_ = torch.arange(e, dtype=torch.float32) + 97
        tensors, *references = _nvfp4_moe_checkpoint_tensors(
            source_format, w1, w3, w2, globals_, 2 * globals_, 3 * globals_
        )
    else:
        tensors, references = _fp8_moe_checkpoint_tensors(
            source_format, scheme, e, h, i, block, num_shards
        )
    for name, tensor in tensors.items():
        param = getattr(online, name)
        shard_tensor = shard(
            tensor,
            name.split("_")[0],
            channel_scale=scheme == "channel" and "scale" in name,
        )
        param.data.copy_(shard_tensor.reshape_as(param))

    if padding:
        # Compare staged FP32 weights with an independently padded reference.
        expected = tuple(
            shard(weight, stem)
            for stem, weight in zip(("w13", "w2"), references, strict=True)
        )
        _check_moe_padding(method, online, expected)
        return

    offline = layer("offline")
    offline_method = _moe_offline_method(target, offline)
    if target == "mxfp4":
        # Compare packed bytes only after both paths select the same layout.
        assert method.mxfp4_backend == offline_method.mxfp4_backend
    offline_method.create_weights(layer=offline, **sizes)
    reconstructed = method.get_weights_for_quantization(online)
    maxima = []
    requantized = []
    for stem, actual, reference in zip(
        ("w13", "w2"), reconstructed, references, strict=True
    ):
        local = shard(reference, stem)
        torch.testing.assert_close(actual, local, rtol=0, atol=0)
        if tp_size == 2 and target != "mxfp4":
            peer = shard(reference, stem, shard_rank=1 - rank)
            if target in ("fp8_per_tensor", "nvfp4_per_token"):
                maxima.append(
                    (local.abs().amax(dim=(1, 2)), peer.abs().amax(dim=(1, 2)))
                )
            elif stem == "w2":
                keepdim = target == "fp8_per_channel"
                maxima.append(
                    (
                        local.abs().amax(dim=-1, keepdim=keepdim),
                        peer.abs().amax(dim=-1, keepdim=keepdim),
                    )
                )
        if target == "nvfp4_per_token":
            q, s, g = _nvfp4_target_checkpoint(reference)
            if stem == "w13":
                g = g[:, None].expand(-1, 2)
            getattr(offline, f"{stem}_weight_scale_2").data.copy_(g)
        elif target == "int8":
            s = reference.abs().amax(dim=-1, keepdim=True) / 127
            q = (reference / s).round().to(torch.int8)
        else:
            q, s = _offline_quantize(reference, target)
            if target == "fp8_per_tensor":
                requantized.append(q.float() * s[:, None, None])
                if stem == "w13":
                    s = s[:, None].expand(-1, 2)
        getattr(offline, f"{stem}_weight").data.copy_(shard(q, stem))
        param = getattr(offline, f"{stem}_weight_scale")
        local_scale = shard(
            s, stem, channel_scale=target in ("fp8_per_channel", "int8")
        )
        param.data.copy_(local_scale.reshape_as(param))
    with _emulate_max_reductions(parallel_state.get_ep_group().device_group, maxima):
        method.process_weights_after_loading(online)
    offline_method.process_weights_after_loading(offline)
    _assert_requantized_weights(offline, online, target, "w13_", "w2_")
    if partial_block:
        # The target must not keep either of the old serialized FP8 scale tensors.
        assert not hasattr(online, "w13_weight_scale_inv")
        assert not hasattr(online, "w2_weight_scale_inv")

    if scheme == "nvfp4" and target == "fp8_per_tensor":
        inputs = torch.randn(e, 8, h, dtype=torch.float32) * 0.4
        expected = _expert_output(
            inputs, torch.cat((w1, w3), dim=1).float(), w2.float()
        )
        actual = _expert_output(inputs, *requantized)
        # Bound the compounded NVFP4 and FP8 error relative to the original experts.
        assert (actual - expected).norm() / expected.norm() < 0.2


def _check_uninitialized_padding(tp_size, backend):
    e, h, i = 4, 240, 208  # Neither logical dimension is MXFP4-block aligned.
    local_i = i // tp_size
    dtype = torch.bfloat16
    online = _make_moe_layer("online", e, h, i, backend)
    offline = _make_moe_layer("offline", e, h, i, backend)
    method = Mxfp4OnlineMoEMethod(moe=online.moe_config)
    checkpoint = _moe_offline_method("mxfp4", offline)
    # Packed weight equality requires the online and offline layouts to agree.
    assert method.mxfp4_backend == checkpoint.mxfp4_backend
    roundup = dict(
        hidden_size=h,
        intermediate_size_per_partition=local_i,
        act_dtype=dtype,
        moe_parallel_config=online.moe_config.moe_parallel_config,
    )
    padded_h, padded_i = method.maybe_roundup_sizes(**roundup)
    # Both methods must pad each dimension by the same amount.
    assert (padded_h, padded_i) == checkpoint.maybe_roundup_sizes(**roundup)
    assert padded_h > h and padded_i > local_i
    for layer in (online, offline):
        cfg = layer.moe_config
        # Keep the logical dimensions so the target can identify unloaded cells.
        assert (
            cfg.hidden_dim_unpadded,
            cfg.intermediate_size_per_partition_unpadded,
        ) == (h, local_i)
        cfg.hidden_dim = padded_h
        cfg.intermediate_size_per_partition = padded_i
    sizes = dict(
        num_experts=e,
        hidden_size=padded_h,
        intermediate_size_per_partition=padded_i,
        params_dtype=dtype,
    )
    method.create_weights(layer=online, **sizes)
    checkpoint.create_weights(layer=offline, **sizes)

    # Keep blocks at very different magnitudes so their scales differ too.
    amplitudes = torch.tensor(
        [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4], dtype=dtype
    )

    def weight(n, k):
        block_amplitudes = amplitudes[torch.arange(k) // 32 % len(amplitudes)]
        return torch.randn(e, n, k, dtype=dtype) * block_amplitudes

    logical = (weight(2 * local_i, h), weight(h, local_i))
    pad = (local_i, padded_h - h, padded_i - local_i)
    poisoned = _pad_expert_reference(*logical, *pad, value=float("nan"))
    zeros = _pad_expert_reference(*logical, *pad)
    for stem, dirty, reference in zip(("w13", "w2"), poisoned, zeros, strict=True):
        # Only the online buffer contains NaNs; the checkpoint assumes zero padding.
        replace_parameter(online, f"{stem}_weight", dirty)
        q, s = _offline_quantize(reference, "mxfp4")
        getattr(offline, f"{stem}_weight").data.copy_(q)
        getattr(offline, f"{stem}_weight_scale").data.copy_(s)
    method.process_weights_after_loading(online)
    checkpoint.process_weights_after_loading(offline)
    _assert_requantized_weights(offline, online, "mxfp4", "w13_", "w2_")


_MOE_SOURCES = (
    "fp8/block",
    "fp8/tensor",
    "compressed_tensors/block",
    "compressed_tensors/tensor",
    "compressed_tensors/channel",
    "quark/block",
    "quark/tensor",
    "quark/channel",
    "compressed_tensors/nvfp4",
    "modelopt/nvfp4",
    "quark/nvfp4",
)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm."
)
@pytest.mark.parametrize(
    "source,target,case",
    [
        # Keep the full source × target grid for serialized checkpoints.
        *(
            pytest.param(source, target, case, id=f"{case}-{target}-{source}")
            for source, target, case in product(
                _MOE_SOURCES,
                _MOE_TARGETS,
                ("packing", "padding-gated", "padding-ungated"),
            )
        ),
        # Poisoned storage is a separate BF16 online-quantization path.
        pytest.param("bf16", "mxfp4", "nan-padding-aiter"),
        pytest.param("bf16", "mxfp4", "nan-padding-emulation"),
    ],
)
def test_online_moe_requantization_numerics(run_numerics, source, target, case):
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
    from vllm.utils.import_utils import has_quark

    if case == "nan-padding-aiter" and not current_platform.is_rocm():
        pytest.skip("AITER MXFP4 is ROCm-only.")
    if source != "bf16" and current_platform.is_fp8_fnuz():
        pytest.skip("Offline FP8 comparisons use the E4M3FN checkpoint format.")
    if case == "packing" and target == "mxfp4" and not on_gfx950() and not has_quark():
        pytest.skip("MXFP4 MoE emulation requires Quark.")
    if target == "nvfp4_per_token" and not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and (case != "packing" or has_flashinfer_trtllm_fused_moe())
    ):
        pytest.skip("Per-token NVFP4 MoE requires Blackwell and FlashInfer TRTLLM.")
    run_numerics(_run_moe_with_backend, source, target, case)


def _run_moe_with_backend(source, target, rank, tp_size, case):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    use_aiter = case == "nan-padding-aiter" or (
        case == "packing" and target == "mxfp4" and on_gfx950()
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            if use_aiter:
                _enable_aiter_mxfp4_moe(patch)
            if source == "bf16":
                if not use_aiter:
                    # Exercise weights-only emulation without requiring Quark's runtime.
                    patch.setattr(
                        "vllm.model_executor.layers.fused_moe.experts."
                        "ocp_mx_emulation_moe.has_quark",
                        lambda: True,
                    )
                _check_uninitialized_padding(
                    tp_size, "aiter" if use_aiter else "emulation"
                )
                return
            if tp_size == 2 and case == "packing":
                # Exercise one-expert chunks in addition to the no-copy path.
                patch.setattr(fp8_utils, "_DEQUANT_CHUNK_BYTES", 1)
            _moe_requantization_numerics(source, target, rank, tp_size, case)
            if use_aiter and source == "fp8/block" and tp_size == 1:
                # The unaligned intermediate size requires a refined source grid.
                _moe_requantization_numerics(source, target, rank, tp_size, case, i=192)
    finally:
        if use_aiter:
            rocm_aiter_ops.refresh_env_variables()
