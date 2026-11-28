# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm FP8 dense linears: the Ulysses weight gather (gloo; one device
per rank) and, on gfx950, the online `fp8_per_channel` mapping, the GEMMs on BF16
and pre-quantized input, the fused-norm numerics and prompt logprobs against the
BF16 path."""

import json
import zlib

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    DeviceConfig,
    KernelConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    _fp8_channel_scale,
    _fp8_quant_per_channel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.models.muse_spark.amd import fp8_dense
from vllm.models.muse_spark.amd import ulysses_sp as usp
from vllm.models.muse_spark.common.layers import MuseSparkQKVOGateParallelLinear
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    fused_norm_residual_norm_quant_fwd,
    fused_rmsnorm_fwd,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import multi_gpu_marks, multi_gpu_test
from .test_muse_spark_rocm_mono import TINY_TEXT, _checkpoint_config
from .test_muse_spark_rocm_ulysses_sp import (
    LAYERS,
    ON,
    _check_reloads,
    _copies,
    _held_copies,
    _random_weights,
    _tiny_llm,
)

F8 = torch.float8_e4m3fn
FP8_DENSE = {"linear": "fp8_per_channel", "ignore": ["*o_proj"]}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("muse_spark")
    (path / "config.json").write_text(json.dumps(_checkpoint_config(TINY_TEXT)))
    return path


@pytest.fixture(scope="module")
def two_layer_checkpoint(tmp_path_factory):
    """Layer 0 with BF16 experts, layer 1 with Quark MXFP4 experts: deep enough for
    both expert kinds, shallow enough that random O(1) weights stay well
    conditioned."""
    layers = 2
    text = TINY_TEXT | {
        "num_hidden_layers": layers,
        **{
            key: TINY_TEXT[key][:layers]
            for key in ("layer_types", "layer_rope_theta", "mlp_layer_types")
        },
    }
    path = tmp_path_factory.mktemp("muse_spark_2l")
    (path / "config.json").write_text(json.dumps(_checkpoint_config(text)))
    return path


@pytest.fixture(scope="module")
def config(checkpoint) -> VllmConfig:
    return _config(checkpoint)


def _config(
    checkpoint=None,
    monokernel: bool = False,
    lora: bool = False,
    quantization_config: QuantizationConfigArgs | None = None,
) -> VllmConfig:
    model = {}
    if checkpoint is not None:
        model["model_config"] = ModelConfig(
            model=str(checkpoint),
            tokenizer_mode="skip",
            dtype="bfloat16",
            max_model_len=4096,
            language_model_only=True,
            enforce_eager=True,
            quantization_config=quantization_config,
        )
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        kernel_config=KernelConfig(monokernel=monokernel),
        lora_config=LoRAConfig() if lora else None,
        **model,
    )


class _Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def _fp8_linear(weight: torch.Tensor, config: VllmConfig) -> _Linear:
    """A layer as online `fp8_per_channel` leaves it: `weight` the `[K, N]` view of
    the `[N, K]` FP8 tensor, `weight_scale` `[N, 1]`."""
    scale = _fp8_channel_scale(weight_amax(weight, dim=-1, keepdim=True))
    layer = _Linear(_fp8_quant_per_channel(weight, scale).t())
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    with set_current_vllm_config(config):
        layer.quant_method = Fp8PtpcOnlineLinearMethod()
    return layer


def _gather_worker(
    rank: int, world_size: int, port: int, checkpoint, layout: str
) -> None:
    q, kv, k = 16, 8, 32

    def weights(seed: int, rows: int, cols: int) -> list[torch.Tensor]:
        return [
            torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed + r))
            .to(torch.bfloat16)
            .to(f"cuda:{rank}")
            for r in range(world_size)
        ]

    torch.accelerator.set_device_index(rank)
    shards = weights(0, 2 * q + 2 * kv, k)
    o_proj = _Linear(weights(100, k, q)[rank])
    with set_current_vllm_config(_config()):
        init_distributed_environment(
            world_size, rank, f"tcp://127.0.0.1:{port}", rank, backend="gloo"
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size, backend="gloo")
        config = _config(checkpoint)
        layer = _fp8_linear(shards[rank].clone(), config)
        if layout == "n_k":
            # AiterPerToken stores the `[N, K]` tensor itself.
            layer.weight = torch.nn.Parameter(
                layer.weight.data.t(), requires_grad=False
            )
            assert layer.weight.is_contiguous()
        own = layer.weight.data.clone()
        fp8 = usp.build_weights(layer, o_proj, q, kv, alias=True)
        bf16 = usp.build_weights(_Linear(shards[rank].clone()), o_proj, q, kv)

        def quantize(w: torch.Tensor):
            ref = _fp8_linear(w, config)
            return ref.weight.t(), ref.weight_scale.view(-1)

        # The gathered FP8 shards are the bytes that quantizing the gathered BF16
        # weight yields: rows are independent.
        assert fp8.qkv is None and fp8.gate is None and fp8.qkv_gate is not None
        w, s = fp8.qkv_gate
        ref_w, ref_s = quantize(torch.cat(shards))
        assert torch.equal(w.view(torch.uint8), ref_w.view(torch.uint8))
        assert s.shape == (1, ref_s.numel())
        assert torch.equal(s.view(-1), ref_s)
        # The layer's own weight is now the view of its rows in the fused copy, in
        # its original layout and with its original bytes.
        rows = shards[rank].shape[0]
        assert layer.weight.data_ptr() == w[rank * rows].data_ptr()
        assert layer.weight.shape == own.shape and layer.weight.stride() == own.stride()
        assert torch.equal(layer.weight.view(torch.uint8), own.view(torch.uint8))
        # O stays BF16 and is built as without FP8 dense.
        assert all(torch.equal(a, b) for a, b in zip(fp8.o, bf16.o))
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(2, marks=multi_gpu_marks(num_gpus=2)),
        pytest.param(4, marks=multi_gpu_marks(num_gpus=4)),
    ],
)
@pytest.mark.parametrize("layout", ["k_n", "n_k"])
def test_ulysses_gathers_the_fp8_qkv_shards_exactly(
    checkpoint, world_size: int, layout: str
) -> None:
    """The FP8 qkv layer has no BF16 weight, so Ulysses builds its full FP8 weights
    from the FP8 shards; they equal quantizing the gathered BF16. Both weight
    layouts: the `[K, N]` view (RowWise) and the `[N, K]` tensor (AiterPerToken)."""
    mp.spawn(
        _gather_worker,
        args=(world_size, get_open_port(), checkpoint, layout),
        nprocs=world_size,
        join=True,
    )


rocm_gfx950 = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark FP8 dense needs ROCm gfx950"
)


def _skip_unless_gfx950() -> None:
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("Muse Spark FP8 dense needs gfx950")


# Per-rank (N, K) of the TEP8 qkv-gate and of the unpooler (K = 4096).
DENSE_SHAPES = {
    "qkvg_tp8": (2304, 8192),
    "unpooler": (8192, 4096),
}


def _online_linear(
    config: VllmConfig, n: int, k: int, name: str, args: dict = FP8_DENSE
) -> tuple[OnlineQuantizationConfig, ReplicatedLinear]:
    from vllm.utils.torch_utils import set_default_torch_dtype

    online = OnlineQuantizationConfig(QuantizationConfigArgs(**args))
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = ReplicatedLinear(
            k, n, bias=False, quant_config=online, prefix=f"model.layers.0.{name}"
        )
    return online, layer


@pytest.fixture
def aiter_enabled(monkeypatch: pytest.MonkeyPatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@rocm_gfx950
@pytest.mark.parametrize("name", sorted(DENSE_SHAPES))
@pytest.mark.parametrize("m", [1, 257, 2048])
@pytest.mark.parametrize("fp8_rows", [False, True])
def test_dense_gemms_match_the_dequantized_reference(
    dist_init, aiter_enabled, config, name: str, m: int, fp8_rows: bool
) -> None:
    """The online FP8 dense layers on BF16 input and on the fused norms' per-token
    FP8 rows (`fp8_input`), against a float32 GEMM on the dequantized weight and
    activation. Shapes AITER has tuned run on its per-token GEMMs."""
    _skip_unless_gfx950()
    n, k = DENSE_SHAPES[name]
    _, layer = _online_linear(config, n, k, f"feed_forward.{name}")
    if fp8_rows and not fp8_dense.takes_fp8_input(layer):
        pytest.skip("the installed AITER has no per-token FP8 GEMM for this shape")
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, generator=gen, device="cuda").to(torch.bfloat16)
    layer.weight.weight_loader(layer.weight, w)
    # `[N, K]`, whichever layout the selected kernel keeps.
    wq = layer.weight if layer.weight.shape == (n, k) else layer.weight.t()
    w_dq = wq.float() * layer.weight_scale.view(-1, 1)
    x = torch.randn(m, k, generator=gen, device="cuda").to(torch.bfloat16)
    xq, xs = usp.quant_rows(x)
    ref = (xq.float() * xs.view(-1, 1)) @ w_dq.t()
    got = layer(
        fp8_dense.fp8_input(xq, xs.view(-1), torch.bfloat16) if fp8_rows else x
    )[0]
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    assert (got.float() - ref).norm() / ref.norm() < 5e-3


def _assert_quant_close(q, qs, ref_q, ref_s) -> None:
    """Per-token FP8 of the same BF16 rows: scales within the BF16 rounding,
    values within one FP8 code."""
    ref_s = ref_s.view(-1)
    torch.testing.assert_close(qs, ref_s, rtol=2**-7, atol=0)
    assert (q.float() * qs[:, None] - ref_q.float() * ref_s[:, None]).abs().max() <= (
        ref_s.max() * 448 * 2**-3
    )


@rocm_gfx950
def test_fused_norm_quant_matches_the_unfused_norm_then_quant() -> None:
    """`fused_rmsnorm_fwd` with `out_scale` gives the per-token FP8 and scales of
    the BF16 `RMSNorm` output that `quant_rows` would quantize."""
    _skip_unless_gfx950()
    from vllm.models.muse_spark.common.layers import MuseSparkRMSNorm

    gen = torch.Generator(device="cuda").manual_seed(0)
    d = 4096
    x = torch.randn(37, d, generator=gen, device="cuda").to(torch.bfloat16)
    norm = MuseSparkRMSNorm(d).cuda()
    torch.nn.init.normal_(norm.weight, generator=gen)
    q = torch.empty_like(x, dtype=F8)
    qs = torch.empty(37, device="cuda")
    fused_rmsnorm_fwd(
        x,
        norm.weight if norm.has_weight else None,
        norm.variance_epsilon,
        gain_center=norm.fused_gain_center,
        input_scale=norm.applies_weight_before_norm and norm.has_weight,
        out=q,
        out_scale=qs,
    )
    _assert_quant_close(q, qs, *usp.quant_rows(norm(x, out_dtype=torch.bfloat16)))


@rocm_gfx950
@pytest.mark.parametrize("gate_stream", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("store_out", [False, True])
def test_fused_boundary_quant_matches_the_boundary_then_quant(
    gate_stream: bool, per_channel: bool, store_out: bool
) -> None:
    """`fused_norm_residual_norm_quant_fwd` gives the stream and BF16 output of
    `fused_norm_residual_norm_fwd`, and the per-token FP8 of that output."""
    _skip_unless_gfx950()
    gen = torch.Generator(device="cuda").manual_seed(0)
    t, d = 37, 8192

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device="cuda")

    branch = randn(t, d).to(torch.bfloat16)
    stream = randn(t, d)
    width = d if per_channel else 1
    beta = randn(width).abs() + 0.5
    alpha = randn(width).abs() + 0.5 if gate_stream else 0.75
    kwargs = dict(
        alpha=alpha,
        beta=beta,
        post_gain_center=1.0,
        pre_gain_center=1.0,
        pre_input_scale=True,
    )
    args = (branch, stream, randn(d).to(torch.bfloat16), randn(d).to(torch.bfloat16))
    ref_out, ref_stream = fused_norm_residual_norm_fwd(
        *args, 1e-6, 1e-5, out_dtype=torch.bfloat16, **kwargs
    )
    out, new_stream, q, qs = fused_norm_residual_norm_quant_fwd(
        *args, 1e-6, 1e-5, out_dtype=torch.bfloat16 if store_out else None, **kwargs
    )
    torch.testing.assert_close(new_stream, ref_stream)
    if store_out:
        torch.testing.assert_close(out, ref_out)
    else:
        assert out is None
    _assert_quant_close(q, qs, *usp.quant_rows(ref_out))


def _online_linears(
    config: VllmConfig, args: dict, *names: str
) -> tuple[OnlineQuantizationConfig, list[ReplicatedLinear]]:
    online = OnlineQuantizationConfig(QuantizationConfigArgs(**args))
    with set_current_vllm_config(config), torch.device("cuda"):
        layers = [
            ReplicatedLinear(
                512,
                256,
                bias=False,
                quant_config=online,
                prefix=f"model.layers.0.{name}",
            )
            for name in names
        ]
    return online, layers


@rocm_gfx950
def test_online_fp8_per_channel_loads_the_fused_qkv_gate_layer(
    dist_init, config
) -> None:
    """`--quantization-config` `fp8_per_channel` applies to the fused QKV-gate
    layer: its `load_weights` goes through the online loader, which quantizes once
    every shard is loaded (and counts the layer in its summary), to the same bytes
    as quantizing the BF16 weight directly."""
    _skip_unless_gfx950()
    from vllm.utils.torch_utils import set_default_torch_dtype

    online = OnlineQuantizationConfig(QuantizationConfigArgs(**FP8_DENSE))
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        qkv = MuseSparkQKVOGateParallelLinear(
            512,
            64,
            4,
            2,
            quant_config=online,
            prefix="model.layers.0.self_attn.qkv_proj",
        )

    gen = torch.Generator(device="cuda").manual_seed(0)
    shards = {"q": 256, "k": 128, "v": 128, "o_gate": 256}
    w = torch.randn(768, 512, generator=gen, device="cuda").to(torch.bfloat16)
    for shard_id, rows in zip(shards, w.split(list(shards.values())), strict=True):
        rows.shard_id = shard_id
        list(qkv.load_weights([("weight", rows)]))
    ref = _fp8_linear(w.clone(), config)
    assert torch.equal(qkv.weight.view(torch.uint8), ref.weight.view(torch.uint8))
    assert torch.equal(qkv.weight_scale, ref.weight_scale)
    x = torch.randn(5, 512, generator=gen, device="cuda").to(torch.bfloat16)
    xq, xs = usp.quant_rows(x)
    ref_out = (xq.float() * xs) @ (ref.weight.float() * ref.weight_scale.t())
    assert (qkv(x)[0].float() - ref_out).norm() / ref_out.norm() < 5e-3


@rocm_gfx950
@pytest.mark.parametrize(
    "args, name, setup, error, match",
    [
        ({"linear": "fp8_per_block"}, "feed_forward.pooler", None,
         NotImplementedError, "fp8_per_channel"),
        (FP8_DENSE, "self_attn.qkv_proj", "monokernel", ValueError, "monokernel"),
        (FP8_DENSE, "self_attn.qkv_proj", "lora", ValueError, "LoRA"),
        (FP8_DENSE, "self_attn.qkv_proj", "int4", ValueError, "QUICK_REDUCE"),
    ],
)  # fmt: skip
def test_unsupported_online_fp8_dense_setups_are_refused(
    dist_init, checkpoint, config, monkeypatch, args, name, setup, error, match
) -> None:
    """Another online scheme, the monokernel (it reads the BF16 weights), LoRA
    (its wrappers do not take the fused norms' FP8 input) and INT4 quick
    all-reduce (collapses accuracy on top of FP8)."""
    _skip_unless_gfx950()
    if setup == "int4":
        monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    online, (layer,) = _online_linears(config, args, name)
    with (
        set_current_vllm_config(
            _config(checkpoint, monokernel=setup == "monokernel", lora=setup == "lora")
        ),
        pytest.raises(error, match=match),
    ):
        fp8_dense.use_online_fp8(layer, online)


@rocm_gfx950
def test_model_refuses_an_online_o_proj_with_ulysses(
    dist_init, two_layer_checkpoint, monkeypatch
) -> None:
    """The Muse Spark model refuses an online `o_proj` when the Ulysses prefill is
    on (its FP8 copy is built from the BF16 weight)."""
    _skip_unless_gfx950()
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.muse_spark.amd.multimodal import (
        MuseSparkForConditionalGeneration,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    # One rank never shards the stream, so Ulysses is switched on directly.
    monkeypatch.setattr(usp, "ulysses_prefill_enabled", lambda *args: True)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    config = _config(
        two_layer_checkpoint,
        quantization_config=QuantizationConfigArgs(linear="fp8_per_channel"),
    )
    try:
        with (
            set_current_vllm_config(config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
            pytest.raises(ValueError, match="ignore"),
        ):
            MuseSparkForConditionalGeneration(vllm_config=config)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def _random_weights_fp8(model: torch.nn.Module) -> int:
    """`_random_weights`, with each FP8 dense layer given the `fp8_per_channel`
    quantization of the BF16 weight `_random_weights` draws for it, and run on
    AITER's per-token FP8 kernel class, so the fused norms feed it per-token FP8
    rows."""
    from unittest import mock

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.kernels.linear import (
        AiterPerTokenFp8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import (
        expose_input_quant_key,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    built = _random_weights(model)
    rank = get_tensor_model_parallel_rank()
    for prefix, module in model.named_modules():
        if not fp8_dense.is_fp8(module):
            continue
        tag = rank if isinstance(module, ColumnParallelLinear) else -1
        seed = zlib.crc32(f"{prefix}.weight:{tag}".encode())
        gen = torch.Generator(device=module.weight.device).manual_seed(seed)
        weight = torch.empty(
            module.weight.t().shape, dtype=torch.bfloat16, device=module.weight.device
        )
        weight.normal_(generator=gen).mul_(weight.shape[-1] ** -0.5)
        scale = _fp8_channel_scale(weight_amax(weight, dim=-1, keepdim=True))
        module.weight.copy_(_fp8_quant_per_channel(weight, scale).t())
        module.weight_scale.copy_(scale)
        method = module.quant_method
        with (
            mock.patch.object(
                rocm_aiter_ops,
                "is_triton_gemm_a8w8_per_token_tuned",
                lambda *args: True,
            ),
            set_current_vllm_config(VllmConfig()),
        ):
            method.fp8_linear = AiterPerTokenFp8ScaledMMLinearKernel(
                method.fp8_linear.config, method.fp8_linear.layer_param_names
            )
        method.fp8_linear.process_weights_after_loading(module)
        expose_input_quant_key(module, method.fp8_linear)
    for layer in model.modules():
        if getattr(layer, "ulysses_weights", None):
            layer.build_ulysses_weights()
    return built


def _converted_layers(model: torch.nn.Module) -> list[str]:
    """The FP8 dense layers that take the fused norms' FP8 rows."""
    return sorted(
        name.split("layers.", 1)[1]
        for name, module in model.named_modules()
        if fp8_dense.is_fp8(module) and fp8_dense.takes_fp8_input(module)
    )


@rocm_gfx950
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("stream", ["replicated", "token_sharded", "ulysses"])
def test_fp8_dense_prompt_logprobs_stay_close_to_the_bf16_path(
    two_layer_checkpoint, monkeypatch, stream: str
) -> None:
    """Prompt logprobs of a two-layer Muse Spark with O(1) random weights at TEP2
    with the FP8 dense linears stay within the FP8 error of BF16: on a replicated
    stream, on a token-sharded stream (gathered FP8 activations) and with the
    Ulysses prefill gathering the FP8 shards."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    # `apply_model` sends the weight function to the TP workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if stream != "replicated":
        monkeypatch.setenv("VLLM_ROCM_MUSE_SP_MIN_TOKENS", "256")
    from vllm import LLM, SamplingParams

    _skip_unless_gfx950()
    prompt = list(range(3, 512)) * 2

    def prompt_logprobs(
        additional_config: dict, converted: list[str], **kwargs
    ) -> torch.Tensor:
        llm = LLM(
            model=str(two_layer_checkpoint),
            skip_tokenizer_init=True,
            language_model_only=True,
            load_format="dummy",
            dtype="bfloat16",
            max_model_len=4096,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enforce_eager=True,
            additional_config=additional_config,
            **kwargs,
        )
        llm.apply_model(_random_weights_fp8)
        assert llm.apply_model(_converted_layers) == [converted] * 2
        out = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0),
        )[0]
        values = [
            lp[tok].logprob
            for tok, lp in zip(prompt[1:], out.prompt_logprobs[1:], strict=True)
        ]
        del llm
        return torch.tensor(values)

    base = {"ulysses_prefill": "full"} if stream == "ulysses" else {}
    reference = prompt_logprobs(base, [])
    names = ["feed_forward.pooler", "feed_forward.unpooler", "self_attn.qkv_proj"]
    fp8_layers = sorted(f"{layer}.{name}" for layer in range(2) for name in names)
    fp8 = prompt_logprobs(base, fp8_layers, quantization_config=FP8_DENSE)
    delta = (reference - fp8).abs()
    assert delta.mean() < 0.06, delta.mean()


@rocm_gfx950
@multi_gpu_test(num_gpus=2)
def test_ulysses_prefill_follows_fp8_weight_reloads(tmp_path, monkeypatch) -> None:
    """With an online `fp8_per_channel` `qkv_proj`, a weight reload refills the
    fused Ulysses copy in place, it equals one built fresh, the `qkv_proj` weight
    stays the view of its rows in it, and the Ulysses step follows the weights."""
    llm = _tiny_llm(
        tmp_path,
        monkeypatch,
        ON,
        init=_random_weights_fp8,
        quantization_config=FP8_DENSE,
    )
    assert llm.apply_model(_held_copies) == [[_copies(ON, fp8=True)] * LAYERS] * 2
    _check_reloads(llm, ON, fp8=True)
