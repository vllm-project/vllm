# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft fusions against the unfused ops they replace.

- The context-KV CUDA kernel
  (vllm/models/minimax_m3/common/ops/dspark_context_kv.py) against flashinfer
  gemma_rmsnorm per layer, vllm._custom_ops.rotary_embedding (partial NeoX) and
  vllm._custom_ops.reshape_and_cache_flash into FlashAttention-layout caches
  [num_blocks, nkv, block_size, 2 * hd] viewed as
  kv_cache.transpose(1, 2).split(hd, dim=-1).
- The draft attention's fused q/k Gemma norm + RoPE against DFlash's q-norm,
  k-norm and RoPE.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm._custom_ops as ops
from vllm.models.minimax_m3.common.ops.dspark_context_kv import (
    DSparkContextKVWriter,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

flashinfer_norm = pytest.importorskip("flashinfer.norm")

HEAD_DIM = 128
ROTARY_DIM = 64
BLOCK_SIZE = 16
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def make_cos_sin_cache(max_pos, rotary_dim, dtype, device):
    inv_freq = 1.0 / (
        5_000_000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def make_slots(num_ctx, num_blocks, device):
    """Unique valid slots with ~1/4 padding (-1), at least one valid."""
    num_slots = num_blocks * BLOCK_SIZE
    slots = torch.randperm(num_slots, device=device)[:num_ctx].to(torch.int64)
    num_pad = max(1, num_ctx // 4)
    slots[:num_pad] = -1
    return slots


def build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed, unit_scale=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kv_cache_dtype == "fp8" and not unit_scale:
        dtype = torch.uint8
        scale = lambda: (  # noqa: E731
            torch.rand((), generator=g).to(device) * 1.5 + 0.25
        )
    else:
        dtype = torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
        scale = lambda: torch.ones((), device=device)  # noqa: E731
    layers = []
    for _ in range(L):
        layers.append(
            SimpleNamespace(
                kv_cache=torch.zeros(
                    num_blocks,
                    nkv,
                    BLOCK_SIZE,
                    2 * HEAD_DIM,
                    dtype=dtype,
                    device=device,
                ),
                _k_scale=scale(),
                _v_scale=scale(),
                kv_cache_dtype=kv_cache_dtype,
            )
        )
    return layers


def reference_update(
    all_kv, k_norm_w, positions, cos_sin, slot_mapping, layers, kv_cache_dtype
):
    """The unfused path, run on `layers`' caches."""
    num_ctx, L, _, nkv, hd = all_kv.shape
    all_k = torch.stack(
        [
            flashinfer_norm.gemma_rmsnorm(all_kv[:, layer, 0], k_norm_w[layer], EPS)
            for layer in range(L)
        ]
    )  # [L, num_ctx, nkv, hd]
    all_v = all_kv[:, :, 1]  # [num_ctx, L, nkv, hd]
    k_flat = all_k.view(L * num_ctx, nkv * hd)
    ops.rotary_embedding(
        positions.repeat(L),
        k_flat,
        None,
        hd,
        cos_sin.to(k_flat.dtype),
        True,
    )
    all_k = k_flat.view(L, num_ctx, nkv, hd)
    for layer, a in enumerate(layers):
        kc, vc = a.kv_cache.transpose(1, 2).split(hd, dim=-1)
        ops.reshape_and_cache_flash(
            all_k[layer],
            all_v[:, layer].contiguous(),
            kc,
            vc,
            slot_mapping,
            kv_cache_dtype,
            a._k_scale,
            a._v_scale,
        )


def assert_bf16_close(got, exp):
    diff = (got.view(torch.int16).int() - exp.view(torch.int16).int()).abs()
    assert int(diff.max()) <= 1, (
        f"bf16 cache differs by {int(diff.max())} ulp "
        f"in {int((diff > 0).sum())} elements"
    )


def assert_fp8_close(got, exp):
    equal = got == exp
    frac = equal.float().mean().item()
    assert frac >= 0.999, f"fp8 cache matches only {frac:.4%} of bytes"
    bad = ~equal
    if bad.any():
        byte_diff = (got[bad].int() - exp[bad].int()).abs()
        g = got[bad].view(torch.float8_e4m3fn).float()
        e = exp[bad].view(torch.float8_e4m3fn).float()
        ok = (byte_diff <= 1) | ((g == 0) & (e == 0))
        assert bool(ok.all()), (
            f"fp8 cache differs by more than 1 step in {int((~ok).sum())} bytes"
        )


def run_case(
    kv_cache_dtype, num_ctx, nkv, L, seed=0, rotary_dim=ROTARY_DIM, unit_scale=False
):
    torch.manual_seed(seed)
    device = "cuda"
    num_blocks = (num_ctx * 2 + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

    layers, ref_layers = (
        build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed, unit_scale)
        for _ in range(2)
    )

    all_kv = torch.randn(
        num_ctx, L, 2, nkv, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_norm_w = torch.randn(L, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.2
    positions = torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    cos_sin = make_cos_sin_cache(2048, rotary_dim, torch.bfloat16, device)
    slot_mapping = make_slots(num_ctx, num_blocks, device)

    writer = DSparkContextKVWriter(layers)
    writer(all_kv, k_norm_w, EPS, positions, cos_sin, rotary_dim, slot_mapping)
    reference_update(
        all_kv, k_norm_w, positions, cos_sin, slot_mapping, ref_layers, kv_cache_dtype
    )

    for a, b in zip(layers, ref_layers):
        if kv_cache_dtype == "fp8":
            assert_fp8_close(a.kv_cache, b.kv_cache)
        else:
            assert_bf16_close(a.kv_cache, b.kv_cache)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_ctx", [1, 7, 64, 300])
@pytest.mark.parametrize("nkv", [1, 2])
@pytest.mark.parametrize("L", [1, 5])
def test_dspark_context_kv(kv_cache_dtype, num_ctx, nkv, L):
    run_case(kv_cache_dtype, num_ctx, nkv, L)


@pytest.mark.parametrize("num_ctx", [7, 300])
def test_dspark_context_kv_fp8_unit_scale(num_ctx):
    """Unit k/v scales take the kernel's direct bf16 -> e4m3 conversion."""
    run_case("fp8", num_ctx, nkv=2, L=5, unit_scale=True)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("rotary_dim", [16, 32])
def test_dspark_context_kv_short_rope(kv_cache_dtype, rotary_dim):
    """Only some lanes rotate when rotary_dim < 64."""
    run_case(kv_cache_dtype, 64, nkv=2, L=2, rotary_dim=rotary_dim)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_dspark_context_kv_cuda_graph(kv_cache_dtype):
    torch.manual_seed(0)
    device = "cuda"
    num_ctx, nkv, L = 16, 1, 5
    num_blocks = 8

    layers = build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed=0)
    all_kv = torch.randn(
        num_ctx, L, 2, nkv, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_norm_w = torch.randn(L, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.2
    positions = torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    cos_sin = make_cos_sin_cache(2048, ROTARY_DIM, torch.bfloat16, device)
    slot_mapping = make_slots(num_ctx, num_blocks, device)

    writer = DSparkContextKVWriter(layers)
    writer._ensure_tables()  # build pointer tables outside the graph

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        writer(all_kv, k_norm_w, EPS, positions, cos_sin, ROTARY_DIM, slot_mapping)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        writer(all_kv, k_norm_w, EPS, positions, cos_sin, ROTARY_DIM, slot_mapping)

    # Change every input in place, replay, and compare against the unfused
    # reference run on the new inputs.
    all_kv.copy_(torch.randn_like(all_kv))
    positions.copy_(
        torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    )
    slot_mapping.copy_(make_slots(num_ctx, num_blocks, device))

    sentinel = 0x5A if kv_cache_dtype == "fp8" else 0x3F80
    for a in layers:
        a.kv_cache.fill_(sentinel)
    graph.replay()

    ref_layers = build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed=0)
    for a in ref_layers:
        a.kv_cache.fill_(sentinel)
    reference_update(
        all_kv, k_norm_w, positions, cos_sin, slot_mapping, ref_layers, kv_cache_dtype
    )

    torch.accelerator.synchronize()
    for a, b in zip(layers, ref_layers):
        if kv_cache_dtype == "fp8":
            assert_fp8_close(a.kv_cache, b.kv_cache)
        else:
            assert_bf16_close(a.kv_cache, b.kv_cache)


def test_dspark_context_kv_supported_by_cache_layout():
    """The writer engages on the [nb, nkv, bs, 2 * 128] page that FLASH_ATTN
    and FLASHINFER_TS share, whatever the backend, and not on a 5-D cache."""
    layers = build_layers("fp8", 5, 1, 4, "cuda", seed=0, unit_scale=True)
    assert DSparkContextKVWriter.supported(layers)
    layers[0].kv_cache = torch.zeros(
        4, 2, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    assert not DSparkContextKVWriter.supported(layers)


class _CaptureAttention(torch.nn.Module):
    """Stands in for the draft's Attention: records the q/k/v it is given."""

    _q_scale_float = 1.0

    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name

    def forward(self, q, k, v, output_dtype=None):
        self.qkv = tuple(None if t is None else t.clone() for t in (q, k, v))
        return torch.zeros_like(q, dtype=output_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_draft_attention_fused_qk_norm_rope(
    dist_init, default_vllm_config, kv_cache_dtype
):
    """One fused launch gives the q of DFlash's q-norm and RoPE and stores its
    k/v in a bf16 or fp8 cache as the layer's reshape_and_cache_flash would;
    the attention layer gets q alone."""
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention
    from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkAttention
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    torch.manual_seed(0)
    config = MiniMaxM3TextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=8192,
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=None,
            causal=False,
            cache_config=None,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
    for param in (attn.qkv_proj.weight, attn.q_norm.weight, attn.k_norm.weight):
        param.data.normal_(std=0.05)
    layer = attn.attn
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
    layer.kv_cache = torch.zeros(
        4, 2, BLOCK_SIZE, 2 * HEAD_DIM, dtype=cache_dtype, device="cuda"
    )
    attn.attn = _CaptureAttention(layer.layer_name)

    hidden = torch.randn(7, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 127, 128, 4000, 8190, 9], device="cuda")
    slots = torch.tensor([5, -1, 17, 18, 40, 63, 0], device="cuda")
    DFlashQwen3Attention.forward(attn, positions, hidden)
    q_ref, k_ref, v_ref = attn.attn.qkv
    assert k_ref is not None and v_ref is not None
    with set_forward_context(
        None, default_vllm_config, slot_mapping={layer.layer_name: slots}
    ):
        attn(positions, hidden)
    q, k, v = attn.attn.qkv
    torch.testing.assert_close(q, q_ref, atol=2e-2, rtol=2e-2)
    assert k is None and v is None
    expected = torch.zeros_like(layer.kv_cache)
    key_cache, value_cache = expected.transpose(1, 2).split(HEAD_DIM, dim=-1)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    ops.reshape_and_cache_flash(
        k_ref.reshape(7, 2, HEAD_DIM),
        v_ref.reshape(7, 2, HEAD_DIM),
        key_cache,
        value_cache,
        slots,
        kv_cache_dtype,
        one,
        one,
    )
    got, rtol = layer.kv_cache, 2e-2
    if kv_cache_dtype == "fp8":
        # k/v_ref come from DFlash's unfused kernels: allow one e4m3 step.
        got = got.view(torch.float8_e4m3fn).float()
        expected = expected.view(torch.float8_e4m3fn).float()
        rtol = 0.13
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=rtol)


class _FixedOutputAttention(torch.nn.Module):
    """Stands in for the draft's Attention: returns a fixed output in the
    requested dtype and records the dtypes asked for."""

    def __init__(self, layer_name, out):
        super().__init__()
        self.layer_name, self.out, self.output_dtypes = layer_name, out, []

    def forward(self, q, k, v, output_dtype):
        self.output_dtypes.append(output_dtype)
        return self.out.to(output_dtype)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Native MXFP8 kernels require SM100-family CUDA",
)
@torch.inference_mode()
def test_draft_fp16_attention_output_feeds_mxfp8_o_proj(dist_init, default_vllm_config):
    """A backend that writes an FP8 query's output as FP16 hands it to the
    MXFP8 o_proj without a conversion pass; o_proj quantizes it and still
    returns bf16, bit-identical to the bf16 path for values both dtypes hold."""
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkAttention
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    torch.manual_seed(0)
    config = MiniMaxM3TextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=8192,
    )
    quant_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "ignored_layers": [],
        }
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=None,
            causal=False,
            cache_config=None,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn",
            query_pre_quantized=True,
        )
    for proj in (attn.qkv_proj, attn.o_proj):
        weight, scale = _mxfp8_e4m3_quantize_torch(
            torch.randn(proj.weight.shape, device="cuda") / 16
        )
        proj.weight.data.copy_(weight)
        proj.weight_scale.data.copy_(scale)
        proj.quant_method.process_weights_after_loading(proj)
    layer = attn.attn
    layer.kv_cache = torch.zeros(
        4, 2, BLOCK_SIZE, 2 * HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    num_tokens = 7
    out = torch.randint(-256, 256, (num_tokens, attn.q_size), device="cuda") / 128
    attn.attn = _FixedOutputAttention(layer.layer_name, out.to(torch.bfloat16))

    hidden = torch.randn(
        num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(num_tokens, device="cuda")
    outputs = []
    for attn_output_dtype in (torch.float16, None):
        attn._attn_output_dtype = attn_output_dtype
        with set_forward_context(
            None, default_vllm_config, slot_mapping={layer.layer_name: positions}
        ):
            outputs.append(attn(positions, hidden))
    assert attn.attn.output_dtypes == [torch.float16, torch.bfloat16]
    assert outputs[0].dtype == torch.bfloat16
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
