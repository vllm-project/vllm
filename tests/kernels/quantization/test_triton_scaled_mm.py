# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_scaled_mm kernel.

Run `pytest tests/kernels/quantization/test_triton_scaled_mm.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

device = current_platform.device_type

triton_scaled_mm_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
)
triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
fp8_utils_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.utils.fp8_utils"
)


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def get_8bit_types():
    types = [torch.int8]
    if current_platform.supports_fp8():
        types.append(current_platform.fp8_dtype())
    return types


@pytest.fixture
def per_token_fp8_configs(monkeypatch):
    configs: dict[int, dict[str, int]] = {}
    monkeypatch.setattr(
        fp8_utils_module, "get_w8a8_per_token_fp8_configs", lambda *_args: configs
    )
    return configs


# This test is to check regressions for int8 support on ROCm.
@pytest.mark.parametrize(
    "model_path",
    [
        "neuralmagic/Llama-3.2-1B-quantized.w8a8",
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [10])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Should only run on ROCm")
def test_rocm_compressed_tensors_w8a8(
    vllm_runner, example_prompts, model_path, max_tokens, num_logprobs
):
    dtype = "bfloat16"

    with vllm_runner(model_path, dtype=dtype) as vllm_model:
        vllm_model.generate_greedy_logprobs(example_prompts, max_tokens, num_logprobs)


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    set_random_seed(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    c_check = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    c_actual = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)


# TD operand loads must be bit-exact vs the plain masked-load path.
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton scaled_mm runs on CUDA-alike or XPU.",
)
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]
)
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm_td_matches_plain(M, N, K, in_dtype, use_scalar_scale_a, use_bias):
    dev = current_platform.device_type
    out_dtype = torch.bfloat16
    set_random_seed(0)

    is_fp = torch.tensor([1, 1], dtype=in_dtype).is_floating_point()
    if is_fp:
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=dev)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=dev)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=dev)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=dev)

    scale_a = (
        torch.rand((1, 1), device=dev)
        if use_scalar_scale_a
        else 0.25 * torch.rand((M, 1), device=dev)
    )
    scale_b = 0.25 * torch.rand((N, 1), device=dev)
    bias = torch.rand((N,), device=dev, dtype=out_dtype) if use_bias else None

    out_plain = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=False)
    out_td = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=True)
    torch.testing.assert_close(out_td, out_plain, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Requires FP8")
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("rdna", [False, True])
@pytest.mark.parametrize("num_processors", [16, 32])
@pytest.mark.parametrize("cache_size_bytes", [32 * 1024**2, 64 * 1024**2])
def test_triton_fp8_per_token_backend_preserves_linear_shape(
    per_token_fp8_configs,
    monkeypatch,
    out_dtype,
    use_bias,
    tuned,
    rdna,
    num_processors,
    cache_size_bytes,
):
    """Select the backend and validate quantization, weight layout and 3D output."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonFp8PerTokenScaledMMKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    if current_platform.is_rocm():
        monkeypatch.setattr("vllm.platforms.rocm.on_rdna", lambda: rdna)
        monkeypatch.setattr(
            "vllm.utils.platform_utils.num_compute_units", lambda *_args: num_processors
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic.get_device_cache_size",
            lambda *_args: cache_size_bytes,
        )

    if tuned:
        per_token_fp8_configs[33] = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "num_warps": 4,
            "num_stages": 1,
        }
    set_random_seed(0)
    n, k = 256, 128
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="triton"))
    ):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            input_dtype=out_dtype,
            out_dtype=out_dtype,
            weight_shape=(n, k),
        )
    assert isinstance(kernel, TritonFp8PerTokenScaledMMKernel)
    assert kernel.use_heuristic == (current_platform.is_rocm() and rdna)
    assert kernel.num_processors == (num_processors if kernel.use_heuristic else 0)
    assert kernel.cache_size_bytes == (cache_size_bytes if kernel.use_heuristic else 0)
    if not tuned:

        def unexpected_lookup(*_args):
            pytest.fail("An untuned layer must skip runtime tuning lookup")

        monkeypatch.setattr(
            fp8_utils_module, "get_w8a8_per_token_fp8_config", unexpected_lookup
        )
    layer = torch.nn.Module()
    layer.weight = (
        (0.2 * torch.randn(n, k, device=device)).to(current_platform.fp8_dtype()).t()
    )
    layer.weight_scale = torch.rand(n, 1, device=device) + 0.5
    x = torch.randn(3, 11, k, device=device, dtype=out_dtype)
    bias = torch.randn(n, device=device, dtype=out_dtype) if use_bias else None
    x_q, x_s = kernel.quant_fp8(x.reshape(-1, k))
    expected = torch_scaled_mm(
        x_q, layer.weight, x_s, layer.weight_scale, out_dtype, bias
    ).view(3, 11, n)
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if out_dtype == torch.bfloat16 and not use_bias:
        qa = QuantizedActivation(x_q, x_s, x.dtype, x.shape, kFp8DynamicTokenSym)
        compiled = torch.compile(kernel.apply_weights, fullgraph=True)
        torch.testing.assert_close(
            compiled(layer, qa, bias), expected, rtol=1e-2, atol=1e-2
        )


@pytest.mark.skipif(
    not current_platform.supports_fp8() or not current_platform.is_cuda_alike(),
    reason="Requires FP8 and CUDA/ROCm graph support",
)
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("use_heuristic", [False, True])
def test_triton_fp8_per_token_dynamic_shapes_and_graph_replay(
    per_token_fp8_configs, tuned, use_heuristic
):
    """Compiled prefill/decode and graph replay must preserve current inputs."""
    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    tuned_config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "num_warps": 4,
        "num_stages": 1,
        "GROUP_M": 3,
    }
    if tuned:
        per_token_fp8_configs.update({1: tuned_config, 33: tuned_config})
    set_random_seed(0)
    n, k = 256, 128
    b = (0.2 * torch.randn(n, k, device=device)).to(current_platform.fp8_dtype()).t()
    sb = torch.rand(n, 1, device=device) + 0.5

    def run(a, sa):
        return torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func(
            a, b, sa, sb, torch.bfloat16, None, tuned, use_heuristic
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for m in (256, 1, 33):
        a = (0.2 * torch.randn(m, k, device=device)).to(current_platform.fp8_dtype())
        sa = torch.rand(m, 1, device=device) + 0.5
        expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
        actual = compiled(a, sa)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(a, sa)
    for _ in range(2):
        a.copy_((0.2 * torch.randn(m, k, device=device)).to(a.dtype))
        sa.mul_(1.25)
        graph.replay()
        torch.accelerator.synchronize()
        expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_fp8(),
    reason="Requires ROCm FP8",
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_triton_fp8_per_token_new_inputs_and_offset_storage(
    per_token_fp8_configs, out_dtype
):
    """Reused launches must read new allocations and unaligned tensor views."""
    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    m, n, k = 33, 256, 128
    for offset in (0, 1, 0, 1):
        a_store = (0.2 * torch.randn(m * k + offset, device=device)).to(
            current_platform.fp8_dtype()
        )
        w_store = (0.2 * torch.randn(n * k + offset, device=device)).to(a_store.dtype)
        a, b = a_store[offset:].view(m, k), w_store[offset:].view(n, k).t()
        sa = torch.rand(m + offset, device=device)[offset:].view(m, 1) + 0.5
        sb = torch.rand(n + offset, device=device)[offset:].view(n, 1) + 0.5
        bias = torch.randn(n, device=device, dtype=out_dtype)
        actual = torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func(
            a, b, sa, sb, out_dtype, bias, False, True
        )
        expected = torch_scaled_mm(a, b, sa, sb, out_dtype, bias)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_fp8(),
    reason="Requires ROCm FP8",
)
def test_triton_fp8_per_token_large_parent_storage(per_token_fp8_configs):
    """A narrow view of a large allocation must preserve launch correctness."""
    if torch.accelerator.get_memory_info()[0] < 6 * 1024**3:
        pytest.skip("Requires room for a storage larger than 2 GiB")
    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    m, n, k = 33, 256, 128
    b = (0.2 * torch.randn(n, k, device=device)).to(current_platform.fp8_dtype()).t()
    sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)
    for large_storage in (False, True, False):
        allocation = torch.empty(
            2**31 + m * k if large_storage else m * k,
            device=device,
            dtype=current_platform.fp8_dtype(),
        )
        a = allocation[-m * k :].view(m, k)
        a.copy_((0.2 * torch.randn(m, k, device=device)).to(a.dtype))
        actual = torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func(
            a, b, sa, sb, torch.bfloat16, None, False, True
        )
        torch.testing.assert_close(
            actual, torch_scaled_mm(a, b, sa, sb, torch.bfloat16), rtol=1e-2, atol=1e-2
        )


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_fp8(),
    reason="Requires ROCm FP8",
)
def test_triton_fp8_per_token_preserves_launch_hooks(per_token_fp8_configs):
    """Instrumentation added after warmup must still observe launches."""
    from vllm.triton_utils import triton

    knobs = triton.knobs

    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    m, n, k = 33, 256, 128
    a = (0.2 * torch.randn(m, k, device=device)).to(current_platform.fp8_dtype())
    b = (0.2 * torch.randn(n, k, device=device)).to(a.dtype).t()
    sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)

    def run():
        return torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func(
            a, b, sa, sb, torch.bfloat16, None, False, True
        )

    run()
    observed = []

    def observe(*_args, **_kwargs):
        observed.append(True)

    knobs.runtime.launch_enter_hook.add(observe)
    try:
        actual = run()
    finally:
        knobs.runtime.launch_enter_hook.remove(observe)
    assert observed
    torch.testing.assert_close(
        actual, torch_scaled_mm(a, b, sa, sb, torch.bfloat16), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("scale_ndim", [1, 2])
def test_scaled_mm_noncontiguous_scales_preserve_values(in_dtype, scale_ndim):
    """Strided scale views must use their logical values, including in the fallback."""
    set_random_seed(0)
    m, n, k = 33, 97, 128
    a = torch.randint(-4, 4, (m, k), device=device).to(in_dtype)
    b = torch.randint(-4, 4, (n, k), device=device).to(in_dtype).t()
    sa = torch.rand(2 * m, device=device)[1::2]
    sb = torch.rand(2 * n, device=device)[1::2]
    if scale_ndim == 2:
        sa, sb = sa[:, None], sb[:, None]
    assert not sa.is_contiguous() and not sb.is_contiguous()
    expected = torch_scaled_mm(
        a, b, sa.reshape(-1, 1), sb.reshape(-1, 1), torch.bfloat16
    )
    actual = triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Requires FP8")
@pytest.mark.parametrize(
    "variant",
    [
        "group_tail",
        "stripe_tail",
        "aligned",
        "m_tail",
        "n_tail",
        "k_tail",
        "padded",
        "unaligned",
    ],
)
def test_scaled_mm_launch_policy_handles_tile_and_layout_boundaries(variant):
    """Cover partial groups/stripes and mask incomplete operand tiles."""
    set_random_seed(0)
    m, n, k = 64, 160, 128
    group_m, n_stripe, unmasked = 1, 0, True
    if variant in ("group_tail", "stripe_tail"):
        m = 129
        group_m = 3 if variant == "group_tail" else 1
        n_stripe = 4 if variant == "stripe_tail" else 0
        unmasked = False
    elif variant == "m_tail":
        m += 1
    elif variant == "n_tail":
        n += 1
    elif variant == "k_tail":
        k += 16
    dtype = current_platform.fp8_dtype()
    a = torch.randint(-4, 4, (m, k), device=device).to(dtype)
    b = torch.randint(-4, 4, (n, k), device=device).to(dtype).t()
    if variant == "padded":
        padded_a = torch.empty(m, k + 16, device=device, dtype=dtype)
        padded_b = torch.empty(n, k + 16, device=device, dtype=dtype)
        padded_a[:, :k].copy_(a)
        padded_b[:, :k].copy_(b.t())
        a, b = padded_a[:, :k], padded_b[:, :k].t()
    elif variant == "unaligned":
        offset_a = torch.empty(m * k + 1, device=device, dtype=dtype)[1:].view(m, k)
        offset_b = torch.empty(n * k + 1, device=device, dtype=dtype)[1:].view(n, k)
        offset_a.copy_(a)
        offset_b.copy_(b.t())
        a, b = offset_a, offset_b.t()
    sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)
    actual = triton_scaled_mm(
        a,
        b,
        sa,
        sb,
        torch.bfloat16,
        use_heuristic=False,
        use_td=False,
        block_size_m=32,
        block_size_n=32,
        block_size_k=64,
        group_m=group_m,
        n_stripe=n_stripe,
        unmasked_loads=unmasked,
    )
    expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_fp8(),
    reason="Requires ROCm FP8",
)
@pytest.mark.parametrize("n,k", [(256, 8192), (3840, 2048), (2048, 4096), (4096, 4096)])
@pytest.mark.parametrize("num_processors", [16, 32])
@pytest.mark.parametrize("cache_size_bytes", [32 * 1024**2, 64 * 1024**2])
def test_triton_fp8_per_token_warmup_covers_unseen_token_counts(
    n, k, num_processors, cache_size_bytes, monkeypatch
):
    """Startup compilation must cover odd and aligned serving token counts."""
    from vllm.model_executor.layers.quantization.utils import (
        fp8_per_token_heuristic as heuristic,
    )
    from vllm.triton_utils import triton

    heuristic.fp8_per_token_warmup.warmup(
        n=n,
        k=k,
        out_dtype=torch.bfloat16,
        max_tokens=257,
        num_processors=num_processors,
        cache_size_bytes=cache_size_bytes,
    )
    misses = []
    monkeypatch.setattr(
        triton.knobs.runtime,
        "jit_cache_hook",
        lambda **kwargs: misses.append(kwargs),
    )
    monkeypatch.setattr(
        triton.knobs.runtime,
        "kernel_load_end_hook",
        lambda *args: misses.append(args),
    )
    for m in (1, 3, 15, 17, 33, 63, 65, 127, 129, 161, 191, 192, 193, 255, 257):
        a = (0.2 * torch.randn(m, k, device=device)).to(current_platform.fp8_dtype())
        b = (0.2 * torch.randn(n, k, device=device)).to(a.dtype).t()
        sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)
        out = heuristic.launch_fp8_per_token(
            a,
            b,
            sa,
            sb,
            torch.bfloat16,
            None,
            heuristic.get_heuristic_config(m, n, k, num_processors, cache_size_bytes),
        )
        torch.testing.assert_close(
            out,
            torch_scaled_mm(a, b, sa, sb, torch.bfloat16),
            rtol=1e-2,
            atol=1e-2,
        )
    assert not misses
