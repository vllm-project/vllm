# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.model_executor.layers.mamba.gdn.input_projection import gdn_input_gemms
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Flashinfer FP8 gemms requires compute capability of 10.0 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
) -> None:
    set_random_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((n, k), dtype=dtype, device=device) / k

    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)

    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=dtype)

    if use_bias:
        bias = torch.randn((n,), dtype=dtype, device=device)
        expected_out = expected_out + bias
    else:
        bias = None

    import flashinfer

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp8_mm(
            a_fp8,
            b_fp8.t(),
            a_scale,
            b_scale,
            dtype,
            bias=bias,
        )

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "m, breakable",
    [(m, False) for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]] + [(8, True)],
)
@pytest.mark.parametrize("scalar_shape", [(), (1,)])
@torch.inference_mode()
def test_gdn_concurrent_fp8_pair_graph_replay(m, breakable, scalar_shape):
    """Check scratch reuse and replay with concurrent and breakable captures."""
    device = "cuda:0"
    set_random_seed(42)
    x = torch.randn(m, 512, device=device).to(torch.float8_e4m3fn)
    weights = [
        torch.randn(n, 512, device=device).to(torch.float8_e4m3fn).t()
        for n in (256, 32)
    ]
    scales = [
        torch.tensor(s, device=device).reshape(scalar_shape)
        for s in (0.101, 0.271, 0.183)
    ]
    args = (x, *weights, *scales)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gdn_input_gemms(*args)
    torch.accelerator.synchronize()

    graph = BreakableCUDAGraphCapture() if breakable else torch.cuda.CUDAGraph()
    capture = graph if breakable else torch.cuda.graph(graph, stream=stream)
    with torch.cuda.stream(stream), capture:
        first = gdn_input_gemms(*args)
        second = gdn_input_gemms(*args)

    for factor in (1.0, -0.75, 2.0):
        x.copy_((torch.randn_like(x, dtype=torch.float32) * factor).to(x.dtype))
        scales[0].fill_(0.101 * abs(factor))
        for output in (*first, *second):
            output.fill_(float("nan"))
        graph.replay()
        for outputs in (first, second):
            for output, weight, weight_scale in zip(outputs, weights, scales[1:]):
                expected = (x.float() @ weight.float()) * scales[0] * weight_scale
                assert output.is_contiguous()
                assert output.dtype == torch.bfloat16
                torch.testing.assert_close(
                    output, expected.to(output.dtype), atol=1e-2, rtol=1e-2
                )

    if m == 8:
        snapshots = [tensor.clone() for tensor in args]
        # SchemaCheckMode's allclose does not support FP8 on CUDA. Check input
        # immutability byte-for-byte and exercise FakeTensor separately.
        torch.library.opcheck(gdn_input_gemms, args, test_utils=("test_faketensor",))
        compiled = torch.compile(gdn_input_gemms, backend="aot_eager", fullgraph=True)
        for actual, expected in zip(compiled(*args), gdn_input_gemms(*args)):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(args, snapshots):
            torch.testing.assert_close(
                actual.reshape(-1).view(torch.uint8),
                expected.reshape(-1).view(torch.uint8),
                atol=0,
                rtol=0,
            )
