# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.gguf import _fused_moe_gguf
from vllm.platforms import current_platform

GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")


def get_gguf_sample_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


def get_gguf_MoE_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE_MOE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.half, torch.bfloat16, torch.float32]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 83, 128, 2048]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    # k-quants
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize(hidden_size: int, dtype: torch.dtype,
                    quant_type: GGMLQuantizationType):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(dequantize(tensor.data, quant_type),
                                  device="cuda").to(dtype)
        output = ops.ggml_dequantize(torch.tensor(tensor.data, device="cuda"),
                                     quant_type, *list(shape), dtype)

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq(hidden_size: int, dtype: torch.dtype,
              quant_type: GGMLQuantizationType):
    current_platform.seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((1, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type),
                              device="cuda").to(dtype)
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ops.ggml_mul_mat_vec_a8(qweight, x, quant_type,
                                         qweight.shape[0]).to(dtype)

        torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type",
    [
        # k-quants
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        # standard quants
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q8_0,
    ])
@torch.inference_mode()
def test_mmq(num_tokens: int, hidden_size: int, dtype: torch.dtype,
             quant_type: GGMLQuantizationType):
    current_platform.seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type),
                              device="cuda").to(dtype)
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ops.ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(output,
                                   ref_output,
                                   atol=atols[dtype],
                                   rtol=rtols[dtype])


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_moe(num_tokens: int, hidden_size: int, dtype: torch.dtype,
             quant_type: GGMLQuantizationType, top_k: int):
    current_platform.seed_everything(0)
    H, E = 1024, 256

    x = torch.rand((num_tokens, H), dtype=dtype, device="cuda")

    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=dtype)
    topk_ids = torch.randint(0,
                             E, (num_tokens, top_k),
                             device="cuda",
                             dtype=torch.int32)

    tensors = get_gguf_MoE_tensors(hidden_size, quant_type)

    w13 = tensors[0]
    w2 = tensors[1]

    w13_dequant = torch.tensor(dequantize(w13.data, quant_type),
                               device="cuda").to(dtype)

    w2_dequant = torch.tensor(dequantize(w2.data, quant_type),
                              device="cuda").to(dtype)
    act = SiluAndMul()

    output = _fused_moe_gguf(x, torch.tensor(w13.data, device="cuda"),
                             torch.tensor(w2.data,
                                          device="cuda"), topk_weights,
                             topk_ids, quant_type, quant_type, act)

    ref_output = fused_experts(x, w13_dequant, w2_dequant, topk_weights,
                               topk_ids).reshape(output.shape)
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)
