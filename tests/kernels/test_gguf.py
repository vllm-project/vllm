from typing import List
from pathlib import Path

import pytest
import torch

from gguf import GGMLQuantizationType, dequantize, GGUFReader, ReaderTensor
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops


def get_gguf_sample_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> List[ReaderTensor]:
    # sample_dir = snapshot_download("Isotr0py/test-gguf-sample")
    sample_dir = "/mnt/g/libtorch_develop/ggml-scripts"
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.half]
HIDDEN_SIZES = [256, 1024]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPE = [
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.Q8_0,
]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPE)
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
                                     quant_type, *list(shape)).to(dtype)

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)
