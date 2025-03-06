
from pathlib import Path
from typing import List

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.gguf import _fused_moe_gguf
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts

def get_gguf_MOE_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> List[ReaderTensor]:
    sample_dir = "." 
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors

@torch.inference_mode()
def test_moe(num_tokens: int, hidden_size: int, dtype: torch.dtype,
             quant_type: GGMLQuantizationType):
    current_platform.seed_everything(0)

    x = torch.randn((num_tokens, 1024), dtype=dtype, device="cuda")
    # x = torch.rms_norm(x, (2048,))

    topk_weights = torch.rand(num_tokens, 8, device="cuda", dtype=dtype)
    topk_ids = torch.randint(0, 256, (num_tokens, 8), device="cuda")

    tensors = get_gguf_MOE_tensors(512, quant_type)
    w13 = tensors[0]
    w2 = tensors[1]

    w13_dequant = torch.tensor(dequantize(w13.data, quant_type),
                               device="cuda").to(dtype)#.transpose(1,2).contiguous()

    w2_dequant = torch.tensor(dequantize(w2.data, quant_type),
                               device="cuda").to(dtype)#.transpose(1,2).contiguous()
    act = SiluAndMul()

    output = _fused_moe_gguf(x, torch.tensor(w13.data, device="cuda"),
                             torch.tensor(w2.data, device="cuda"),
                             topk_weights, topk_ids,
                             quant_type, quant_type, act)

    # print("dequantized up", w13_dequant)
    # print("dequantized down", w2_dequant)

    ref_output = fused_experts(x, w13_dequant, w2_dequant, topk_weights, topk_ids).reshape(output.shape)
    print(output.shape)
    print(ref_output.shape)

    print(output)
    print(ref_output)
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)
test_moe(1, 512, torch.float16, GGMLQuantizationType.Q4_K)
