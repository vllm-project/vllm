import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_token_quant 

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


#@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
#@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
#@pytest.mark.parametrize("dtype", DTYPES)
#@pytest.mark.parametrize("seed", SEEDS)
#@torch.inference_mode()
#def test_dynamic_per_token_fp8_quant(num_tokens: int, hidden_size: int,
#                                  dtype: torch.dtype, seed: int) -> None:
#    torch.random.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#
#    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
#
#    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.float8_e4m3fn)
#    ops_out, ops_scales = ops.dynamic_per_token_scaled_fp8_quant(x)
#
#    #ref_out_flt = ref_out.to(dtype=torch.float32)
#    #ops_out_flt = ops_out.to(dtype=torch.float32)
#    #for i in range(num_tokens):
#    #    for j in range(hidden_size):
#    #        if not torch.allclose(ref_out_flt[i][j], ops_out_flt[i][j]):
#    #            print (f"first error at token {i} - col {j}")
#    #            assert False
#
#    #torch.set_printoptions(profile="full")
#    #idx = 522 
#    #print (f"ref out {ref_out[idx].to(dtype=torch.float32)}")
#    #print (f"ops out {ops_out[idx].to(dtype=torch.float32)}")
#    #print (f"ref scales : {ref_scales[idx].item()}")
#    #print(f"ops scales : {ops_scales[idx].item()}")
#
#
#    assert torch.allclose(ref_scales, ops_scales)
#    assert torch.allclose(ref_out.to(dtype=torch.float32),
#                          ops_out.to(dtype=torch.float32))

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    fp8_traits = torch.finfo(torch.float8_e4m3fn)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    # reference
    x_max = x.abs().max().to(dtype=torch.float32) 
    fp8_max = torch.as_tensor([fp8_traits.max], dtype=torch.float32, device='cuda') 
    ref_scale = x_max / fp8_max 
    ref_iscale = torch.as_tensor([1.0], dtype=torch.float32, device='cuda') / ref_scale 
    ref_out = (x.to(dtype=torch.float32) * ref_iscale).clamp(fp8_traits.min, fp8_traits.max).to(dtype=torch.float8_e4m3fn)
    # kernel
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    assert torch.allclose(ref_scale, ops_scale)
    assert torch.allclose(ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32))
