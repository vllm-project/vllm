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


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.float8_e4m3fn)
    ops_out, ops_scales = ops.dynamic_per_token_scaled_fp8_quant(x)

    #ref_out_flt = ref_out.to(dtype=torch.float32)
    #ops_out_flt = ops_out.to(dtype=torch.float32)
    #for i in range(num_tokens):
    #    for j in range(hidden_size):
    #        if not torch.allclose(ref_out_flt[i][j], ops_out_flt[i][j]):
    #            print (f"first error at token {i} - col {j}")
    #            assert False

    #torch.set_printoptions(profile="full")
    #idx = 522 
    #print (f"ref out {ref_out[idx].to(dtype=torch.float32)}")
    #print (f"ops out {ops_out[idx].to(dtype=torch.float32)}")
    #print (f"ref scales : {ref_scales[idx].item()}")
    #print(f"ops scales : {ops_scales[idx].item()}")


    assert torch.allclose(ref_scales, ops_scales)
    assert torch.allclose(ref_out.to(dtype=torch.float32),
                          ops_out.to(dtype=torch.float32))

#@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
#@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
#@pytest.mark.parametrize("dtype", DTYPES)
#@pytest.mark.parametrize("seed", SEEDS)
#@torch.inference_mode()
#def test_dynamic_per_tensor_fp8_quant(num_tokens: int, hidden_size: int,
#                                  dtype: torch.dtype, seed: int) -> None:
#    torch.random.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#
#    fp8_traits = torch.finfo(torch.float8_e4m3fn)
#
#    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
#
#    # reference
#    ref_scale = x.abs().max().to(dtype=torch.float32) / float(fp8_traits.max)
#    assert ref_scale.dtype == torch.float32
#    ref_out = (x.to(dtype=torch.float32) / ref_scale).clamp(fp8_traits.min, fp8_traits.max).to(dtype=torch.float8_e4m3fn)
#    # kernel
#    assert x.dtype == dtype
#    ops_out, ops_scale = ops.scaled_fp8_quant(x)
#    assert ops_out.dtype == torch.float8_e4m3fn
#
#    assert torch.allclose(ref_scale, ops_scale)
#    # TODO (varun) : For some test cases, the computed scale in the kernel is different
#    # from the reference implementation in the 8th/9th digits. example, 
#    # ref_scales : 0.002223423682153225
#    # ops_scales : 0.0022234234493225813
#    # This precludes an exact match in the outputs. This needs to be investigated further.
#    assert torch.allclose(ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32),
#            atol=1)
