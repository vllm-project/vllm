
import os
os.environ['PYTORCH_GEOMETRIC_HOME'] = '/scratch/mc10322/.cache/pyg'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX'
os.environ['TORCHINDUCTOR_PATTERN_MATCH_DEBUG'] = '1'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_mc10322'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.debug = True
torch._inductor.config.compile_threads = 1
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.1+cu130
# torch cuda version: 13.0
# torch git version: 5811a8d7da873dd699ff6687092c225caffcf1bb


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025 
# Cuda compilation tools, release 13.0, V13.0.88 
# Build cuda_13.0.r13.0/compiler.36424714_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4070 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1):
        split = torch.ops.aten.split.Tensor(arg0_1, 256, -1);  arg0_1 = None
        getitem = split[0]
        getitem_1 = split[1];  split = None
        convert_element_type = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type)
        mul = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = sigmoid = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        empty = torch.ops.aten.empty.memory_format([4, 256], dtype = torch.float8_e4m3fn, device = device(type='cuda', index=0), pin_memory = False)
        empty_1 = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._C.per_token_group_fp8_quant.default, input = mul_1, group_size = 128, eps = 1e-10, fp8_min = -448.0, fp8_max = 448.0, scale_ue8m0 = False, _output_q_base_index = 0, _output_s_base_index = 1, _all_bases = [empty, empty_1]);  mul_1 = empty = empty_1 = None
        getitem_3 = auto_functionalized_v2[1]
        getitem_4 = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        return (getitem_3, getitem_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (4, 512), dtype=torch.float16, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)