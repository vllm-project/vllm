class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[4, 512]"):
         # File: /scratch/mc10322/vllm/test_silu_mul_pattern_debug.py:49 in test_function, code: gate, up = x.split(hidden, dim=-1)
        split = torch.ops.aten.split.Tensor(arg0_1, 256, -1);  arg0_1 = None
        getitem: "f16[4, 256]" = split[0]
        getitem_1: "f16[4, 256]" = split[1];  split = None
        
         # File: /scratch/mc10322/vllm/test_silu_mul_pattern_debug.py:50 in test_function, code: silu_out = F.silu(gate) * up
        convert_element_type: "f32[4, 256]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        sigmoid: "f32[4, 256]" = torch.ops.aten.sigmoid.default(convert_element_type)
        mul: "f32[4, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = sigmoid = None
        convert_element_type_1: "f16[4, 256]" = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None
        mul_1: "f16[4, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        
         # File: /scratch/mc10322/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py:903 in per_token_group_quant_fp8, code: x_q = torch.empty(x.shape, device=x.device, dtype=dtype)
        empty: "f8e4m3fn[4, 256]" = torch.ops.aten.empty.memory_format([4, 256], dtype = torch.float8_e4m3fn, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /scratch/mc10322/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py:911 in per_token_group_quant_fp8, code: x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
        empty_1: "f32[4, 2]" = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /scratch/mc10322/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py:916 in per_token_group_quant_fp8, code: torch.ops._C.per_token_group_fp8_quant(
        per_token_group_fp8_quant_default = torch.ops._C.per_token_group_fp8_quant.default(mul_1, empty, empty_1, 128, 1e-10, -448.0, 448.0, False);  mul_1 = per_token_group_fp8_quant_default = None
        return (empty, empty_1)
        