import torch
import cutlass
from cutlass.epilogue import relu
from cutlass import Tensor as FakeTensor
from cutlass.utils.profiler import CUDAEventProfiler

# This controls whether ther C++ GEMM declaration will be printed at each step. Set to `false` to
# omit this information.
print_module = True

# The Epilogue Visitor feature currently only works for SM80 and 90
from cutlass.backend.utils.device import device_cc

if device_cc() not in [86, 80, 90]:
    import sys

    sys.exit()

m = 512
n = 512
k = 512

type_A = torch.float8_e4m3fn
type_B = torch.float8_e4m3fn
type_C = torch.bfloat16
type_D = torch.bfloat16


def to_fp8(tensor):
    # Assuming input tensor is float32
    # Scale tensor to range of FP8 E4M3 by clamping exponent and truncating mantissa
    max_exp = 2**4 - 1  # Maximum exponent for E4M3
    max_mantissa = 2**3 - 1  # Maximum mantissa for E4M3
    base = 2**max_exp
    # Scale the mantissa
    scaled = torch.clamp(tensor, -base, base)
    # Quantize the mantissa
    quantized = torch.round(scaled * max_mantissa) / max_mantissa
    return quantized.to(dtype=torch.float8_e4m3fn)


torch.manual_seed(2023)
tensor_A = to_fp8(torch.rand(size=(m, k), device="cuda"))
tensor_B = to_fp8(torch.rand(size=(n, k), device="cuda").t())
tensor_D = torch.zeros(size=(m, n), dtype=type_C, device="cuda")
tensor_C = torch.zeros(size=(m, n), dtype=type_C, device="cuda")

tensor_scale_a = torch.rand(size=(m, 1), device="cuda")
tensor_scale_b = torch.rand(size=(1, n), device="cuda")

plan = cutlass.op.Gemm(
    element_A=type_A,
    element_B=type_B,
    element_C=type_C,
    element_D=type_D,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.ColumnMajor,
    layout_C=cutlass.LayoutType.RowMajor,
    element_accumulator=torch.float32,
    kernel_cc=90,
)


# Define epilogue visitor
def example_epilogue(accum, scale_a, scale_b):
    D = scale_a * (scale_b * accum)
    return D


# Construct inputs and outputs
epilogue_tensors = {
    "accum": FakeTensor(
        element=torch.float32,
        shape=(m, n),
        layout_tag=cutlass.LayoutType.RowMajor,
    ),
    "D": tensor_D,
    "scale_a": tensor_scale_a,
    "scale_b": tensor_scale_b,
}

# Trace the epilogue visitor
epilogue_visitor = cutlass.epilogue.trace(example_epilogue, epilogue_tensors)

visitor_args = {"scale_a": tensor_scale_a, "scale_b": tensor_scale_b, "D": tensor_D}

plan.epilogue_visitor = epilogue_visitor
plan.run(
    tensor_A,
    tensor_B,
    tensor_C,
    tensor_D,
    visitor_args=visitor_args,
    print_module=print_module,
)


class TorchReference(torch.nn.Module):
    def forward(self, A, B, C, scale_a, scale_b):
        accum = torch.matmul(A.to(dtype=torch.float32), B.to(dtype=torch.float32))
        return example_epilogue(accum.to(dtype=torch.float32), scale_a, scale_b).to(
            type_D
        )


torch_reference = TorchReference()
tensor_D_ref = torch_reference(
    tensor_A, tensor_B, tensor_C, tensor_scale_a, tensor_scale_b
)

print(tensor_D)
print(tensor_D_ref)
assert torch.allclose(tensor_D, tensor_D_ref, 1e-1)

warmup_iterations = 10
profile_iterations = 50
# Profile CUTLASS fused kernel
duration = CUDAEventProfiler(
    plan,
    warmup_iterations,
    profile_iterations,
    tensor_A,
    tensor_B,
    tensor_C,
    tensor_D,
    visitor_args=visitor_args,
)()

print(f"CUTLASS duration: {duration:.2f} ms")
