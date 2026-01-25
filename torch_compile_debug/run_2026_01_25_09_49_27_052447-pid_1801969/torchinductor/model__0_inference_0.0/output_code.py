# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_mc10322/3z/c3zigxft5jlujsvd7a5jgitko5camhga6xsdmg2g4tdak6iuxm5z.py
# Topologically Sorted Source Nodes: [split, silu, silu_out, per_token_group_fp8_quant], Original ATen: [aten.split, aten.silu, aten.mul, _C.per_token_group_fp8_quant]
# Source node to ATen node mapping:
#   per_token_group_fp8_quant => per_token_group_fp8_quant_default
#   silu => convert_element_type, convert_element_type_1, mul, sigmoid
#   silu_out => mul_1
#   split => split
# Graph fragment:
#   %arg0_1 : Tensor "f16[4, 512][512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg0_1, 256, -1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %mul : Tensor "f32[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sigmoid), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.float16), kwargs = {})
#   %mul_1 : Tensor "f16[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %getitem_1), kwargs = {})
#   %per_token_group_fp8_quant_default : [num_users=0] = call_function[target=torch.ops._C.per_token_group_fp8_quant.default](args = (%mul_1, %empty, %empty_1, 128, 1e-10, -448.0, 448.0, False), kwargs = {})
#   return %buf2
triton_poi_fused_mul_per_token_group_fp8_quant_silu_split_0 = async_compile.triton('triton_poi_fused_mul_per_token_group_fp8_quant_silu_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_per_token_group_fp8_quant_silu_split_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4E08A71DC8DA2E51BC5316A41EC2A272D92A2108F426165E455EAC589509EFAC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_per_token_group_fp8_quant_silu_split_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (4, 512), (512, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((4, 256), (256, 1), torch.float8_e4m3fn)
            buf1 = empty_strided_cuda((4, 2), (2, 1), torch.float32)
            buf2 = empty_strided_cuda((4, 256), (256, 1), torch.float16)
            # Topologically Sorted Source Nodes: [split, silu, silu_out, per_token_group_fp8_quant], Original ATen: [aten.split, aten.silu, aten.mul, _C.per_token_group_fp8_quant]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_per_token_group_fp8_quant_silu_split_0.run(arg0_1, buf2, 1024, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [split, silu, silu_out, per_token_group_fp8_quant], Original ATen: [aten.split, aten.silu, aten.mul, _C.per_token_group_fp8_quant]
            torch.ops._C.per_token_group_fp8_quant.default(buf2, buf0, buf1, 128, 1e-10, -448.0, 448.0, False)
            del buf2
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
