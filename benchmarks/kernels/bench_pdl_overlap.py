import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack
from torch.cuda import current_stream

from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _llvm.inline_asm(res=None, operands_=[ns.ir_value(loc=loc, ip=ip)],
        asm_string="nanosleep.u32 $0;", constraints="r",
        has_side_effects=True, loc=loc, ip=ip)

@cute.kernel
def producer_k(gOut: cute.Tensor, tail_ns: cutlass.Int32):
    tidx = cute.arch.thread_idx()[0]
    v = cutlass.Float32(1.0)
    v = v + cutlass.Float32(1.0)
    cute.arch.griddepcontrol_launch_dependents()
    nanosleep(tail_ns)
    if tidx == 0:
        gOut[0] = v

@cute.jit
def host_producer(gOut: cute.Tensor, tail_ns: cutlass.Int32, s: CUstream):
    producer_k(gOut, tail_ns).launch(
        grid=[1, 1, 1], block=[128, 1, 1], stream=s, use_pdl=True)

def bench_cg_n1(fn, n_retries=100):
    with torch.cuda.stream(torch.cuda.Stream()):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        ret = []
        for _ in range(n_retries):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ret.append(s.elapsed_time(e) * 1000)
        ret.sort()
        return ret[len(ret) // 2]


# Producer
buf = torch.empty(1, dtype=torch.float32, device="cuda")
bc = from_dlpack(buf, assumed_align=16).mark_layout_dynamic()
comp_p = cute.compile(host_producer, bc, 0,
                      CUstream(current_stream().cuda_stream))

print("Device:", torch.cuda.get_device_name())
print("Producer: 1 block, 128 threads | n_repeat=1, n_retries=100")
print()

SHAPES = [
    (7168, 256,   "router gate",      "router"),
    (7168, 2112,  "a_proj combined",   "a_gemm"),
    (7168, 576,   "kv_a_proj",         "a_gemm"),
    (7168, 1536,  "q_a_proj",          "a_gemm"),
    (1536, 3072,  "q_b_proj TP8",      "a_gemm"),
    (512,  4096,  "kv_b_proj TP8",     "a_gemm"),
]

TAILS = [0, 2000, 5000, 10000, 20000, 50000]
M = 16

# Producer solo times
prod_times = {}
for tns in TAILS:
    def sp(_t=tns):
        comp_p(bc, _t, CUstream(current_stream().cuda_stream))
    prod_times[tns] = bench_cg_n1(sp)

for K, N, label, ktype in SHAPES:
    print("=" * 95)
    print("M=%d K=%d N=%d — %s" % (M, K, N, label))

    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)

    kernels = {}

    if ktype == "router":
        ll_router_gemm(a, b); torch.cuda.synchronize()
        kernels['p-bf16'] = lambda: ll_router_gemm(a, b)
    else:
        ll_a_gemm(a, b); torch.cuda.synchronize()
        kernels['p-bf16'] = lambda: ll_a_gemm(a, b)

        ll_a_gemm(a8, b8, is_fp8=True); torch.cuda.synchronize()
        kernels['p-fp8'] = lambda: ll_a_gemm(a8, b8, is_fp8=True)

        try:
            ll_a_gemm_tma(a, b); torch.cuda.synchronize()
            kernels['t-bf16'] = lambda: ll_a_gemm_tma(a, b)
        except Exception as e:
            print("  TMA bf16 error: %s" % str(e)[:60])

        try:
            ll_a_gemm_tma(a8, b8, is_fp8=True); torch.cuda.synchronize()
            kernels['t-fp8'] = lambda: ll_a_gemm_tma(a8, b8, is_fp8=True)
        except Exception as e:
            print("  TMA fp8 error: %s" % str(e)[:60])

    # Solo times
    solos = {k: bench_cg_n1(fn) for k, fn in kernels.items()}
    solo_str = "  ".join("%s=%.2fus" % (k, v) for k, v in solos.items())
    print("  Solo: %s" % solo_str)
    print()

    # Header
    kcols = list(kernels.keys())
    hdr = "%8s | %5s |" % ("tail", "prod")
    for k in kcols:
        hdr += " %8s %5s |" % (k, "ovlp")
    hdr += " bf16-best fp8-best"
    print(hdr)
    print("-" * len(hdr))

    for tns in TAILS:
        pr = prod_times[tns]
        pairs = {}
        ovlps = {}

        for k, fn in kernels.items():
            def pair_fn(_t=tns, _fn=fn):
                comp_p(bc, _t, CUstream(current_stream().cuda_stream))
                _fn()
            pairs[k] = bench_cg_n1(pair_fn)
            ovlps[k] = pr + solos[k] - pairs[k]

        # Winners by dtype
        bf16_keys = [k for k in kcols if 'bf16' in k]
        fp8_keys = [k for k in kcols if 'fp8' in k]
        best_bf16 = min(bf16_keys, key=lambda k: pairs[k]) if bf16_keys else ""
        best_fp8 = min(fp8_keys, key=lambda k: pairs[k]) if fp8_keys else ""

        row = "%6dns | %4.1f |" % (tns, pr)
        for k in kcols:
            row += " %7.2fus %4.1f |" % (pairs[k], ovlps[k])
        row += " %-9s %s" % (best_bf16, best_fp8)
        print(row)

    print()
